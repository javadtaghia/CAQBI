import os
import re
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from openai import OpenAI


TARGET_WORD_DEFAULT = "BLACK"

CONTEXTS = {
    "free":   "No special context. Just free association.",
    "color":  "In the context of COLORS and DESIGN.",
    "people": "In the context of PEOPLE and SOCIETY.",
    "music":  "In the context of MUSIC.",
    "news":   "In the context of NEWS TOPICS.",
}

PROMPT_TEMPLATE = """You are running a cognitive free-association task for research.

Rules:
- Produce exactly {n_words} single-word associations to the stimulus: {target}
- Output as a comma-separated list
- Single words only (no spaces, no hyphens, no numbers)
- No explanations, no extra text

Context: {context}

Stimulus: {target}
Output:"""

_WORD_RE = re.compile(r"^[a-zA-Z]+$")


def normalize_token(tok: str) -> Optional[str]:
    tok = tok.strip().lower()
    tok = re.sub(r"[^a-zA-Z]", "", tok)
    if not tok:
        return None
    if not _WORD_RE.match(tok):
        return None
    return tok


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "unknown"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def estimate_eta(start_time: float, done: int, total: int) -> Optional[float]:
    if done <= 0:
        return None
    elapsed = time.time() - start_time
    if elapsed <= 0:
        return None
    rate = done / elapsed
    if rate <= 0:
        return None
    remaining = total - done
    if remaining <= 0:
        return 0.0
    return remaining / rate


def maybe_report_progress(
    stage_label: str,
    done: int,
    total: int,
    stage_start: float,
    overall_done: int,
    overall_total: int,
    overall_start: float,
    last_report_time: float,
    min_interval: float = 10.0,
    min_step: int = 25,
) -> float:
    now = time.time()
    if total <= 0:
        return last_report_time
    if done <= 0:
        return last_report_time
    if done == total or done % min_step == 0 or (now - last_report_time) >= min_interval:
        stage_eta = estimate_eta(stage_start, done, total)
        overall_eta = estimate_eta(overall_start, overall_done, overall_total)
        stage_rate = done / max(1e-6, now - stage_start)
        print(
            f"{stage_label}: {done}/{total} ({done/total:.1%}), "
            f"{stage_rate:.2f} calls/s, eta {format_duration(stage_eta)}, "
            f"overall eta {format_duration(overall_eta)}"
        )
        return now
    return last_report_time


def parse_assoc_list(text: str, n_words: int) -> List[str]:
    raw = re.split(r"[,;\n]+", text.strip())
    toks = []
    for r in raw:
        t = normalize_token(r)
        if t:
            toks.append(t)
    return toks[:n_words]


def rank_weighted_scores(trials: List[List[str]], target: str) -> Dict[str, float]:
    scores = {}
    T = max(1, len(trials))
    for words in trials:
        for i, w in enumerate(words):
            if w == target:
                continue
            scores[w] = scores.get(w, 0.0) + 1.0 / (i + 1)
    for w in list(scores.keys()):
        scores[w] /= T
    return scores


def call_association(
    client: OpenAI,
    model: str,
    target: str,
    context_text: str,
    n_words: int,
    temperature: float,
    max_retries: int = 8,
    base_sleep: float = 1.5,
) -> str:
    prompt = PROMPT_TEMPLATE.format(n_words=n_words, target=target, context=context_text)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                #reasoning={"effort":"none"},
                temperature=temperature,
            )
            return resp.output_text
        except Exception as e:
            last_err = e
            sleep_s = min(30.0, base_sleep * (2 ** attempt)) * (0.6 + random.random() * 0.8)
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


@dataclass
class TrialRecord:
    trial_id: int
    model: str
    stimulus: str
    context: str
    temperature: float
    n_words: int
    prompt: str
    raw_text: str
    tokens: List[str]
    stage: str  # "stage1" or "stage2"


def plot_two_hop_graph(G: nx.DiGraph, outpath: str, title: str, max_label_nodes: int = 60):
    plt.figure(figsize=(13, 10))
    pos = nx.spring_layout(G, seed=7, k=0.75)

    # edge widths from weights
    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    wmax = max(weights) if weights else 1.0
    widths = [0.7 + 4.3 * (w / wmax) for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=240)
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowsize=16, alpha=0.65)

    # label: always label stimulus, label top-degree nodes
    deg = dict(G.degree())
    label_nodes = sorted(deg.keys(), key=lambda n: deg[n], reverse=True)[:max_label_nodes]
    if "black" not in label_nodes:
        label_nodes.append("black")
    nx.draw_networkx_labels(G, pos, labels={n: n for n in label_nodes}, font_size=9)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt-5-mini")
    ap.add_argument("--target", type=str, default=TARGET_WORD_DEFAULT)
    ap.add_argument("--n_words", type=int, default=25)

    # Stage 1
    ap.add_argument("--total_calls", type=int, default=1000)
    ap.add_argument("--temps", type=str, default="0.2,0.7,1.0")
    ap.add_argument("--contexts", type=str, default="free,color,people,music,news")

    # Stage 2 (two-hop)
    ap.add_argument("--twohop_topk", type=int, default=30)
    ap.add_argument("--twohop_trials_per_neighbor", type=int, default=20)
    ap.add_argument("--twohop_context", type=str, default="free")
    ap.add_argument("--twohop_temperature", type=float, default=0.7)
    ap.add_argument("--twohop_min_edge_weight", type=int, default=3)

    ap.add_argument("--outdir", type=str, default="aga_run_twohop")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    stage1_path = os.path.join(args.outdir, "raw_trials_stage1.jsonl")
    stage2_path = os.path.join(args.outdir, "raw_trials_stage2.jsonl")

    temps = [float(x.strip()) for x in args.temps.split(",") if x.strip()]
    ctx_keys = [x.strip() for x in args.contexts.split(",") if x.strip()]
    for ck in ctx_keys + [args.twohop_context]:
        if ck not in CONTEXTS:
            raise ValueError(f"Unknown context key: {ck}. Available: {list(CONTEXTS.keys())}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in environment")

    # -------------------------
    # Stage 1: collect trials
    # -------------------------
    conditions = [(ck, t) for ck in ctx_keys for t in temps]
    per_cond = args.total_calls // len(conditions)
    remainder = args.total_calls - per_cond * len(conditions)
    alloc = {cond: per_cond for cond in conditions}
    for i in range(remainder):
        alloc[conditions[i % len(conditions)]] += 1

    records: List[TrialRecord] = []
    trial_id = 0

    stage1_total = args.total_calls
    stage2_total = args.twohop_topk * args.twohop_trials_per_neighbor
    overall_total = stage1_total + stage2_total
    overall_start = time.time()

    print(f"Stage 1: running {stage1_total} calls, saving to {stage1_path}")
    stage1_start = time.time()
    stage1_last_report = stage1_start
    stage1_done = 0
    with open(stage1_path, "w", encoding="utf-8") as f:
        for (ctx_key, temp) in conditions:
            n_trials = alloc[(ctx_key, temp)]
            context_text = CONTEXTS[ctx_key]
            for _ in range(n_trials):
                trial_id += 1
                prompt = PROMPT_TEMPLATE.format(n_words=args.n_words, target=args.target, context=context_text)
                raw_text = call_association(
                    client=client,
                    model=args.model,
                    target=args.target,
                    context_text=context_text,
                    n_words=args.n_words,
                    temperature=temp,
                )
                toks = parse_assoc_list(raw_text, args.n_words)
                rec = TrialRecord(
                    trial_id=trial_id,
                    model=args.model,
                    stimulus=args.target.lower(),
                    context=ctx_key,
                    temperature=float(temp),
                    n_words=int(args.n_words),
                    prompt=prompt,
                    raw_text=raw_text,
                    tokens=toks,
                    stage="stage1",
                )
                records.append(rec)
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                stage1_done += 1
                stage1_last_report = maybe_report_progress(
                    stage_label="Stage 1 progress",
                    done=stage1_done,
                    total=stage1_total,
                    stage_start=stage1_start,
                    overall_done=stage1_done,
                    overall_total=overall_total,
                    overall_start=overall_start,
                    last_report_time=stage1_last_report,
                )

    df1 = pd.DataFrame([asdict(r) for r in records])

    # Choose top neighbors from a single condition for reproducibility
    pick = df1[(df1["context"] == args.twohop_context) & (df1["temperature"] == args.twohop_temperature)]
    if pick.empty:
        raise RuntimeError(
            "No trials found for two-hop selection condition. "
            "Check --twohop_context and --twohop_temperature match Stage 1 settings."
        )

    base_trials = pick["tokens"].tolist()
    scores = rank_weighted_scores(base_trials, target=args.target.lower())
    top_neighbors = [w for w, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:args.twohop_topk]]

    top_neighbors_path = os.path.join(args.outdir, "twohop_top_neighbors.csv")
    pd.DataFrame({"neighbor": top_neighbors}).to_csv(top_neighbors_path, index=False)

    # -------------------------
    # Stage 2: expand each neighbor
    # -------------------------
    print(
        f"Stage 2: expanding top-{args.twohop_topk} neighbors with "
        f"{args.twohop_trials_per_neighbor} calls each "
        f"({args.twohop_topk * args.twohop_trials_per_neighbor} total), saving to {stage2_path}"
    )

    stage2_records: List[TrialRecord] = []
    stage2_start = time.time()
    stage2_last_report = stage2_start
    stage2_done = 0
    with open(stage2_path, "w", encoding="utf-8") as f:
        for neigh in top_neighbors:
            context_text = CONTEXTS[args.twohop_context]
            for _ in range(args.twohop_trials_per_neighbor):
                trial_id += 1
                prompt = PROMPT_TEMPLATE.format(n_words=args.n_words, target=neigh, context=context_text)
                raw_text = call_association(
                    client=client,
                    model=args.model,
                    target=neigh,
                    context_text=context_text,
                    n_words=args.n_words,
                    temperature=args.twohop_temperature,
                )
                toks = parse_assoc_list(raw_text, args.n_words)
                rec = TrialRecord(
                    trial_id=trial_id,
                    model=args.model,
                    stimulus=neigh,
                    context=args.twohop_context,
                    temperature=float(args.twohop_temperature),
                    n_words=int(args.n_words),
                    prompt=prompt,
                    raw_text=raw_text,
                    tokens=toks,
                    stage="stage2",
                )
                stage2_records.append(rec)
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                stage2_done += 1
                overall_done = stage1_total + stage2_done
                stage2_last_report = maybe_report_progress(
                    stage_label="Stage 2 progress",
                    done=stage2_done,
                    total=stage2_total,
                    stage_start=stage2_start,
                    overall_done=overall_done,
                    overall_total=overall_total,
                    overall_start=overall_start,
                    last_report_time=stage2_last_report,
                )

    df2 = pd.DataFrame([asdict(r) for r in stage2_records])
    df2.to_csv(os.path.join(args.outdir, "trials_stage2_flat.csv"), index=False)

    # -------------------------
    # Build two-hop directed graph
    # -------------------------
    target = args.target.lower()
    G2 = nx.DiGraph()
    G2.add_node(target)

    # First hop edges from Stage 1 scores
    for neigh in top_neighbors:
        w = float(scores.get(neigh, 0.0))
        G2.add_node(neigh)
        G2.add_edge(target, neigh, weight=w)

    # Second hop edges from Stage 2 outputs
    # Weight edges neigh -> token by frequency across neigh trials
    edge_counts: Dict[Tuple[str, str], int] = {}
    for neigh in top_neighbors:
        sub = df2[df2["stimulus"] == neigh]
        trials = sub["tokens"].tolist()
        for words in trials:
            for w in words:
                w = w.lower()
                if w == neigh:
                    continue
                edge_counts[(neigh, w)] = edge_counts.get((neigh, w), 0) + 1

    # Add edges above threshold
    for (u, v), c in edge_counts.items():
        if c >= args.twohop_min_edge_weight:
            G2.add_node(v)
            G2.add_edge(u, v, weight=int(c))

    # Save edges for paper artifacts
    edges_out = []
    for u, v, data in G2.edges(data=True):
        edges_out.append({"src": u, "dst": v, "weight": data.get("weight", 1.0)})
    edges_df = pd.DataFrame(edges_out).sort_values(["src", "weight"], ascending=[True, False])
    edges_df.to_csv(os.path.join(args.outdir, "two_hop_edges.csv"), index=False)

    # Plot
    fig5 = os.path.join(args.outdir, "fig5_two_hop_map.png")
    plot_two_hop_graph(G2, fig5, title=f"Figure 5. Two-hop association map (seed neighbors from {args.twohop_context}, T={args.twohop_temperature})")

    print("\nDone.")
    print(f"Results in: {args.outdir}")
    print(f"- Two-hop map: {fig5}")
    print(f"- Top neighbors: {top_neighbors_path}")
    print(f"- Two-hop edges: {os.path.join(args.outdir, 'two_hop_edges.csv')}")
    print(f"- Stage 2 raw: {stage2_path}")


if __name__ == "__main__":
    main()
