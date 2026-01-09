# is_like — Two-Hop Association Maps + CAQBI

A research toolkit to probe how language models associate concepts. It generates two-hop association graphs and computes a composite **CAQBI** score (Concept Association Quality and Bias Index) from free-association trials.

This repo is designed for quick experiments on a single stimulus (e.g., "black") with clear artifacts you can compare across models.

## What you can do
- Build **two-hop association maps** from model outputs.
- Measure **stability, richness, and sensitive skew** via CAQBI.
- Export **CSV/JSON/PNG** artifacts for papers or notebooks.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
```

## Run: Two-hop map
```bash
python aga_eval_twohop1.py \
  --model gpt-5.2 \
  --target BLACK \
  --total_calls 1000 \
  --twohop_topk 30 \
  --twohop_trials_per_neighbor 20 \
  --outdir runs/black_twohop
```

Outputs (in `runs/black_twohop`):
- `raw_trials_stage1.jsonl`, `raw_trials_stage2.jsonl`
- `twohop_top_neighbors.csv`
- `two_hop_edges.csv`
- `fig5_two_hop_map.png`

## Run: CAQBI (quick)
```bash
python aga_eval_twohop_caqbi.py \
  --model gpt-5.2 \
  --target BLACK \
  --total_calls 1000 \
  --outdir run_caqbi
```

Outputs (in `run_caqbi`):
- `caqbi.json`

## Run: CAQBI (refined + report)
```bash
python aga_eval_twhop_caqbi_refined.py \
  --model gpt-5.2 \
  --target BLACK \
  --total_calls 1000 \
  --outdir caqbi_run
```

Outputs (in `caqbi_run`):
- `caqbi.json`
- `caqbi_report.txt`
- `s0_distribution.png`



## Tips
- These scripts can be **API-expensive**. Reduce `--total_calls` for quick iteration.
- Keep `OPENAI_API_KEY` in your shell or a local secrets manager. **Do not commit keys.**

## Repo layout
- `aga_eval_twohop1.py` — full two-hop pipeline + plots
- `aga_eval_twohop_caqbi.py` — minimal CAQBI runner
- `aga_eval_twhop_caqbi_refined.py` — CAQBI + report + bootstrap plot
- `runs/` — example run artifacts

## License
MIT
