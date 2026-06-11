# ⚽ WC Edge Finder

A World Cup match-analysis app that finds **betting value**: it models match
probabilities from 49,000+ international results, strips the bookmaker's
margin from the odds you enter, and flags selections where the model's
probability beats the market's — with Kelly-sized stake recommendations.

## Quick start

```bash
./run.sh
```

then open the URL Streamlit prints (usually http://localhost:8501).

First run trains the model (~1 min); after that everything is cached.
During the tournament, hit **🔄 Refresh results data** in the sidebar daily —
the dataset updates with new results, and ratings update automatically.

## Setup from scratch

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh
```

## What's inside

| File | What it does |
|---|---|
| `wc_edge/data.py` | Downloads/loads the [martj42 international results](https://github.com/martj42/international_results) dataset (1872–present, includes WC 2026 fixtures) |
| `wc_edge/elo.py` | World Football Elo ratings: importance-weighted K, goal-margin multiplier, home advantage |
| `wc_edge/model.py` | Elo → expected goals (Poisson GLM, time-decayed) → Dixon-Coles score matrix → every market probability |
| `wc_edge/value.py` | Devig (Shin / proportional), EV, Kelly staking |
| `wc_edge/backtest.py` | Walk-forward validation on World Cups 2006–2022 |
| `app.py` | Streamlit dashboard |

## Backtest results (no peeking — trained only on pre-tournament data)

| Metric | Model | Random guessing |
|---|---|---|
| Pick accuracy (320 WC matches) | **57.8%** | 33.3% |
| Brier score | **0.565** | 0.667 |
| Log loss | **0.961** | 1.099 |

Calibration is good: when the model says 39%, it happens ~38% of the time.

## The philosophy (read this)

- **Win rate is the wrong target.** The goal is positive expected value, not
  picking winners. A 40%-to-win bet at odds of 3.00 is a great bet.
- Most realized edge comes from **line shopping** (enter the best odds you can
  find across books) and softer secondary markets — not from out-modelling
  the closing line.
- **Quarter-Kelly staking**, never more. The model doesn't know about
  injuries, rotation, or motivation — when you have news it doesn't, skip.
- Judge yourself on **closing line value** over many bets, not on one week's
  profit. Variance at a World Cup is enormous.
- This is a decision-support tool, not a money printer. Bet responsibly.
