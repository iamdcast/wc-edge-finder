"""Bet log + closing line value (CLV) tracking.

The only reliable way to know you're beating the book is process, not
results: log the odds you took, log the closing odds, and check that you
consistently got a better price than the close. Positive average CLV means
long-run profit regardless of how any single week swings.

Stored as a plain CSV (decimal odds, model probability as a 0-1 fraction)
so it survives app restarts locally and can be downloaded/restored on
ephemeral hosts like Streamlit Cloud.
"""
import numpy as np
import pandas as pd
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "bet_log.csv"

COLUMNS = ["placed", "match", "selection", "odds", "stake",
           "model_p", "closing_odds", "result"]
RESULTS = ["pending", "won", "lost", "push"]


def load_log():
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def save_log(df):
    LOG_PATH.parent.mkdir(exist_ok=True)
    df[COLUMNS].to_csv(LOG_PATH, index=False)


def with_metrics(df):
    """Per-bet computed columns: P/L $, EV $ (at placement), CLV %."""
    out = df.copy()
    for c in ("odds", "stake", "model_p", "closing_odds"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    res = out["result"].fillna("pending").astype(str)
    odds = out["odds"]
    stake = out["stake"].fillna(0.0)
    out["P/L $"] = np.where(res == "won", stake * (odds - 1.0),
                            np.where(res == "lost", -stake, 0.0))
    out["EV $"] = np.where(out["model_p"].notna(),
                           (out["model_p"] * odds - 1.0) * stake, np.nan)
    out["CLV %"] = np.where(out["closing_odds"].notna() & (out["closing_odds"] > 1),
                            odds / out["closing_odds"] - 1.0, np.nan)
    return out


def summary(m):
    """Headline stats from a with_metrics() frame."""
    res = m["result"].fillna("pending").astype(str)
    settled = m[res.isin(["won", "lost"])]
    staked_settled = float(settled["stake"].sum())
    clv = m["CLV %"].dropna()
    return {
        "bets": len(m),
        "record": f"{int((res == 'won').sum())}–{int((res == 'lost').sum())}",
        "staked": float(m["stake"].fillna(0).sum()),
        "pl": float(settled["P/L $"].sum()),
        "roi": float(settled["P/L $"].sum()) / staked_settled if staked_settled else 0.0,
        "expected_pl": float(np.nansum(m["EV $"].values)) if len(m) else 0.0,
        "avg_clv": float(clv.mean()) if len(clv) else None,
        "beat_close": float((clv > 0).mean()) if len(clv) else None,
    }
