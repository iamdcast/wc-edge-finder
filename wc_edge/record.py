"""Live WC 2026 prediction record — what the model called, scored honestly.

For every completed 2026 World Cup match, reconstruct the prediction the
model made BEFORE kickoff: fit on everything up to the tournament start,
then walk forward updating Elo with each real result — the exact protocol
of the historical backtest, so the live numbers are comparable to the
2006–2022 baseline (57.8% pick accuracy, 0.565 Brier).

This is a scoreboard, not a tuning set. A World Cup is ~104 matches;
re-tuning parameters on a handful of them is how you overfit to noise.
The model already learns the right way — every refresh updates Elo with
the new results and re-conditions the tournament sim.
"""
import numpy as np
import pandas as pd

from .elo import EloRatings
from .model import MatchModel

WC_YEAR = 2026
OUTCOMES = ["home", "draw", "away"]


def live_record(played):
    """One row per completed WC 2026 match: pick, exact-score call, scoring."""
    wc = played[
        (played["tournament"] == "FIFA World Cup")
        & (played["date"].dt.year == WC_YEAR)
    ].sort_values("date")
    if wc.empty:
        return pd.DataFrame()

    cutoff = wc["date"].min()
    train = played[played["date"] < cutoff]
    elo = EloRatings().fit(train)
    model = MatchModel.fit(train, elo=elo, fit_date=cutoff)

    rows = []
    for row in wc.itertuples(index=False):
        venue = "neutral" if row.neutral else "a_home"
        pred = model.predict(row.home_team, row.away_team, venue=venue)
        p = np.array([pred["home"], pred["draw"], pred["away"]])
        outcome = (
            "home" if row.home_score > row.away_score
            else ("away" if row.away_score > row.home_score else "draw")
        )
        o = np.array([outcome == k for k in OUTCOMES], dtype=float)
        pick_idx = int(np.argmax(p))
        pick_label = [row.home_team, "Draw", row.away_team][pick_idx]
        (bi, bj), p_score = pred["top_scores"][0]
        rows.append({
            "date": row.date,
            "match": f"{row.home_team} v {row.away_team}",
            "final": f"{row.home_score}–{row.away_score}",
            "pick": pick_label,
            "p_pick": float(p[pick_idx]),
            "pick_hit": OUTCOMES[pick_idx] == outcome,
            "called_score": f"{bi}–{bj}",
            "p_called_score": float(p_score),
            "score_hit": bool(bi == row.home_score and bj == row.away_score),
            "p_outcome": float(p[OUTCOMES.index(outcome)]),
            "brier": float(np.sum((p - o) ** 2)),
            "logloss": float(-np.log(max(p[o.astype(bool)][0], 1e-12))),
        })
        # walk forward: the next prediction knows this result, like live use
        model.elo.update(
            row.date, row.home_team, row.away_team,
            row.home_score, row.away_score, row.tournament, row.neutral,
            record_history=False,
        )
    return pd.DataFrame(rows)


def record_summary(rec):
    if rec.empty:
        return None
    return {
        "matches": len(rec),
        "pick_hits": int(rec["pick_hit"].sum()),
        "accuracy": float(rec["pick_hit"].mean()),
        "score_hits": int(rec["score_hit"].sum()),
        "brier": float(rec["brier"].mean()),
        "logloss": float(rec["logloss"].mean()),
    }
