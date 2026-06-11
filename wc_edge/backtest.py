"""Walk-forward backtest on past World Cups.

For each tournament the model is trained ONLY on matches played before it
(no peeking), then predicts every WC match in order, updating Elo as results
come in — exactly how it would be used live.

Caveat: this dataset records knockout scores after extra time, so a small
share of knockout games (~the ones that went to ET) are scored against the
120' result rather than the 90' market. Group-stage metrics are exact.
"""
import numpy as np
import pandas as pd

from .elo import EloRatings
from .model import MatchModel, score_matrix, matrix_markets


def backtest_world_cups(played, years=(2006, 2010, 2014, 2018, 2022)):
    rows = []
    for year in years:
        wc = played[
            (played["tournament"] == "FIFA World Cup")
            & (played["date"].dt.year == year)
        ].sort_values("date")
        if wc.empty:
            continue
        cutoff = wc["date"].min()
        train = played[played["date"] < cutoff]

        elo = EloRatings().fit(train)
        model = MatchModel.fit(train, elo=elo, fit_date=cutoff)

        # group stage = first 3 calendar weeks of the tournament (heuristic:
        # everything before the first gap-day after matchday 3; simpler and
        # robust: first 48 matches for 32-team WCs, 36 for 1998-2018 format)
        n_group = {2006: 48, 2010: 48, 2014: 48, 2018: 48, 2022: 48}.get(year, 48)

        for i, row in enumerate(wc.itertuples(index=False)):
            venue = "neutral"
            if not row.neutral:
                venue = "a_home"
            pred = model.predict(row.home_team, row.away_team, venue=venue)
            outcome = (
                "home" if row.home_score > row.away_score
                else ("away" if row.away_score > row.home_score else "draw")
            )
            p = np.array([pred["home"], pred["draw"], pred["away"]])
            o = np.array([
                outcome == "home", outcome == "draw", outcome == "away"
            ], dtype=float)
            pick = ["home", "draw", "away"][int(np.argmax(p))]
            elo_pick = "home" if pred["elo_a"] >= pred["elo_b"] else "away"
            total_goals = row.home_score + row.away_score
            rows.append({
                "year": year,
                "date": row.date,
                "match": f"{row.home_team} v {row.away_team}",
                "stage": "group" if i < n_group else "knockout",
                "p_home": p[0], "p_draw": p[1], "p_away": p[2],
                "p_over25": pred["over"][2.5],
                "outcome": outcome,
                "brier": float(np.sum((p - o) ** 2)),
                "logloss": float(-np.log(max(p[o.astype(bool)][0], 1e-12))),
                "hit": pick == outcome,
                "elo_fav_hit": elo_pick == outcome,
                "over25_hit": total_goals > 2.5,
            })
            # walk forward: update Elo with the real result
            model.elo.update(
                row.date, row.home_team, row.away_team,
                row.home_score, row.away_score, row.tournament, row.neutral,
                record_history=False,
            )

    return pd.DataFrame(rows)


def summarize(bt):
    """Per-tournament and overall metrics."""
    def agg(g):
        return pd.Series({
            "matches": len(g),
            "brier": g["brier"].mean(),
            "log_loss": g["logloss"].mean(),
            "accuracy": g["hit"].mean(),
            "elo_favorite_acc": g["elo_fav_hit"].mean(),
        })

    per_year = bt.groupby("year")[["brier", "logloss", "hit", "elo_fav_hit"]].apply(agg)
    overall = agg(bt)
    overall.name = "ALL"
    group_only = agg(bt[bt["stage"] == "group"])
    group_only.name = "ALL (group stage)"
    return per_year, overall, group_only


def calibration_table(bt, bins=8):
    """Predicted probability vs actual frequency, pooled over all outcomes."""
    recs = []
    for _, r in bt.iterrows():
        for k in ("home", "draw", "away"):
            recs.append((r[f"p_{k}"], 1.0 if r["outcome"] == k else 0.0))
    df = pd.DataFrame(recs, columns=["p", "hit"])
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 0.9, bins + 1))
    out = df.groupby("bin", observed=True).agg(
        predicted=("p", "mean"), actual=("hit", "mean"), n=("hit", "size")
    ).dropna().reset_index(drop=True)
    return out
