"""Klement-style tournament simulator for the 2026 World Cup.

Joachim Klement (the analyst who called Germany 2014, France 2018 and
Argentina 2022) doesn't predict matches better than anyone else — he rates
every team, converts ratings to match probabilities, then simulates the
WHOLE tournament thousands of times.  His "prediction" is the team that
lifts the trophy most often.  This module does exactly that, except the
match probabilities come from this app's Elo -> Poisson -> Dixon-Coles
engine rather than his socioeconomic regression (GDP/capita, population,
temperature, FIFA points — which all just proxy for team strength that Elo
measures directly).

Implements the real 2026 format: 12 groups of 4, top two plus the 8 best
third-placed teams into a round of 32, then the fixed FIFA bracket
(matches 73-104).  Group games that have already been played are taken as
fact; only unplayed games are simulated, so re-running mid-tournament
conditions the forecast on results so far.
"""
from collections import Counter

import numpy as np
import pandas as pd

GROUP_STAGE_END = pd.Timestamp("2026-06-27")
LETTERS = "ABCDEFGHIJKL"

# One unambiguous team per group (from the December 2025 draw) — lets us
# derive full group membership from the round-robin fixture structure
# instead of hardcoding 48 names that may be spelled differently upstream.
ANCHOR_TO_GROUP = {
    "Mexico": "A", "Canada": "B", "Brazil": "C", "United States": "D",
    "Germany": "E", "Netherlands": "F", "Belgium": "G", "Spain": "H",
    "France": "I", "Argentina": "J", "Portugal": "K", "England": "L",
}

# Round of 32, matches 73-88: "1A" = Group A winner, "2A" = runner-up,
# "3" = the third-placed team allocated to that match (see THIRD_SLOTS).
R32 = [
    (73, "2A", "2B"), (74, "1E", "3"), (75, "1F", "2C"), (76, "1C", "2F"),
    (77, "1I", "3"), (78, "2E", "2I"), (79, "1A", "3"), (80, "1L", "3"),
    (81, "1D", "3"), (82, "1G", "3"), (83, "2K", "2L"), (84, "1H", "2J"),
    (85, "1B", "3"), (86, "1J", "2H"), (87, "1K", "3"), (88, "2D", "2G"),
]
# Which groups' third-placed teams can be sent to each match (FIFA schedule).
THIRD_SLOTS = {
    74: "ABCDF", 77: "CDFGH", 79: "CEFHI", 80: "EHIJK",
    81: "BEFIJ", 82: "AEHIJ", 85: "EFGIJ", 87: "DEIJL",
}
# Later rounds reference winners by match number.
R16 = [(89, 74, 77), (90, 73, 75), (91, 76, 78), (92, 79, 80),
       (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)]
QF = [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)]
SF = [(101, 97, 98), (102, 99, 100)]

# R32/R16 are spread across all three host countries; QF onward is USA-only.
HOSTS_EARLY = {"United States", "Mexico", "Canada"}
HOSTS_LATE = {"United States"}


def derive_groups(group_matches):
    """Group letter -> [4 teams], from the round-robin fixture structure."""
    adj = {}
    for m in group_matches.itertuples(index=False):
        adj.setdefault(m.home_team, set()).add(m.away_team)
        adj.setdefault(m.away_team, set()).add(m.home_team)
    groups, seen = {}, set()
    for t, opps in adj.items():
        if t in seen:
            continue
        comp = {t} | opps
        seen |= comp
        anchors = [x for x in comp if x in ANCHOR_TO_GROUP]
        if len(comp) != 4 or len(anchors) != 1:
            raise ValueError(f"Can't resolve group containing {sorted(comp)}")
        groups[ANCHOR_TO_GROUP[anchors[0]]] = sorted(comp)
    if len(groups) != 12:
        raise ValueError(f"Expected 12 groups, derived {len(groups)}")
    return groups


def _assign_thirds(qualified, rng):
    """Allocate the 8 qualified third-place groups to their bracket slots.

    FIFA's Annex C fixes one assignment per combination of qualifiers; we
    pick a random assignment that respects the allowed-groups constraint
    of each slot, which is equivalent for probability purposes.
    """
    cands = {m: [g for g in qualified if g in allowed]
             for m, allowed in THIRD_SLOTS.items()}
    order = sorted(THIRD_SLOTS, key=lambda m: len(cands[m]))
    assign, used = {}, set()

    def bt(i):
        if i == len(order):
            return True
        m = order[i]
        free = [g for g in cands[m] if g not in used]
        rng.shuffle(free)
        for g in free:
            assign[m] = g
            used.add(g)
            if bt(i + 1):
                return True
            used.discard(g)
        assign.pop(m, None)
        return False

    if not bt(0):  # shouldn't happen, but never crash the sim over it
        pool = list(qualified)
        rng.shuffle(pool)
        assign = dict(zip(THIRD_SLOTS, pool))
    return assign


def simulate(model, played, upcoming, n_sims=10000, seed=7):
    """Monte Carlo the rest of WC 2026. Returns (per-team DataFrame, extras).

    DataFrame columns are P(reach stage): win group, make the round of 32,
    R16, QF, SF, the final, and win the cup.
    """
    rng = np.random.default_rng(seed)
    is_wc = lambda df: ((df["tournament"] == "FIFA World Cup")
                        & (df["date"] >= "2026-01-01"))
    wc_played = played[is_wc(played)]
    wc_upcoming = upcoming[is_wc(upcoming)]
    gs_played = wc_played[wc_played["date"] <= GROUP_STAGE_END]
    gs_upcoming = wc_upcoming[wc_upcoming["date"] <= GROUP_STAGE_END]
    groups = derive_groups(pd.concat([gs_played, gs_upcoming]))
    all_teams = [t for ts in groups.values() for t in ts]

    # ---------------- group stage: actual results + sampled score matrices
    pts = {t: np.zeros(n_sims) for t in all_teams}
    gd = {t: np.zeros(n_sims) for t in all_teams}
    gf = {t: np.zeros(n_sims) for t in all_teams}

    def credit(home, away, hg, ag):
        pts[home] += np.where(hg > ag, 3.0, np.where(hg == ag, 1.0, 0.0))
        pts[away] += np.where(ag > hg, 3.0, np.where(hg == ag, 1.0, 0.0))
        gd[home] += hg - ag
        gd[away] += ag - hg
        gf[home] += hg
        gf[away] += ag

    for m in gs_played.itertuples(index=False):
        credit(m.home_team, m.away_team,
               float(m.home_score), float(m.away_score))
    for m in gs_upcoming.itertuples(index=False):
        venue = "neutral" if m.neutral else "a_home"
        M = model.predict(m.home_team, m.away_team, venue=venue)["matrix"]
        idx = rng.choice(M.size, size=n_sims, p=M.ravel())
        credit(m.home_team, m.away_team,
               (idx // M.shape[1]).astype(float),
               (idx % M.shape[1]).astype(float))

    # Rank within groups: points, GD, GF, then random jitter standing in
    # for the remaining FIFA tiebreakers (head-to-head, fair play, lots).
    win_team, run_team, third_team = {}, {}, {}
    third_key = np.zeros((12, n_sims))
    for gi, g in enumerate(LETTERS):
        ts = groups[g]
        key = np.stack([pts[t] * 1e8 + (gd[t] + 500.0) * 1e4 + gf[t] * 10.0
                        + rng.random(n_sims) for t in ts])
        rk = np.argsort(-key, axis=0)
        names = np.array(ts, dtype=object)
        win_team[g] = names[rk[0]]
        run_team[g] = names[rk[1]]
        third_team[g] = names[rk[2]]
        third_key[gi] = np.take_along_axis(key, rk[2:3], axis=0)[0]

    qual_mask = np.zeros((12, n_sims), dtype=bool)  # best 8 thirds advance
    qual_mask[np.argsort(-third_key, axis=0)[:8], np.arange(n_sims)] = True

    # ---------------- knockout machinery
    # Knockout games already in the books are taken as fact.
    ko_results = {}
    for m in wc_played[wc_played["date"] > GROUP_STAGE_END].itertuples(index=False):
        if m.home_score != m.away_score:
            winner = m.home_team if m.home_score > m.away_score else m.away_team
        else:
            winner = None  # went to pens; shootout result isn't in the data
        ko_results[frozenset((m.home_team, m.away_team))] = winner

    adv_cache = {}

    def p_advance(a, b, early):
        k = (a, b, early)
        if k not in adv_cache:
            hosts = HOSTS_EARLY if early else HOSTS_LATE
            if a in hosts and b not in hosts:
                venue = "a_home"
            elif b in hosts and a not in hosts:
                venue = "b_home"
            else:
                venue = "neutral"
            adv_cache[k] = model.predict(a, b, venue=venue,
                                         knockout=True)["advance_a"]
        return adv_cache[k]

    def play(a, b, early, u):
        known = ko_results.get(frozenset((a, b)), "")
        if known != "":
            if known is not None:
                return known
            return a if u < 0.5 else b  # decided on pens: coin flip
        return a if u < p_advance(a, b, early) else b

    counts = {t: Counter() for t in all_teams}
    final_pairs = Counter()
    U = rng.random((n_sims, 31))

    for s in range(n_sims):
        slot = {}
        for g in LETTERS:
            slot["1" + g] = win_team[g][s]
            slot["2" + g] = run_team[g][s]
        qualified = [LETTERS[i] for i in range(12) if qual_mask[i, s]]
        third_at = _assign_thirds(qualified, rng)

        w, ui = {}, 0
        for mno, sa, sb in R32:
            a = third_team[third_at[mno]][s] if sa == "3" else slot[sa]
            b = third_team[third_at[mno]][s] if sb == "3" else slot[sb]
            counts[a]["r32"] += 1
            counts[b]["r32"] += 1
            w[mno] = play(a, b, True, U[s, ui]); ui += 1
            counts[w[mno]]["r16"] += 1
        for stage, matches, early in (("qf", R16, True), ("sf", QF, False),
                                      ("final", SF, False)):
            for mno, ma, mb in matches:
                w[mno] = play(w[ma], w[mb], early, U[s, ui]); ui += 1
                counts[w[mno]][stage] += 1
        a, b = w[101], w[102]
        final_pairs[tuple(sorted((a, b)))] += 1
        champ = play(a, b, False, U[s, ui])
        counts[champ]["champ"] += 1

    # ---------------- aggregate
    rows = []
    for g in LETTERS:
        for t in groups[g]:
            c = counts[t]
            rows.append({
                "Team": t, "Group": g, "Elo": model.elo.get(t),
                "Win group": float((win_team[g] == t).mean()),
                "Make R32": c["r32"] / n_sims,
                "Make R16": c["r16"] / n_sims,
                "Make QF": c["qf"] / n_sims,
                "Make SF": c["sf"] / n_sims,
                "Make final": c["final"] / n_sims,
                "Win cup": c["champ"] / n_sims,
            })
    table = (pd.DataFrame(rows).set_index("Team")
             .sort_values("Win cup", ascending=False))
    extras = {"final_pairs": final_pairs, "groups": groups, "n_sims": n_sims}
    return table, extras
