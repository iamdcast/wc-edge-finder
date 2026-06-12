"""WC Edge Finder — World Cup match analysis & betting value dashboard.

Run with:  .venv/bin/streamlit run app.py
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from datetime import date

from wc_edge import record as R
from wc_edge import tournament as T
from wc_edge import tracker as TR
from wc_edge import value as V
from wc_edge.backtest import backtest_world_cups, calibration_table, summarize
from wc_edge.data import load_results, refresh_results
from wc_edge.model import MatchModel

st.set_page_config(page_title="WC Edge Finder", page_icon="⚽", layout="wide")


# ----------------------------------------------------------- cached state

@st.cache_resource(show_spinner="Training model on 49,000+ international matches…")
def get_model():
    played, upcoming = load_results()
    model = MatchModel.fit(played)
    return model, played, upcoming


@st.cache_data(show_spinner="Simulating the 2026 World Cup thousands of times…")
def get_simulation(n_sims, data_stamp):
    model, played, upcoming = get_model()
    return T.simulate(model, played, upcoming, n_sims=n_sims)


@st.cache_data(show_spinner="Scoring the model's 2026 predictions…")
def get_live_record(data_stamp):
    model, played, upcoming = get_model()
    return R.live_record(played)


@st.cache_data(show_spinner="Backtesting on World Cups 2006–2022 (walk-forward)…")
def get_backtest():
    played, _ = load_results()
    bt = backtest_world_cups(played)
    per_year, overall, group_only = summarize(bt)
    cal = calibration_table(bt)
    return bt, per_year, overall, group_only, cal


model, played, upcoming = get_model()
TEAMS = sorted(model.elo.ratings.keys(), key=lambda t: -model.elo.get(t))
WC_FIXTURES = upcoming[upcoming["tournament"] == "FIFA World Cup"].copy()
DATA_STAMP = f"{played['date'].max().date()}-{len(played)}"


def rotation_flags(sim_table, threshold=0.999):
    """Teams whose round-of-32 fate is (virtually) sealed before they've
    finished the group — prime rotation / low-motivation spots."""
    flags = {}
    for t, p in sim_table["Make R32"].items():
        if p >= threshold:
            flags[t] = "through"
        elif p <= 1.0 - threshold:
            flags[t] = "out"
    return flags


# ----------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("⚽ WC Edge Finder")
    st.caption(
        "Pick the games **and** beat the price. Every match gets a model "
        "pick and most likely score — tracked on 📡 2026 Record. But a bet "
        "is only good when the book's implied probability is lower than the "
        "true one. Picks fill the scoreboard; value pays the bills."
    )
    odds_format = st.selectbox(
        "Odds format", ["American", "Decimal"],
        help="American: -150 = risk $150 to win $100, +200 = win $200 on $100. "
             "All odds you enter AND all fair odds shown use this format.",
    )
    bankroll = st.number_input("Bankroll ($)", 10.0, 1_000_000.0, 1000.0, step=50.0)
    kelly_mult = st.slider(
        "Kelly fraction", 0.05, 1.0, 0.25, 0.05,
        help="Fraction of the full Kelly stake. 0.25 (quarter Kelly) is the "
             "standard sane choice — full Kelly is a rollercoaster.",
    )
    min_edge = st.slider(
        "Min EV to bet (%)", 0.0, 10.0, 3.0, 0.5,
        help="Only flag bets whose expected value per $1 exceeds this. "
             "Below ~2-3% the edge is usually noise.",
    ) / 100.0
    devig_method = st.selectbox(
        "Devig method", ["shin", "proportional"],
        help="How to strip the bookmaker margin from the odds. Shin corrects "
             "for favorite-longshot bias and is the better default.",
    )
    st.divider()
    if st.button("🔄 Refresh results data", width="stretch"):
        refresh_results()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.caption(
        f"Data through **{played['date'].max().date()}** · "
        f"{len(played):,} matches · {len(WC_FIXTURES)} WC fixtures loaded"
    )

(tab_analyze, tab_fixtures, tab_tournament, tab_record, tab_tracker,
 tab_rankings, tab_validate, tab_help) = st.tabs(
    ["🎯 Match Analyzer", "📅 Fixtures", "🏆 Tournament Sim", "📡 2026 Record",
     "📒 Bet Tracker", "📊 Power Rankings", "🧪 Model Validation",
     "📖 How It Works"]
)


# ------------------------------------------------------------- helpers

def fmt_odds(d):
    """Format decimal odds for display in the user's chosen format."""
    if odds_format == "American":
        a = V.decimal_to_american(d)
        return f"+{a:.0f}" if a > 0 else f"{a:.0f}"
    return f"{d:.2f}"


def odds_input(container, label, key, placeholder_dec):
    """Odds entry in the chosen format; always returns DECIMAL odds (or None)."""
    if odds_format == "American":
        v = container.number_input(
            label, value=None, step=5.0, format="%.0f",
            placeholder="e.g. -150 or +200", key=key + "_am",
        )
        if v is None:
            return None
        if abs(v) < 100:
            container.caption("⚠️ American odds must be ≤ −100 or ≥ +100")
            return None
        return V.american_to_decimal(v)
    return container.number_input(label, min_value=1.01, value=None,
                                  placeholder=placeholder_dec, key=key)


def venue_for_fixture(row):
    return "neutral" if row["neutral"] else "a_home"


def fmt_fixture(row):
    home_tag = "" if row["neutral"] else " 🏠"
    return (f"{row['date'].date()} — {row['home_team']}{home_tag} vs "
            f"{row['away_team']} ({row['city']})")


def value_rows(selections, bankroll, kelly_mult, min_edge, devig_method):
    """selections: list of (label, p_model, odds) for ONE market.
    Devigs within the market, returns table rows."""
    odds = [o for _, _, o in selections]
    fair = V.devig(odds, devig_method)
    return [
        V.evaluate_market(label, p, o, fp, bankroll, kelly_mult, min_edge)
        for (label, p, o), fp in zip(selections, fair)
    ]


def min_betable_odds(p, min_edge):
    """Smallest decimal price at which a selection clears the EV threshold."""
    return (1.0 + min_edge) / p


def render_recommendation(rows, bankroll, min_edge, dead_rubber=False):
    """One clear verdict: the best bet at these prices, or an explicit pass."""
    bets = sorted((r for r in rows if r["Verdict"] == "BET"),
                  key=lambda r: -r["EV %"])
    if bets:
        best = bets[0]
        msg = (
            f"### 🟢 Recommendation: {best['Market']} @ {fmt_odds(best['Book odds'])}\n"
            f"Stake **${best['Stake $']:,.2f}** "
            f"({best['Stake $'] / bankroll:.1%} of bankroll) · "
            f"EV **{best['EV %']:+.1%}** · model {best['Model %']:.1%} vs "
            f"book implied {best['Book implied %']:.1%}"
        )
        for r in bets[1:]:
            msg += (f"\n- Also +EV: **{r['Market']}** @ {fmt_odds(r['Book odds'])}"
                    f" — stake ${r['Stake $']:,.2f}, EV {r['EV %']:+.1%}")
        if len(bets) > 1:
            total = sum(r["Stake $"] for r in bets)
            msg += (f"\n\nTotal stake ${total:,.2f} "
                    f"({total / bankroll:.1%} of bankroll).")
        msg += ("\n\nLog it in the 📒 Bet Tracker with the book and price so "
                "CLV gets scored. +EV bets still lose all the time — the edge "
                "only shows up over dozens of bets.")
        st.success(msg)
    else:
        checks = [r for r in rows if r["Verdict"] == "CHECK INPUTS"]
        if checks:
            best = max(checks, key=lambda r: r["EV %"])
            st.warning(
                f"### 🟡 Recommendation: VERIFY, THEN BET\n"
                f"**{best['Market']}** at {fmt_odds(best['Book odds'])} shows "
                f"EV **{best['EV %']:+.1%}** (model {best['Model %']:.1%} vs "
                f"devigged book {best['Devig %']:.1%}). An edge that big is "
                f"usually a typo or news the model can't see — re-check the "
                f"odds you entered, the team news, and ideally a sharp book's "
                f"line in the anchor above. If the price is real and there's "
                f"no news, it's a bet: stake **${best['Stake $']:,.2f}** "
                f"({best['Stake $'] / bankroll:.1%} of bankroll)."
            )
        else:
            best = max(rows, key=lambda r: r["EV %"])
            need = min_betable_odds(best["Model %"], min_edge)
            st.warning(
                f"### ⚪ Recommendation: PASS\n"
                f"No selection clears the {min_edge:.0%} EV bar at these "
                f"prices — not betting is the correct play here. Closest: "
                f"**{best['Market']}** at {fmt_odds(best['Book odds'])} "
                f"(EV {best['EV %']:+.1%}); it becomes a bet at "
                f"**{fmt_odds(need)}** or better. Shop other books for that "
                f"number."
            )
    if dead_rubber:
        st.caption(
            "⚠️ Dead-rubber risk above still applies — if you bet at all, "
            "prefer the motivated side and reduce the stake."
        )


def render_value_table(rows):
    df = pd.DataFrame(rows)
    styled = (
        df.style
        .format({
            "Model %": "{:.1%}", "Book implied %": "{:.1%}", "Devig %": "{:.1%}",
            "Fair odds": fmt_odds, "Book odds": fmt_odds,
            "EV %": "{:+.1%}", "Stake $": "${:,.2f}",
        })
        .apply(
            lambda r: [
                "background-color: rgba(46, 160, 67, 0.25)" if r["Verdict"] == "BET"
                else ("background-color: rgba(210, 153, 34, 0.2)"
                      if r["Verdict"] == "CHECK INPUTS" else "")
            ] * len(r),
            axis=1,
        )
    )
    st.dataframe(styled, width="stretch", hide_index=True)


# ------------------------------------------------------- Match Analyzer

with tab_analyze:
    st.subheader("Analyze a match")

    fixture_options = ["Manual team selection"] + [
        fmt_fixture(r) for _, r in WC_FIXTURES.iterrows()
    ]
    pick = st.selectbox("Quick-pick an upcoming World Cup fixture", fixture_options)

    if pick != "Manual team selection":
        row = WC_FIXTURES.iloc[fixture_options.index(pick) - 1]
        team_a, team_b = row["home_team"], row["away_team"]
        default_venue = venue_for_fixture(row)
    else:
        team_a, team_b, default_venue = "Argentina", "France", "neutral"

    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
    with c1:
        team_a = st.selectbox("Team A", TEAMS, index=TEAMS.index(team_a))
    with c2:
        team_b = st.selectbox("Team B", TEAMS, index=TEAMS.index(team_b))
    with c3:
        venue = st.radio(
            "Venue", ["neutral", "a_home", "b_home"],
            index=["neutral", "a_home", "b_home"].index(default_venue),
            format_func=lambda v: {
                "neutral": "Neutral", "a_home": f"{team_a} at home",
                "b_home": f"{team_b} at home"}[v],
        )
    with c4:
        knockout = st.radio("Stage", ["Group", "Knockout"]) == "Knockout"

    if team_a == team_b:
        st.warning("Pick two different teams.")
        st.stop()

    pred = model.predict(team_a, team_b, venue=venue, knockout=knockout)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{team_a} Elo", f"{pred['elo_a']:.0f}")
    m2.metric(f"{team_b} Elo", f"{pred['elo_b']:.0f}")
    m3.metric("Expected goals", f"{pred['xg_a']:.2f} – {pred['xg_b']:.2f}")
    m4.metric("Over 2.5 goals", f"{pred['over'][2.5]:.1%}")
    m5.metric("Both teams score", f"{pred['btts']:.1%}")

    p1, p2, p3 = st.columns(3)
    p1.metric(f"{team_a} win (90')", f"{pred['home']:.1%}",
              f"fair odds {fmt_odds(1/pred['home'])}", delta_color="off")
    p2.metric("Draw (90')", f"{pred['draw']:.1%}",
              f"fair odds {fmt_odds(1/pred['draw'])}", delta_color="off")
    p3.metric(f"{team_b} win (90')", f"{pred['away']:.1%}",
              f"fair odds {fmt_odds(1/pred['away'])}", delta_color="off")

    if knockout:
        k1, k2 = st.columns(2)
        k1.metric(f"{team_a} advances", f"{pred['advance_a']:.1%}",
                  f"fair odds {fmt_odds(1/pred['advance_a'])}", delta_color="off")
        k2.metric(f"{team_b} advances", f"{pred['advance_b']:.1%}",
                  f"fair odds {fmt_odds(1/pred['advance_b'])}", delta_color="off")

    outcomes = [(f"{team_a} win", pred["home"]), ("Draw", pred["draw"]),
                (f"{team_b} win", pred["away"])]
    pick_outcome, p_pick = max(outcomes, key=lambda kv: kv[1])
    (bi, bj), p_score = pred["top_scores"][0]
    conf = ("strong" if p_pick >= 0.60 else
            "moderate" if p_pick >= 0.45 else "slight — close to a coin flip")
    pick_line = (f"**Model pick: {pick_outcome}** ({p_pick:.0%}, {conf} "
                 f"confidence) · most likely score **{bi}–{bj}** ({p_score:.0%})")
    if knockout:
        adv = team_a if pred["advance_a"] >= 0.5 else team_b
        pick_line += (f" · **{adv} to advance** "
                      f"({max(pred['advance_a'], pred['advance_b']):.0%})")
    st.info(pick_line)

    dead_rubber = False
    if pick != "Manual team selection" and row["date"] <= T.GROUP_STAGE_END:
        flags = rotation_flags(get_simulation(10000, DATA_STAMP)[0])
        flagged = [(t, flags[t]) for t in (team_a, team_b) if t in flags]
        if flagged:
            dead_rubber = True
            msg = " and ".join(
                f"**{t}** are already "
                + ("through to the round of 32" if f == "through"
                   else "eliminated")
                for t, f in flagged)
            st.warning(
                f"⚠️ **Dead-rubber risk:** {msg} regardless of this result. "
                "Expect rotation and low intensity — the model can't see "
                "lineups, so its probabilities are least trustworthy here. "
                "Skipping (or fading the unmotivated side at a good price) "
                "is usually the play."
            )

    st.divider()
    st.subheader(f"Enter your sportsbook's odds ({odds_format.lower()})")
    st.caption(
        "Shop multiple books and enter the BEST price for each selection — "
        "line shopping is free edge."
    )

    oc1, oc2, oc3 = st.columns(3)
    odds_h = odds_input(oc1, f"{team_a} win", "oh", "e.g. 2.50")
    odds_d = odds_input(oc2, "Draw", "od", "e.g. 3.30")
    odds_a = odds_input(oc3, f"{team_b} win", "oa", "e.g. 2.90")

    with st.expander("🎯 Sharp anchor — blend the model with a sharp book "
                     "(recommended)"):
        st.caption(
            "Enter the 1X2 from the sharpest book you can see (Pinnacle, "
            "Circa, betting exchanges). Their devigged line is the best "
            "single estimate of the truth, so the value report below will "
            "use a model/market blend instead of the raw model — this "
            "protects you from model blind spots like injuries and rotation. "
            "You still profit by betting SOFT books whose prices stray from "
            "the blend."
        )
        sc1, sc2, sc3, sc4 = st.columns([2, 2, 2, 3])
        sharp_h = odds_input(sc1, f"{team_a} win (sharp)", "sh", "e.g. 2.45")
        sharp_d = odds_input(sc2, "Draw (sharp)", "sd", "e.g. 3.25")
        sharp_a = odds_input(sc3, f"{team_b} win (sharp)", "sa", "e.g. 2.95")
        w_model = sc4.slider(
            "Model weight in blend", 0.0, 1.0, 0.3, 0.05,
            help="0.3 = trust the sharp market 70%, the model 30%. "
                 "Raise it only if you really believe the model knows "
                 "something the market doesn't.",
        )

    p_h, p_d, p_a = pred["home"], pred["draw"], pred["away"]
    if sharp_h and sharp_d and sharp_a:
        p_sharp = V.devig([sharp_h, sharp_d, sharp_a], devig_method)
        p_h, p_d, p_a = V.blend_probs([p_h, p_d, p_a], p_sharp, w_model)
        st.caption(
            f"Using blended 1X2 probabilities: {team_a} {p_h:.1%} · "
            f"draw {p_d:.1%} · {team_b} {p_a:.1%} "
            f"(model {w_model:.0%} / sharp market {1 - w_model:.0%})"
        )

    with st.expander("More markets: totals, BTTS" + (", to advance" if knockout else "")):
        t1, t2, t3, t4 = st.columns(4)
        odds_over = odds_input(t1, "Over 2.5", "oo", "e.g. 2.10")
        odds_under = odds_input(t2, "Under 2.5", "ou", "e.g. 1.75")
        odds_btts_y = odds_input(t3, "BTTS — Yes", "oby", "e.g. 2.00")
        odds_btts_n = odds_input(t4, "BTTS — No", "obn", "e.g. 1.80")
        odds_adv_a = odds_adv_b = None
        if knockout:
            a1, a2 = st.columns(4)[:2]
            odds_adv_a = odds_input(a1, f"{team_a} to advance", "oaa", "e.g. 1.90")
            odds_adv_b = odds_input(a2, f"{team_b} to advance", "oab", "e.g. 1.90")

    rows = []
    if odds_h and odds_d and odds_a:
        rows += value_rows(
            [(f"{team_a} win", p_h, odds_h),
             ("Draw", p_d, odds_d),
             (f"{team_b} win", p_a, odds_a)],
            bankroll, kelly_mult, min_edge, devig_method)
        ovr = V.overround([odds_h, odds_d, odds_a])
        st.caption(f"1X2 overround: {ovr:.3f} → bookmaker margin "
                   f"{(ovr - 1):.1%}. Lower margin = better book.")
    if odds_over and odds_under:
        rows += value_rows(
            [("Over 2.5", pred["over"][2.5], odds_over),
             ("Under 2.5", 1 - pred["over"][2.5], odds_under)],
            bankroll, kelly_mult, min_edge, devig_method)
    if odds_btts_y and odds_btts_n:
        rows += value_rows(
            [("BTTS Yes", pred["btts"], odds_btts_y),
             ("BTTS No", 1 - pred["btts"], odds_btts_n)],
            bankroll, kelly_mult, min_edge, devig_method)
    if knockout and odds_adv_a and odds_adv_b:
        rows += value_rows(
            [(f"{team_a} advances", pred["advance_a"], odds_adv_a),
             (f"{team_b} advances", pred["advance_b"], odds_adv_b)],
            bankroll, kelly_mult, min_edge, devig_method)

    if rows:
        st.subheader("Value report")
        render_value_table(rows)
        render_recommendation(rows, bankroll, min_edge, dead_rubber)
        check = [r for r in rows if r["Verdict"] == "CHECK INPUTS"]
        if check:
            st.warning(
                "Some edges look implausibly large (>15%). Double-check you "
                "entered the odds correctly and that there's no team news "
                "(injuries, rotation) the model can't know about."
            )
    else:
        st.subheader("📋 Recommendation — prices to shop for")
        st.caption(
            "The model pick is the scoreboard call, not the bet. The bet is "
            f"whatever clears the {min_edge:.0%} EV bar at your book. Shop "
            "your books for any price at or above **Bet at ≥**, then enter "
            "the odds above to get the full verdict with stakes."
        )
        shop_targets = [
            (f"{team_a} win", p_h), ("Draw", p_d), (f"{team_b} win", p_a),
            ("Over 2.5", pred["over"][2.5]),
            ("Under 2.5", 1.0 - pred["over"][2.5]),
            ("BTTS Yes", pred["btts"]), ("BTTS No", 1.0 - pred["btts"]),
        ]
        if knockout:
            shop_targets += [(f"{team_a} to advance", pred["advance_a"]),
                             (f"{team_b} to advance", pred["advance_b"])]
        shop_df = pd.DataFrame(
            [(label, p, 1.0 / p, min_betable_odds(p, min_edge))
             for label, p in shop_targets],
            columns=["Market", "Model %", "Fair odds", "Bet at ≥"],
        )
        st.dataframe(
            shop_df.style.format({
                "Model %": "{:.1%}", "Fair odds": fmt_odds, "Bet at ≥": fmt_odds,
            }),
            hide_index=True, width="stretch",
        )

    st.divider()
    st.subheader("Most likely scorelines")
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        score_df = pd.DataFrame(
            [(f"{i}–{j}", p) for (i, j), p in pred["top_scores"]],
            columns=["Score", "Probability"],
        )
        st.dataframe(
            score_df.style.format({"Probability": "{:.1%}"}),
            hide_index=True, width="stretch",
        )
    with sc2:
        M = pred["matrix"][:6, :6]
        heat = pd.DataFrame(
            [(i, j, M[i, j]) for i in range(6) for j in range(6)],
            columns=[team_a, team_b, "p"],
        )
        chart = (
            alt.Chart(heat)
            .mark_rect()
            .encode(
                x=alt.X(f"{team_b}:O", title=f"{team_b} goals"),
                y=alt.Y(f"{team_a}:O", title=f"{team_a} goals",
                        sort=alt.EncodingSortField(field=team_a, order="descending")),
                color=alt.Color("p:Q", legend=None, scale=alt.Scale(scheme="greens")),
                tooltip=[team_a, team_b, alt.Tooltip("p:Q", format=".1%")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)


# ------------------------------------------------------------- Fixtures

with tab_fixtures:
    st.subheader("Upcoming World Cup fixtures — model board")
    st.caption(
        "Model probabilities and fair odds for every scheduled match. Compare "
        "fair odds against your book: value = their odds HIGHER than fair."
    )
    if WC_FIXTURES.empty:
        st.info("No upcoming WC fixtures in the dataset. Hit refresh in the sidebar.")
    else:
        fx_flags = rotation_flags(get_simulation(10000, DATA_STAMP)[0])
        recs = []
        for _, r in WC_FIXTURES.iterrows():
            p = model.predict(r["home_team"], r["away_team"],
                              venue=venue_for_fixture(r))
            probs = {r["home_team"]: p["home"], "Draw": p["draw"],
                     r["away_team"]: p["away"]}
            pick = max(probs, key=probs.get)
            warn = ""
            if r["date"] <= T.GROUP_STAGE_END:
                warn = " · ".join(
                    f"⚠️ {t} {'through' if fx_flags[t] == 'through' else 'out'}"
                    for t in (r["home_team"], r["away_team"]) if t in fx_flags)
            recs.append({
                "Date": r["date"].date(),
                "Match": f"{r['home_team']}{'' if r['neutral'] else ' 🏠'} vs {r['away_team']}",
                "City": r["city"],
                "Model pick": f"{pick} ({probs[pick]:.0%})",
                "Motivation": warn,
                f"P(home)": p["home"], "P(draw)": p["draw"], "P(away)": p["away"],
                "Fair 1": 1 / p["home"], "Fair X": 1 / p["draw"], "Fair 2": 1 / p["away"],
                "O2.5 %": p["over"][2.5],
                "xG": f"{p['xg_a']:.2f}–{p['xg_b']:.2f}",
            })
        fx = pd.DataFrame(recs)
        st.dataframe(
            fx.style.format({
                "P(home)": "{:.0%}", "P(draw)": "{:.0%}", "P(away)": "{:.0%}",
                "Fair 1": fmt_odds, "Fair X": fmt_odds, "Fair 2": fmt_odds,
                "O2.5 %": "{:.0%}",
            }),
            hide_index=True, width="stretch", height=600,
        )


# --------------------------------------------------------- Tournament Sim

with tab_tournament:
    st.subheader("Klement-style tournament simulation")
    st.caption(
        "Joachim Klement called the last three World Cup winners not by "
        "predicting matches better than anyone else, but by simulating the "
        "**whole tournament** thousands of times and backing the team that "
        "lifts the trophy most often. This tab does the same with this app's "
        "match engine: every remaining group game is sampled from its "
        "Dixon-Coles score matrix, groups are ranked (points, GD, GF), the 8 "
        "best third-placed teams fill the real FIFA bracket, and knockouts "
        "run through extra time and penalties. **Games already played count "
        "as fact** — refresh the data daily and the forecast re-conditions "
        "on results so far."
    )
    n_sims = st.select_slider(
        "Tournament simulations", [2000, 5000, 10000, 20000], value=10000,
        help="More sims = smoother probabilities. 10,000 runs in a few seconds.",
    )
    data_stamp = f"{played['date'].max().date()}-{len(played)}"
    sim_table, sim_extras = get_simulation(n_sims, data_stamp)

    best_team = sim_table.index[0]
    (fin_a, fin_b), n_pair = sim_extras["final_pairs"].most_common(1)[0]
    p_ned = float(sim_table["Win cup"].get("Netherlands", 0.0))
    s1, s2, s3 = st.columns(3)
    s1.metric("Model's champion pick", best_team,
              f"wins {sim_table['Win cup'].iloc[0]:.1%} of sims",
              delta_color="off")
    s2.metric("Most likely final", f"{fin_a} v {fin_b}",
              f"{n_pair / sim_extras['n_sims']:.1%} of sims", delta_color="off")
    s3.metric("Klement's 2026 pick", "Netherlands",
              f"this model gives them {p_ned:.1%}", delta_color="off")

    top12 = sim_table.head(12).reset_index()
    sim_chart = (
        alt.Chart(top12).mark_bar().encode(
            x=alt.X("Win cup:Q", axis=alt.Axis(format="%"),
                    title="P(win the World Cup)"),
            y=alt.Y("Team:N", sort="-x", title=None),
            tooltip=["Team", alt.Tooltip("Win cup:Q", format=".1%")],
        ).properties(height=320)
    )
    st.altair_chart(sim_chart, use_container_width=True)

    show = sim_table.copy()
    show["Fair outright"] = [fmt_odds(1.0 / p) if p > 0 else "—"
                             for p in show["Win cup"]]
    pct_cols = ["Win group", "Make R32", "Make R16", "Make QF", "Make SF",
                "Make final", "Win cup"]
    st.dataframe(
        show.style.format({"Elo": "{:.0f}",
                           **{c: "{:.1%}" for c in pct_cols}}),
        width="stretch", height=600,
    )
    st.caption(
        "Each column is the probability of **reaching** that stage. "
        "'Fair outright' is the no-vig price to win the cup — a futures "
        "price longer than it is +EV."
    )

    with st.expander("💰 Outright futures value checker"):
        st.caption(
            "Futures markets carry much bigger margins than match markets "
            "(often 20–40% summed across 48 teams) and tie up bankroll for "
            "weeks — demand a larger edge than you would on a match bet."
        )
        f1, f2, f3 = st.columns([3, 3, 2])
        fut_team = f1.selectbox("Team", list(sim_table.index))
        fut_market = f2.selectbox("Market",
                                  ["Win the World Cup", "Reach the final"])
        fut_odds = odds_input(f3, "Book odds", "fut", "e.g. 9.00")
        if fut_odds:
            col = "Win cup" if fut_market == "Win the World Cup" else "Make final"
            p_fut = float(sim_table.loc[fut_team, col])
            ev = V.expected_value(p_fut, fut_odds)
            stake = (min(V.kelly_fraction(p_fut, fut_odds) * kelly_mult, 0.10)
                     * bankroll if ev >= min_edge else 0.0)
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Model probability", f"{p_fut:.1%}")
            g2.metric("Fair odds",
                      fmt_odds(1.0 / p_fut) if p_fut > 0 else "—")
            g3.metric("EV per $1", f"{ev:+.1%}")
            g4.metric("Kelly stake", f"${stake:,.2f}")
            if ev >= min_edge:
                st.success("Positive expected value at this price — but "
                           "remember the futures-margin caveat above.")
            else:
                st.info("No value at this price.")


# ------------------------------------------------------------ 2026 Record

with tab_record:
    st.subheader("Model record — every 2026 call, scored")
    st.caption(
        "For each completed match this reconstructs what the model said "
        "**before kickoff** — trained only on matches played earlier, the "
        "same no-peeking protocol as the backtest — and scores it against "
        "the final. Hit 🔄 Refresh in the sidebar after each matchday; new "
        "results land here automatically (the upstream dataset usually "
        "updates within a day)."
    )
    rec = get_live_record(DATA_STAMP)
    if rec.empty:
        st.info(
            "No completed 2026 World Cup matches in the dataset yet. "
            "Refresh after each matchday and the scoreboard fills in — "
            "every pick, every called score, ✅ or ❌."
        )
    else:
        rs = R.record_summary(rec)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Picks correct", f"{rs['pick_hits']}/{rs['matches']}",
                  f"{rs['accuracy']:.0%} — backtest baseline 57.8%",
                  delta_color="off")
        r2.metric("Exact scores called", f"{rs['score_hits']}/{rs['matches']}",
                  "most likely score = final score", delta_color="off")
        r3.metric("Brier score", f"{rs['brier']:.3f}",
                  "backtest 0.565 · guessing 0.667 · lower = better",
                  delta_color="off")
        r4.metric("Log loss", f"{rs['logloss']:.3f}",
                  "backtest 0.961 · guessing 1.099", delta_color="off")

        rec_show = pd.DataFrame({
            "Date": rec["date"].dt.date,
            "Match": rec["match"],
            "Final": rec["final"],
            "Model pick": [f"{p} ({q:.0%})"
                           for p, q in zip(rec["pick"], rec["p_pick"])],
            "Pick": np.where(rec["pick_hit"], "✅", "❌"),
            "Called score": [f"{s} ({q:.0%})"
                             for s, q in zip(rec["called_score"],
                                             rec["p_called_score"])],
            "Score": np.where(rec["score_hit"], "🎯", "—"),
            "P(what happened)": rec["p_outcome"],
        })
        st.dataframe(
            rec_show.style.format({"P(what happened)": "{:.0%}"}).apply(
                lambda r: ["background-color: rgba(46, 160, 67, 0.15)"
                           if r["Pick"] == "✅" else ""] * len(r),
                axis=1,
            ),
            hide_index=True, width="stretch",
            height=min(600, 60 + 35 * len(rec_show)),
        )
        st.caption(
            "**Does this feed back into the model?** Yes — the right way. "
            "Every refresh updates Elo with the new results and "
            "re-conditions the tournament sim, so tomorrow's predictions "
            "already know today's scores. What it deliberately does *not* "
            "do is re-tune model parameters on a handful of 2026 games — "
            "that's how you fit noise and get burned. Use this tab as the "
            "scoreboard: if Brier drifts well above the 0.565 baseline "
            "over 20+ matches, trust the sharp anchor more. "
            "Knockout caveat: the dataset records scores after extra time, "
            "so ET games are judged on the 120' score."
        )


# ------------------------------------------------------------ Bet Tracker

with tab_tracker:
    st.subheader("Bet tracker — prove the edge with closing line value")
    st.caption(
        "Log every bet **when you place it**. After kickoff, fill in the "
        "**closing odds** (the book's final pre-match price on your "
        "selection) and the result. If you consistently beat the close, you "
        "have a real edge no matter what this week's results say. "
        "⚠️ On Streamlit Cloud the log resets when the app reboots — "
        "download the CSV regularly, or do your real tracking on the local "
        "copy."
    )

    bet_log = TR.load_log()

    with st.form("add_bet", clear_on_submit=True):
        a1, a2 = st.columns([3, 2])
        match_opts = ["Type manually"] + [
            fmt_fixture(r) for _, r in WC_FIXTURES.iterrows()
        ]
        match_pick = a1.selectbox("Match", match_opts)
        match_manual = a2.text_input("Manual match (if not listed)")
        b1, b2, b3, b4 = st.columns([3, 2, 2, 2])
        bet_sel = b1.text_input("Market / selection",
                                placeholder="e.g. 1X2 — Draw, or Over 2.5")
        bet_odds = odds_input(b2, "Odds taken", "tr_odds", "e.g. 3.30")
        bet_stake = b3.number_input("Stake $", min_value=0.0, value=None,
                                    placeholder="25")
        bet_modelp = b4.number_input("Model prob % (optional)", 0.0, 100.0,
                                     value=None, placeholder="38")
        bet_submit = st.form_submit_button("➕ Log bet")

    if bet_submit:
        match_txt = (match_manual if match_pick == "Type manually"
                     else match_pick)
        if not (match_txt and bet_sel and bet_odds and bet_stake):
            st.warning("Need at least a match, selection, odds and stake.")
        else:
            new = pd.DataFrame([{
                "placed": str(date.today()), "match": match_txt,
                "selection": bet_sel, "odds": bet_odds, "stake": bet_stake,
                "model_p": bet_modelp / 100.0 if bet_modelp else None,
                "closing_odds": None, "result": "pending",
            }])
            bet_log = pd.concat([bet_log, new], ignore_index=True)
            TR.save_log(bet_log)
            st.success("Bet logged. Fill in closing odds + result after the match.")

    if bet_log.empty:
        st.info("No bets logged yet. Add your first one above — and from now "
                "on, log EVERY bet, including the losers. CLV only means "
                "something on a complete record.")
    else:
        st.caption(
            "Edit directly in the table (odds in **decimal** here). "
            "Set the result and closing odds as matches finish — changes "
            "save automatically."
        )
        edited = st.data_editor(
            bet_log, num_rows="dynamic", width="stretch", key="bet_editor",
            column_config={
                "placed": st.column_config.TextColumn("Placed"),
                "match": st.column_config.TextColumn("Match", width="large"),
                "selection": st.column_config.TextColumn("Selection"),
                "odds": st.column_config.NumberColumn(
                    "Odds taken (dec)", format="%.2f", min_value=1.01),
                "stake": st.column_config.NumberColumn(
                    "Stake $", format="$%.2f", min_value=0.0),
                "model_p": st.column_config.NumberColumn(
                    "Model p", format="%.3f", min_value=0.0, max_value=1.0,
                    help="Model probability as a fraction, e.g. 0.38"),
                "closing_odds": st.column_config.NumberColumn(
                    "Closing (dec)", format="%.2f", min_value=1.01),
                "result": st.column_config.SelectboxColumn(
                    "Result", options=TR.RESULTS),
            },
        )
        if not edited.equals(bet_log):
            TR.save_log(edited)
            bet_log = edited

        m = TR.with_metrics(bet_log)
        s = TR.summary(m)
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        t1.metric("Bets", f"{s['bets']} ({s['record']})")
        t2.metric("Staked", f"${s['staked']:,.2f}")
        t3.metric("P/L", f"${s['pl']:+,.2f}")
        t4.metric("ROI (settled)", f"{s['roi']:+.1%}")
        t5.metric("Avg CLV", "—" if s["avg_clv"] is None
                  else f"{s['avg_clv']:+.1%}",
                  "the number that matters", delta_color="off")
        t6.metric("Beat the close", "—" if s["beat_close"] is None
                  else f"{s['beat_close']:.0%}",
                  "aim for >50%", delta_color="off")

        mm = m.reset_index(drop=True)
        mm["Bet #"] = mm.index + 1
        mm["Actual P/L"] = mm["P/L $"].cumsum()
        mm["Expected P/L"] = mm["EV $"].fillna(0.0).cumsum()
        cum = mm.melt("Bet #", ["Actual P/L", "Expected P/L"],
                      var_name="Series", value_name="Cumulative $")
        clv_chart = (
            alt.Chart(cum).mark_line(point=True).encode(
                x="Bet #:Q",
                y="Cumulative $:Q",
                color=alt.Color("Series:N", title=None),
                tooltip=["Bet #", "Series",
                         alt.Tooltip("Cumulative $:Q", format="$,.2f")],
            ).properties(height=280)
        )
        st.altair_chart(clv_chart, use_container_width=True)
        st.caption(
            "If **Actual** tracks **Expected** over dozens of bets, the "
            "model's edges are real and you're just riding variance. If "
            "Actual lags far below Expected long-term, the edges were "
            "imaginary — tighten up (raise min EV, lean more on the sharp "
            "anchor)."
        )

        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Download log (CSV)", m.to_csv(index=False),
            file_name="bet_log.csv", width="stretch",
        )
        with d2.expander("⬆️ Restore log from CSV"):
            up = st.file_uploader("Upload a previously downloaded bet_log.csv",
                                  type="csv")
            if up is not None and st.button("Restore (replaces current log)"):
                TR.save_log(pd.read_csv(up))
                st.rerun()


# --------------------------------------------------------- Power Rankings

with tab_rankings:
    st.subheader("Elo power rankings")
    max_date = played["date"].max()
    year_ago = max_date - pd.Timedelta(days=365)
    rank_rows = []
    for i, (team, rating) in enumerate(model.elo.top(60), 1):
        prev = model.elo.rating_at(team, year_ago)
        rank_rows.append({
            "#": i, "Team": team, "Elo": rating, "1-yr Δ": rating - prev,
        })
    rk = pd.DataFrame(rank_rows)
    st.dataframe(
        rk.style.format({"Elo": "{:.0f}", "1-yr Δ": "{:+.0f}"}).map(
            lambda v: ("color: #2ea043" if isinstance(v, float) and v > 0
                       else ("color: #f85149" if isinstance(v, float) and v < 0 else "")),
            subset=["1-yr Δ"],
        ),
        hide_index=True, width="stretch", height=600,
    )


# -------------------------------------------------------- Model Validation

with tab_validate:
    st.subheader("Walk-forward backtest: World Cups 2006–2022")
    st.caption(
        "For each tournament the model was trained ONLY on matches played "
        "before it, then predicted every match in order — exactly how you'd "
        "use it live. No peeking."
    )
    bt, per_year, overall, group_only, cal = get_backtest()

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Matches predicted", f"{int(overall['matches'])}")
    b2.metric("Pick accuracy", f"{overall['accuracy']:.1%}",
              "vs 33.3% random / ~48% always-favorite-ish", delta_color="off")
    b3.metric("Brier score", f"{overall['brier']:.3f}",
              "0.667 = guessing, lower is better", delta_color="off")
    b4.metric("Log loss", f"{overall['log_loss']:.3f}",
              "1.099 = guessing, lower is better", delta_color="off")

    st.dataframe(
        per_year.style.format({
            "matches": "{:.0f}", "brier": "{:.3f}", "log_loss": "{:.3f}",
            "accuracy": "{:.1%}", "elo_favorite_acc": "{:.1%}",
        }),
        width="stretch",
    )

    st.subheader("Calibration — does 60% mean 60%?")
    st.caption(
        "Each point pools predictions in a probability bin. Points on the "
        "diagonal = the model's probabilities are trustworthy, which is what "
        "value betting lives or dies on."
    )
    cal_chart = (
        alt.Chart(cal).mark_circle(size=120).encode(
            x=alt.X("predicted:Q", title="Predicted probability",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("actual:Q", title="Actual frequency",
                    scale=alt.Scale(domain=[0, 1])),
            size=alt.Size("n:Q", legend=None),
            tooltip=["predicted", "actual", "n"],
        )
        + alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
    ).properties(height=350)
    st.altair_chart(cal_chart, use_container_width=True)

    st.info(
        "**Caveat:** this dataset records knockout scores after extra time, so "
        "a handful of knockout games are evaluated against the 120' result. "
        "Group-stage-only metrics (exact 90' market): "
        f"accuracy {group_only['accuracy']:.1%}, Brier {group_only['brier']:.3f}."
    )


# ------------------------------------------------------------ How It Works

with tab_help:
    st.subheader("How this finds an edge (and what it can't do)")
    st.markdown("""
#### The model
1. **Elo ratings** over 49,000+ international matches since 1872, using the
   World Football Elo methodology — match importance weights (World Cup ×3
   a friendly), goal-margin multipliers, +100 pts home advantage.
2. **Expected goals**: a Poisson regression maps the Elo gap to each team's
   goal expectation, with a 10-year half-life so recent form dominates, and a
   competitive-match adjustment (friendlies score differently).
3. **Dixon-Coles correction** fixes the known Poisson bias on low-scoring
   results (0-0, 1-1), fit by maximum likelihood.
4. The full **score matrix** prices every market: 1X2, totals, BTTS, correct
   score, and knockout advancement (extra time at 1/3 rates, penalties ≈ coin
   flip).
5. The **Tournament Sim** tab is the Joachim Klement trick (he called the
   2014/2018/2022 winners): simulate the *entire* tournament thousands of
   times — sample every remaining group game from its score matrix, rank
   groups, fill the real 2026 bracket (top two + 8 best thirds → round of
   32), and play the knockouts. Champion/stage probabilities are just how
   often each outcome happens. Klement's own inputs (GDP, population,
   temperature, FIFA points) are weaker proxies for what Elo measures
   directly — his edge is the simulation layer, which this tab replicates.

#### The edge
- **Win rate is the wrong target.** Winning 55% of even-money bets profits;
  winning 55% at odds 1.60 loses badly. What matters is **expected value**:
  `EV = your probability × decimal odds − 1`.
- The bookmaker's odds contain a **margin** (overround), typically 5–8% on
  World Cup matches. We strip it (Shin's method) to estimate the market's
  true opinion, then bet only where the model disagrees with the market by
  more than the margin.
- **Line shopping is the most reliable edge there is.** The same match is
  priced differently across books. Always enter the best available odds.
- **Kelly staking** sizes bets proportionally to the edge. Quarter Kelly is
  the default because full Kelly assumes your probabilities are exactly
  right, and they aren't.
- **The sharp anchor** (Match Analyzer): the devigged line at a sharp book
  (Pinnacle, Circa, exchanges) is the best single estimate of the truth —
  better than any public model. Blending the model with it (default 30/70)
  protects you from model blind spots; you still profit by betting SOFT
  books whose prices stray from the blend.
- **Closing line value** (Bet Tracker tab) is how professionals measure
  themselves: did you get a better price than the market's final word? Beat
  the close consistently and profit follows; results over any one week are
  just variance.
- **Dead rubbers are the model's blind spot.** When a team's round-of-32
  fate is sealed before matchday 3, expect rotation — the app flags these
  from the tournament sim. Skipping flagged games (or fading the
  unmotivated side) is usually right.

#### Honest expectations
- The backtest shows the model is **well-calibrated and competitive with
  published academic models** — but bookmaker closing lines are *also* very
  good. Most of your realized edge will come from line shopping, soft prices
  early in the week, and markets the books care less about (totals, BTTS).
- A World Cup is ~104 matches. Even with a genuine 4% average EV, variance
  over one tournament is enormous. **Judge the process (did you beat the
  closing line?), not one month of results.**
- The model knows nothing about **injuries, suspensions, rotation, or
  motivation** (e.g. a team already qualified resting starters in matchday
  3). When you know team news the model can't, skip the bet or adjust.
- Never bet money you can't afford to lose. If it stops being fun, stop.
""")
