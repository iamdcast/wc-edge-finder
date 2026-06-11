"""WC Edge Finder — World Cup match analysis & betting value dashboard.

Run with:  .venv/bin/streamlit run app.py
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from wc_edge import tournament as T
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


# ----------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("⚽ WC Edge Finder")
    st.caption(
        "Find **value**, not winners. A bet is good when the book's implied "
        "probability is lower than the true one — even if it loses more often "
        "than it wins."
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

(tab_analyze, tab_fixtures, tab_tournament, tab_rankings, tab_validate,
 tab_help) = st.tabs(
    ["🎯 Match Analyzer", "📅 Fixtures", "🏆 Tournament Sim",
     "📊 Power Rankings", "🧪 Model Validation", "📖 How It Works"]
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
    pick, p_pick = max(outcomes, key=lambda kv: kv[1])
    (bi, bj), p_score = pred["top_scores"][0]
    conf = ("strong" if p_pick >= 0.60 else
            "moderate" if p_pick >= 0.45 else "slight — close to a coin flip")
    pick_line = (f"**Model pick: {pick}** ({p_pick:.0%}, {conf} confidence) · "
                 f"most likely score **{bi}–{bj}** ({p_score:.0%})")
    if knockout:
        adv = team_a if pred["advance_a"] >= 0.5 else team_b
        pick_line += (f" · **{adv} to advance** "
                      f"({max(pred['advance_a'], pred['advance_b']):.0%})")
    st.info(pick_line)

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
            [(f"{team_a} win", pred["home"], odds_h),
             ("Draw", pred["draw"], odds_d),
             (f"{team_b} win", pred["away"], odds_a)],
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
        bets = [r for r in rows if r["Verdict"] == "BET"]
        if bets:
            total = sum(r["Stake $"] for r in bets)
            st.success(
                f"**{len(bets)} value bet(s) found** — total recommended stake "
                f"${total:,.2f} ({total / bankroll:.1%} of bankroll). "
                f"Remember: +EV bets still lose all the time. The edge only "
                f"shows up over dozens of bets."
            )
        else:
            st.info(
                "No value at these odds — the book's price is at or better than "
                "the model's. **Not betting is the correct play more often than "
                "not.** Check another book or another market."
            )
        check = [r for r in rows if r["Verdict"] == "CHECK INPUTS"]
        if check:
            st.warning(
                "Some edges look implausibly large (>15%). Double-check you "
                "entered the odds correctly and that there's no team news "
                "(injuries, rotation) the model can't know about."
            )
    else:
        st.info("Enter a full market (e.g. all three 1X2 odds) to get the value report.")

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
        recs = []
        for _, r in WC_FIXTURES.iterrows():
            p = model.predict(r["home_team"], r["away_team"],
                              venue=venue_for_fixture(r))
            probs = {r["home_team"]: p["home"], "Draw": p["draw"],
                     r["away_team"]: p["away"]}
            pick = max(probs, key=probs.get)
            recs.append({
                "Date": r["date"].date(),
                "Match": f"{r['home_team']}{'' if r['neutral'] else ' 🏠'} vs {r['away_team']}",
                "City": r["city"],
                "Model pick": f"{pick} ({probs[pick]:.0%})",
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
