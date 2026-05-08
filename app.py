"""
Portfolio performance dashboard — Streamlit version.

Run:
    pip install streamlit pandas numpy plotly
    streamlit run app.py

This app reads the pickle produced by build_v13_periods.py (v13_data.pkl).
To refresh the data, re-run the build pipeline:
    python build_v13_data.py && python build_v13_bm.py && python build_v13_periods.py
...then restart Streamlit (or click 'Refresh data' in the sidebar).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme polish
st.markdown("""
<style>
.main .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
[data-testid="stMetricLabel"] {font-size: 0.75rem; opacity: 0.7;}
div[data-testid="stHorizontalBlock"] {gap: 1rem;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "v13_data.pkl"


@st.cache_data(show_spinner="Loading portfolio data...")
def load_data(path_str: str, mtime: float):
    """Load the pickle. The mtime arg busts the cache when the file changes."""
    with open(path_str, "rb") as f:
        return pickle.load(f)


def get_data():
    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}. Run the build pipeline first.")
        st.stop()
    return load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)


D = get_data()

PERIODS = D["PERIODS"]
AS_OF = D["AS_OF"]
company_list = D["company_list"]
company_values = D["company_values"]
company_cum_contrib = D["company_cum_contrib"]
portfolio = D["portfolio"]
cash = D["cash"]
growth_100 = D["growth_100"]
all_period_contribs = D["all_period_contribs"]
period_data = D["period_data"]
period_std = D["period_std"]
period_portfolio_pl = D["period_portfolio_pl"]
period_stock_returns = D["period_stock_returns"]
period_stock_pl = D["period_stock_pl"]
period_stock_avgweight = D["period_stock_avgweight"]
stock_metrics = D["stock_metrics"]
bm_growth_100 = D["bm_growth_100"]
bm_period_returns = D["bm_period_returns"]
benchmark_name = D["benchmark_name"]

# Optional secondary benchmark (S&P 500). Falls back gracefully if older pickle.
HAS_BM2 = "bm2_growth_100" in D
if HAS_BM2:
    bm2_growth_100 = D["bm2_growth_100"]
    bm2_period_returns = D["bm2_period_returns"]
    benchmark_2_name = D["benchmark_2_name"]
else:
    bm2_growth_100 = None
    bm2_period_returns = None
    benchmark_2_name = None

trading_raw = D["trading_raw"]
date_range = D["date_range"]
instrument_info = D["instrument_info"]
co_trade_flow = D["co_trade_flow"]
tri_clean = D["tri_clean"]
instrument_qty = D["instrument_qty"]
daily_inst_value = D["daily_inst_value"]

held_companies = sorted([co for co in company_list if (company_values[co] > 0).any()])
currently_held = sorted([co for co in company_list if company_values.loc[AS_OF, co] > 10])


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt_gbp(v, decimals=0):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    sign = "-" if v < 0 else ""
    return f"{sign}£{abs(v):,.{decimals}f}"


def fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v*100:+.{decimals}f}%"


def delta_color(v):
    if v is None: return "off"
    return "normal" if v >= 0 else "inverse"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — global controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Portfolio")
    st.caption(f"As of **{AS_OF.strftime('%d %b %Y')}**")
    st.caption(f"Benchmark: **{benchmark_name}**")
    st.divider()

    page = st.radio(
        "Page",
        ["Dashboard", "Stock Detail", "Contributors", "Attribution Scatter",
         "Trade Performance", "Portfolio vs Benchmark", "Snapshot"],
        label_visibility="collapsed",
    )

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Re-run the build pipeline then click refresh.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.header("Dashboard")

    period = st.selectbox("Period", PERIODS, index=PERIODS.index("1Y"), key="dash_period")

    sd, ret = period_data[period]
    std = period_std[period]
    pl = period_portfolio_pl[period]

    # Top KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PORTFOLIO VALUE", fmt_gbp(D["PORTFOLIO_VALUE"]))
    c2.metric("SELECTED RETURN", fmt_pct(ret), delta_color=delta_color(ret))
    c3.metric("STD DEV (ANN)", fmt_pct(std, 1) if std is not None else "—", delta_color="off")
    c4.metric("SELECTED P&L", fmt_gbp(pl), delta_color=delta_color(pl))

    # Second row
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("CASH", fmt_gbp(D["CASH"]))
    c6.metric("OPEN POSITIONS", str(D["OPEN_COS"]))

    # Top/Bot contributors
    sorted_contribs = sorted(all_period_contribs[period].items(), key=lambda x: x[1], reverse=True)
    non_zero = [(n, v) for n, v in sorted_contribs if abs(v) > 1e-6]
    if non_zero:
        top_name, top_c = non_zero[0]
        bot_name, bot_c = non_zero[-1]
        top_r = period_stock_returns[period].get(top_name, 0)
        bot_r = period_stock_returns[period].get(bot_name, 0)
        c7.metric(
            "TOP CONTRIB",
            top_name,
            delta=f"Contrib {fmt_pct(top_c)} · Ret {fmt_pct(top_r)}",
            delta_color="normal",
        )
        c8.metric(
            "BOT CONTRIB",
            bot_name,
            delta=f"Contrib {fmt_pct(bot_c)} · Ret {fmt_pct(bot_r)}",
            delta_color="inverse",
        )

    st.divider()

    # Period-by-period summary table
    st.subheader("Period summary")
    rows = []
    for p in PERIODS:
        psd, pret = period_data[p]
        pstd = period_std[p]
        ppl = period_portfolio_pl[p]
        sorted_c = sorted(all_period_contribs[p].items(), key=lambda x: x[1], reverse=True)
        nz = [(n, v) for n, v in sorted_c if abs(v) > 1e-6]
        top = nz[0] if nz else ("—", 0)
        bot = nz[-1] if nz else ("—", 0)
        rows.append({
            "Period": p,
            "Start": psd.strftime("%d/%m/%Y"),
            "Return": fmt_pct(pret),
            "Std Dev": fmt_pct(pstd, 1) if pstd is not None else "—",
            "P&L": fmt_gbp(ppl),
            "Top contributor": f"{top[0]} ({fmt_pct(top[1])})",
            "Bottom contributor": f"{bot[0]} ({fmt_pct(bot[1])})",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Stock Detail
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Stock Detail":
    st.header("Stock Detail")

    c_sel, c_period = st.columns([2, 1])
    company = c_sel.selectbox("Company", held_companies,
                              index=held_companies.index("LEMONADE") if "LEMONADE" in held_companies else 0)
    period = c_period.selectbox("Period", PERIODS, index=PERIODS.index("1Y"), key="stock_period")

    # KPI row
    av = D["asof_values"].get(company, 0)
    wt = av / D["PORTFOLIO_VALUE"] if D["PORTFOLIO_VALUE"] else 0
    is_held = av > 10
    sel_ret = period_stock_returns[period].get(company, 0)
    sel_pl = period_stock_pl[period].get(company, 0)
    sel_contrib = all_period_contribs[period].get(company, 0)
    itd_contrib = all_period_contribs["ITD"].get(company, 0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AS-OF VALUE", fmt_gbp(av))
    k2.metric("CURRENT WEIGHT", f"{wt*100:.2f}%")
    k3.metric("CURRENTLY HELD", "Yes" if is_held else "No")
    k4.metric("SELECTED RETURN", fmt_pct(sel_ret), delta_color=delta_color(sel_ret))

    k5, k6, k7, _ = st.columns(4)
    k5.metric("SELECTED P&L", fmt_gbp(sel_pl), delta_color=delta_color(sel_pl))
    k6.metric("SELECTED CONTRIB", fmt_pct(sel_contrib), delta_color=delta_color(sel_contrib))
    k7.metric("ITD CONTRIB", fmt_pct(itd_contrib), delta_color=delta_color(itd_contrib))

    # Risk metrics
    m = stock_metrics.get(company, {}) or {}
    st.caption(
        f"Held days: **{m.get('holding_days', '—')}** · "
        f"Ann. return: **{fmt_pct(m.get('ann_return'))}** · "
        f"Std dev (ann): **{fmt_pct(m.get('std_ann'), 1)}** · "
        f"Sharpe: **{m.get('sharpe', 0):.2f}** · "
        f"Max DD: **{fmt_pct(m.get('max_dd'))}**"
    )

    st.divider()

    # Chart: cumulative return indexed to 100 + weight on secondary axis
    insts = [i for i, info in instrument_info.items() if info["display_co"] == company]
    cv_series = company_values[company]
    first_idx = cv_series[cv_series > 0].index
    if len(first_idx):
        fd = first_idx[0]
        co_trades = trading_raw[
            (trading_raw["Canonical"].isin(insts)) &
            (trading_raw["Direction"].isin(["Buy", "Sell"]))
        ]
        last_trade_date = co_trades["Trade date"].max() if not co_trades.empty else AS_OF
        still_held = cv_series.loc[AS_OF] > 0
        ld = AS_OF if still_held else last_trade_date
        ld_valid = date_range[date_range <= ld]
        if len(ld_valid): ld = ld_valid[-1]
        dates_co = date_range[(date_range >= fd) & (date_range <= ld)]

        # Build per-day weighted return across instruments
        daily_co_ret = pd.Series(0.0, index=dates_co)
        for j in range(1, len(dates_co)):
            d, dp = dates_co[j], dates_co[j-1]
            sum_prev = sum(daily_inst_value[i].get(dp, 0) for i in insts)
            if sum_prev == 0: continue
            for i in insts:
                v_prev = daily_inst_value[i].get(dp, 0)
                if v_prev == 0: continue
                tri_s = tri_clean.get(instrument_info[i]["tri_src"])
                if tri_s is None: continue
                t0s = tri_s[tri_s.index <= dp].dropna()
                t1s = tri_s[tri_s.index <= d].dropna()
                if t0s.empty or t1s.empty: continue
                t0, t1 = t0s.iloc[-1], t1s.iloc[-1]
                if t0 == 0: continue
                daily_co_ret[d] += (v_prev / sum_prev) * (t1/t0 - 1)

        cum = pd.Series(100.0, index=dates_co)
        for j in range(1, len(dates_co)):
            cum.iloc[j] = cum.iloc[j-1] * (1 + daily_co_ret.iloc[j])

        weight = pd.Series(0.0, index=dates_co)
        for d in dates_co:
            p = portfolio.get(d, 0)
            if p > 0: weight[d] = cv_series[d] / p * 100

        # Buy/sell markers
        trades_in_window = co_trades[
            (co_trades["Trade date"] >= fd) & (co_trades["Trade date"] <= ld)
        ]
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        for _, tr in trades_in_window.iterrows():
            td = tr["Trade date"]
            if td in cum.index:
                y = cum.loc[td]
                if tr["Direction"] == "Buy":
                    buy_x.append(td); buy_y.append(y)
                else:
                    sell_x.append(td); sell_y.append(y)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=cum.index, y=cum.values, name="Cum. return (100=start)",
                       line=dict(color="#3FB950", width=2.5)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=weight.index, y=weight.values, name="Weight %",
                       line=dict(color="#F0A020", width=1.5, dash="dot")),
            secondary_y=True,
        )
        if buy_x:
            fig.add_trace(
                go.Scatter(x=buy_x, y=buy_y, mode="markers", name="Buy",
                           marker=dict(color="#3FB950", size=10, symbol="triangle-up",
                                       line=dict(color="white", width=1))),
                secondary_y=False,
            )
        if sell_x:
            fig.add_trace(
                go.Scatter(x=sell_x, y=sell_y, mode="markers", name="Sell",
                           marker=dict(color="#EF4444", size=10, symbol="triangle-down",
                                       line=dict(color="white", width=1))),
                secondary_y=False,
            )
        fig.update_xaxes(tickformat="%b-%Y", dtick="M3")
        fig.update_yaxes(title_text="Indexed return (100 = first day)", secondary_y=False)
        fig.update_yaxes(title_text="Portfolio weight (%)", secondary_y=True)
        fig.update_layout(
            height=450, hovermode="x unified",
            title=f"{company} — Cumulative Return with Weight",
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cum contrib chart
        cc_series = company_cum_contrib[company].reindex(dates_co).fillna(0) * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=cc_series.index, y=cc_series.values,
                                  name="Cum contribution (%)",
                                  line=dict(color="#58A6FF", width=2)))
        fig2.add_hline(y=0, line_dash="dash", line_color="#30363D")
        fig2.update_xaxes(tickformat="%b-%Y", dtick="M3")
        fig2.update_yaxes(title_text="Contribution (%)")
        fig2.update_layout(
            height=300, hovermode="x unified",
            title="Cumulative Contribution to Portfolio Return",
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Trading history
    st.subheader("Trading history")
    company_insts = [i for i, info in instrument_info.items() if info["display_co"] == company]
    th = trading_raw[
        trading_raw["Canonical"].isin(company_insts) &
        trading_raw["Direction"].isin(["Buy", "Sell"])
    ].sort_values("Trade date").copy()
    if len(th):
        th_disp = th[["Trade date", "Direction", "EIKON Name", "Quantity", "Value (£)", "Description"]].copy()
        th_disp["Trade date"] = th_disp["Trade date"].dt.strftime("%d/%m/%Y")
        th_disp["Value (£)"] = th_disp["Value (£)"].apply(lambda v: fmt_gbp(v, 2))
        st.dataframe(th_disp, hide_index=True, use_container_width=True)
    else:
        st.info("No trades on record.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Contributors
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Contributors":
    st.header("Contributors")

    period = st.selectbox("Period", PERIODS, index=PERIODS.index("1Y"), key="contrib_period")

    sorted_c = sorted(all_period_contribs[period].items(), key=lambda x: x[1], reverse=True)
    non_zero = [(n, v) for n, v in sorted_c if abs(v) > 1e-6]
    top5 = non_zero[:5]
    bot5 = non_zero[-5:][::-1] if len(non_zero) >= 5 else []

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 5")
        rows = []
        for i, (n, contrib) in enumerate(top5, 1):
            rows.append({
                "#": i,
                "Company": n,
                "Avg Wt": f"{period_stock_avgweight[period].get(n, 0)*100:.2f}%",
                "Return": fmt_pct(period_stock_returns[period].get(n, 0)),
                "Contribution": fmt_pct(contrib),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with c2:
        st.subheader("Bottom 5")
        rows = []
        for i, (n, contrib) in enumerate(bot5, 1):
            rows.append({
                "#": i,
                "Company": n,
                "Avg Wt": f"{period_stock_avgweight[period].get(n, 0)*100:.2f}%",
                "Return": fmt_pct(period_stock_returns[period].get(n, 0)),
                "Contribution": fmt_pct(contrib),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Full attribution — all companies × all periods")
    st.caption(
        "Note: each period's contribution measures what a stock added DURING that window only. "
        "A stock that lost money then recovered can show a higher 3Y contribution than ITD. "
        "Peak/Trough show the full range of each stock's cumulative contribution journey."
    )

    peak_contrib = {co: company_cum_contrib[co].max() for co in company_list}
    trough_contrib = {co: company_cum_contrib[co].min() for co in company_list}

    rows = []
    for co in sorted(company_list):
        m = stock_metrics.get(co, {}) or {}
        av = D["asof_values"].get(co, 0)
        wt = av / D["PORTFOLIO_VALUE"] if D["PORTFOLIO_VALUE"] else 0
        row = {
            "Company": co,
            "Total Return": fmt_pct(m.get("total_return")),
            "Current Wt": f"{wt*100:.2f}%",
        }
        for p in PERIODS:
            cv = all_period_contribs[p].get(co, 0)
            row[p] = fmt_pct(cv) if abs(cv) > 1e-6 else "—"
        row["Peak"] = fmt_pct(peak_contrib.get(co, 0))
        row["Trough"] = fmt_pct(trough_contrib.get(co, 0))
        row["Held Days"] = m.get("holding_days", "—")
        row["Sharpe"] = f"{m.get('sharpe', 0):.2f}" if m.get("sharpe") is not None else "—"
        row["Max DD"] = fmt_pct(m.get("max_dd"))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=600)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Attribution Scatter
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Attribution Scatter":
    st.header("Attribution Scatter")
    st.caption(
        "Each bubble is a stock. **X-axis** = average weight in the portfolio during the period. "
        "**Y-axis** = the stock's own return during the period. **Bubble size** = absolute "
        "contribution to portfolio return. **Color** = whether the stock added (green) or detracted (red)."
    )

    period = st.selectbox("Period", PERIODS, index=PERIODS.index("1Y"), key="scatter_period")

    only_held = st.toggle(
        "Only show stocks held during the period",
        value=True,
        help="If off, includes companies that weren't held during the period (which sit at zero on both axes).",
    )

    # Build dataframe
    rows = []
    for co in company_list:
        avg_wt = period_stock_avgweight[period].get(co, 0)
        ret = period_stock_returns[period].get(co, 0)
        contrib = all_period_contribs[period].get(co, 0)
        # Skip stocks with zero weight (never held during the period)
        if only_held and avg_wt < 1e-6 and abs(ret) < 1e-6:
            continue
        rows.append({
            "Company": co,
            "Avg Weight (%)": avg_wt * 100,
            "Return (%)": ret * 100,
            "Contribution (%)": contrib * 100,
            "abs_contrib": abs(contrib),
            "color": "#3FB950" if contrib >= 0 else "#EF4444",
        })

    if not rows:
        st.info("No stocks with non-zero weight during this period.")
    else:
        df = pd.DataFrame(rows)

        # Compute bubble size: sqrt of |contribution|, scaled so the largest is ~50px
        max_abs = df["abs_contrib"].max() if df["abs_contrib"].max() > 0 else 1
        df["size"] = (df["abs_contrib"] / max_abs).pow(0.5) * 50 + 8

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Avg Weight (%)"],
            y=df["Return (%)"],
            mode="markers+text",
            text=df["Company"],
            textposition="top center",
            textfont=dict(size=9, color="#C9D1D9"),
            marker=dict(
                size=df["size"],
                color=df["color"],
                opacity=0.7,
                line=dict(color="white", width=1),
            ),
            customdata=df[["Contribution (%)", "Company"]].values,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Avg Weight: %{x:.2f}%<br>"
                "Return: %{y:+.2f}%<br>"
                "Contribution: %{customdata[0]:+.2f}%"
                "<extra></extra>"
            ),
        ))

        # Reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="#30363D")
        fig.add_vline(x=0, line_dash="dash", line_color="#30363D")

        fig.update_xaxes(title_text="Average Weight (%)", zeroline=False)
        fig.update_yaxes(title_text="Return (%)", zeroline=False)
        fig.update_layout(
            height=600,
            title=f"Stock Performance vs Portfolio Weight — {period}",
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Reading guide
        with st.expander("How to read this chart"):
            st.markdown(
                "- **Top-right green** — biggest wins. High weight, positive return, large bubble = these "
                "are the bets that drove the portfolio.\n"
                "- **Top-right red** — biggest disappointments. High weight, negative return = "
                "concentrated bets that didn't pay off.\n"
                "- **Bottom-left** — small positions, small impact regardless of direction.\n"
                "- **Far right (any color)** — your highest-conviction positions for this period; "
                "their direction matters most."
            )

        # Quadrant summary
        col1, col2, col3, col4 = st.columns(4)
        # Median weight as the line dividing 'high weight' vs 'low weight'
        med_wt = df["Avg Weight (%)"].median()
        win_big = df[(df["Avg Weight (%)"] >= med_wt) & (df["Return (%)"] > 0)]
        loss_big = df[(df["Avg Weight (%)"] >= med_wt) & (df["Return (%)"] <= 0)]
        win_small = df[(df["Avg Weight (%)"] < med_wt) & (df["Return (%)"] > 0)]
        loss_small = df[(df["Avg Weight (%)"] < med_wt) & (df["Return (%)"] <= 0)]
        col1.metric("Big winners", len(win_big), delta_color="off",
                    help="High weight + positive return")
        col2.metric("Big losers", len(loss_big), delta_color="off",
                    help="High weight + negative return")
        col3.metric("Small winners", len(win_small), delta_color="off",
                    help="Low weight + positive return")
        col4.metric("Small losers", len(loss_small), delta_color="off",
                    help="Low weight + negative return")

        # Detailed table
        st.subheader("Detail")
        df_disp = df[["Company", "Avg Weight (%)", "Return (%)", "Contribution (%)"]].copy()
        df_disp["Avg Weight (%)"] = df_disp["Avg Weight (%)"].apply(lambda v: f"{v:.2f}%")
        df_disp["Return (%)"] = df_disp["Return (%)"].apply(lambda v: f"{v:+.2f}%")
        df_disp["Contribution (%)"] = df_disp["Contribution (%)"].apply(lambda v: f"{v:+.2f}%")
        df_disp = df_disp.sort_values(
            "Contribution (%)",
            key=lambda col: col.str.replace("%", "").str.replace("+", "").astype(float),
            ascending=False,
        )
        st.dataframe(df_disp, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Trade Performance (per-trade scatter)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Trade Performance":
    st.header("Trade Performance")
    st.caption(
        "Each bubble is a trade event. **X-axis** = trade weight as % of portfolio "
        "(buys positive, sales negative). **Y-axis** = the stock's return from the trade "
        "date to the chosen endpoint. **Color** = transaction type."
    )

    # ── Compute per-trade records ──────────────────────────────────────────
    # Note: not using @st.cache_data here because newer Streamlit versions
    # don't reliably capture closure variables. The computation is fast
    # (~200 trades) so caching isn't really needed.
    def compute_trade_records():
        """
        Build a per-trade dataframe by walking the trade ledger in date order.
        Uses the split-adjusted instrument_qty series from the pickle for accurate
        Transaction Type classification (New Buy / Addition / Partial Sale / Complete Sale).
        Aggregates same-day, same-instrument trades.
        """
        df = trading_raw[trading_raw["Direction"].isin(["Buy", "Sell"])].copy()
        df["Trade date"] = pd.to_datetime(df["Trade date"])
        df["signed_qty"] = np.where(df["Direction"] == "Buy", df["Quantity"], -df["Quantity"])
        agg = df.groupby(["Trade date", "Canonical"], as_index=False).agg(
            signed_qty=("signed_qty", "sum"),
            abs_value=("Value (£)", lambda v: v.abs().sum()),
        )
        agg = agg[agg["signed_qty"].abs() > 1e-9]
        agg = agg.sort_values(["Trade date", "Canonical"]).reset_index(drop=True)

        records = []
        instrument_qty_local = D["instrument_qty"]   # pre-built, split-adjusted

        for _, row in agg.iterrows():
            inst = row["Canonical"]
            qty = row["signed_qty"]
            td = row["Trade date"]

            # Use the split-adjusted qty series for classification
            qty_series = instrument_qty_local.get(inst)
            if qty_series is not None and len(qty_series):
                prior_idx = qty_series.index[qty_series.index < td]
                prev_qty = float(qty_series[prior_idx[-1]]) if len(prior_idx) else 0.0
                post_idx = qty_series.index[qty_series.index >= td]
                new_qty = float(qty_series[post_idx[0]]) if len(post_idx) else prev_qty + qty
            else:
                prev_qty, new_qty = 0.0, qty

            # Classify based on direction and pre/post quantity
            if qty > 0:  # Buy
                tx_type = "New Buy" if prev_qty <= 1e-6 else "Addition"
            else:  # Sell
                tx_type = "Complete Sale" if abs(new_qty) < 1e-6 else "Partial Sale"

            display_co = instrument_info.get(inst, {}).get("display_co", inst)

            valid = date_range[date_range <= td]
            port_at = portfolio[valid[-1]] if not valid.empty else np.nan
            pct_portfolio = (row["abs_value"] / port_at * 100) if (port_at and port_at > 0) else np.nan

            records.append({
                "Trade Date": td,
                "Canonical": inst,
                "Company": display_co,
                "Transaction Type": tx_type,
                "% Portfolio Order": pct_portfolio,
                "Trade Value (£)": row["abs_value"],
            })

        records_df = pd.DataFrame(records)

        # Pre-compute "next trade date in same stock" for each record.
        # We group by display Company (not Canonical) so renamed/relisted instruments
        # are treated as one continuous holding for next-trade lookups.
        if not records_df.empty:
            records_df = records_df.sort_values(["Company", "Trade Date"]).reset_index(drop=True)
            records_df["Next Trade Date"] = (
                records_df.groupby("Company")["Trade Date"].shift(-1)
            )

        # Build prices dataframe (TRI series) keyed by display company
        prices = {}
        for inst, info in instrument_info.items():
            tri_s = tri_clean.get(info["tri_src"])
            if tri_s is None:
                continue
            display_co = info["display_co"]
            if display_co in prices:
                if len(tri_s) > len(prices[display_co]):
                    prices[display_co] = tri_s
            else:
                prices[display_co] = tri_s
        prices_df = pd.DataFrame(prices)

        return records_df, prices_df

    trade_records, prices_df = compute_trade_records()

    if trade_records.empty:
        st.info("No buy/sell trades found.")
        st.stop()

    # ── Sidebar filters (filter section appears in main area for this page) ─
    f1, f2 = st.columns([2, 1])
    with f1:
        st.markdown("**Trade date range** — only trades within this window are plotted")
        min_td = trade_records["Trade Date"].min().date()
        max_td = trade_records["Trade Date"].max().date()
        date_sel = st.date_input(
            "Trade date range",
            value=(min_td, max_td),
            min_value=min_td,
            max_value=max(max_td, AS_OF.date()),
            label_visibility="collapsed",
        )
        if isinstance(date_sel, tuple) and len(date_sel) == 2:
            sd_filter, ed_filter = date_sel
        else:
            sd_filter = date_sel if hasattr(date_sel, "year") else min_td
            ed_filter = max_td

    with f2:
        st.markdown("**Performance window** — how far forward to measure")
        perf_mode = st.radio(
            "Performance mode",
            ["Until next trade in same stock", "End of selected date range",
             "Latest available", "Fixed days after trade"],
            index=0,
            label_visibility="collapsed",
            help=(
                "**Until next trade in same stock** — best when you trade in and out of stocks. "
                "Each trade is measured from its date until the next trade in that stock, or "
                "today if it's the most recent. Buys → 'how did the stock do while I held it?' "
                "Sales → 'how did the stock do after I sold it?' \n\n"
                "**End of selected date range / Latest available** — measure to a fixed endpoint "
                "regardless of when you actually exited. Useful for evaluating buy decisions in "
                "isolation."
            ),
        )
        fixed_days = None
        if perf_mode == "Fixed days after trade":
            fixed_days = st.number_input(
                "Days after trade", min_value=1, max_value=3650, value=365, step=30
            )

    # Transaction type filter + display options
    f3, f4, f5 = st.columns([2, 1, 1])
    with f3:
        all_types = ["New Buy", "Addition", "Partial Sale", "Complete Sale"]
        present_types = [t for t in all_types if t in trade_records["Transaction Type"].unique()]
        selected_types = st.multiselect(
            "Transaction types",
            options=present_types,
            default=present_types,
        )
    with f4:
        log_y = st.checkbox(
            "Symmetric-log Y axis",
            value=False,
            help="Helpful when a few outliers compress the rest of the chart.",
        )
    with f5:
        clip_outliers = st.checkbox(
            "Clip outliers to ±500%",
            value=False,
            help="Caps performance below -100% or above +500% so the bulk is easier to read.",
        )

    # Company search and multiselect
    all_companies = sorted(trade_records["Company"].unique().tolist())
    currently_held_set = set(currently_held)

    current_only = st.toggle(
        f"📌 Current portfolio only ({len(currently_held_set)} holdings)",
        value=False,
        help="Show only trades for companies still held today.",
    )
    available_companies = sorted(
        c for c in all_companies if c in currently_held_set
    ) if current_only else all_companies

    c_search, c_pick = st.columns([1, 2])
    with c_search:
        company_search = st.text_input(
            "🔍 Search companies",
            placeholder="e.g. Tesla, Lemonade…",
        )
    with c_pick:
        if company_search:
            matching = [c for c in available_companies if company_search.lower() in c.lower()]
        else:
            matching = available_companies
        selected_companies = st.multiselect(
            f"Companies on chart ({len(matching)} match{'es' if len(matching) != 1 else ''})",
            options=matching,
            default=matching,
            placeholder="Pick one or more, or leave to show all matches",
        )

    # ── Apply filters ──────────────────────────────────────────────────────
    mask = (
        (trade_records["Trade Date"].dt.date >= sd_filter)
        & (trade_records["Trade Date"].dt.date <= ed_filter)
        & (trade_records["Transaction Type"].isin(selected_types))
        & (trade_records["Company"].isin(selected_companies))
    )
    filtered = trade_records.loc[mask].copy()

    # ── Determine endpoint and compute performance per trade ────────────────
    if perf_mode == "End of selected date range":
        endpoint = pd.Timestamp(ed_filter)
    elif perf_mode == "Latest available":
        endpoint = AS_OF
    else:
        endpoint = None  # used for "Fixed days" and "Until next trade" modes

    def _price_on_or_after(series: pd.Series, dt: pd.Timestamp):
        s = series.dropna()
        v = s[s.index >= dt]
        return float(v.iloc[0]) if len(v) else None

    def _price_on_or_before(series: pd.Series, dt: pd.Timestamp):
        s = series.dropna()
        v = s[s.index <= dt]
        return float(v.iloc[-1]) if len(v) else None

    perfs, end_used = [], []
    for _, r in filtered.iterrows():
        co = r["Company"]
        td = r["Trade Date"]
        if co not in prices_df.columns:
            perfs.append(np.nan); end_used.append(pd.NaT); continue
        s = prices_df[co]
        sp = _price_on_or_after(s, td)
        if perf_mode == "Fixed days after trade":
            target = min(td + pd.Timedelta(days=int(fixed_days)), prices_df.index.max())
            ep = _price_on_or_before(s, target)
            end_used.append(target)
        elif perf_mode == "Until next trade in same stock":
            # Use the next trade in this stock as the endpoint, or AS_OF if none
            next_td = r.get("Next Trade Date")
            target = next_td if pd.notna(next_td) else AS_OF
            ep = _price_on_or_before(s, target)
            end_used.append(target)
        else:
            ep = _price_on_or_before(s, endpoint)
            end_used.append(endpoint)
        if sp and ep and sp > 0:
            perfs.append((ep / sp - 1) * 100)
        else:
            perfs.append(np.nan)

    filtered["Performance %"] = perfs
    filtered["Measured Until"] = end_used
    plot_data = filtered.dropna(subset=["Performance %"]).copy()

    plot_data["Performance % (plot)"] = (
        plot_data["Performance %"].clip(-100, 500) if clip_outliers else plot_data["Performance %"]
    )

    # Signed weight: sales negative
    SALE_TYPES = {"Partial Sale", "Complete Sale"}
    plot_data["Signed Weight"] = np.where(
        plot_data["Transaction Type"].isin(SALE_TYPES),
        -plot_data["% Portfolio Order"],
        plot_data["% Portfolio Order"],
    )
    plot_data["Perf Display"] = plot_data["Performance %"].apply(lambda x: f"{x:+.0f}%")

    # ── KPI rows: Buys vs Sales ────────────────────────────────────────────
    if perf_mode == "End of selected date range":
        st.markdown(f"_Performance measured to **{ed_filter.strftime('%d %b %Y')}**._")
    elif perf_mode == "Latest available":
        st.markdown(f"_Performance measured to **{AS_OF.strftime('%d %b %Y')}** (latest)._")
    elif perf_mode == "Until next trade in same stock":
        st.markdown(
            "_Performance measured from each trade until the **next trade in that stock** "
            "(or today, if it's the most recent). Buys → 'how did the stock do while I held it?' "
            "Sales → 'how did the stock do while I was out of position?'_"
        )
    else:
        st.markdown(f"_Performance measured **{fixed_days} days** after each trade._")

    buy_data = plot_data[~plot_data["Transaction Type"].isin(SALE_TYPES)]
    sale_data = plot_data[plot_data["Transaction Type"].isin(SALE_TYPES)]

    def _kpi_row(df, label, is_sales=False):
        st.markdown(f"**{label}** — {len(df)} trade{'s' if len(df) != 1 else ''}")
        c1, c2, c3, c4, c5 = st.columns(5)
        if len(df):
            c1.metric("Mean", f"{df['Performance %'].mean():+.0f}%")
            c2.metric("Median", f"{df['Performance %'].median():+.0f}%")
            c3.metric("% positive", f"{(df['Performance %'] > 0).mean()*100:.0f}%")
            if is_sales:
                c4.metric("Best (avoided drop)", f"{df['Performance %'].min():+.0f}%",
                          help="Biggest drop after sale — best-timed exit.")
                c5.metric("Worst (left on table)", f"{df['Performance %'].max():+.0f}%",
                          help="Biggest rise after sale — premature exit.")
            else:
                c4.metric("Best", f"{df['Performance %'].max():+.0f}%")
                c5.metric("Worst", f"{df['Performance %'].min():+.0f}%")
        else:
            for c in (c1, c2, c3, c4, c5):
                c.metric("—", "—")

    _kpi_row(buy_data, "🟢 Buys (New Buy + Addition)")
    _kpi_row(sale_data, "🔴 Sales (Partial + Complete)", is_sales=True)

    # ── Scatter plot ───────────────────────────────────────────────────────
    # Match the existing dark-theme palette
    COLOR_MAP = {
        "New Buy":       "#3FB950",   # green
        "Addition":      "#58A6FF",   # blue
        "Partial Sale":  "#F0A020",   # amber
        "Complete Sale": "#EF4444",   # red
    }

    if len(plot_data) == 0:
        st.warning("No trades match the current filters.")
    else:
        fig = px.scatter(
            plot_data,
            x="Signed Weight",
            y="Performance % (plot)",
            color="Transaction Type",
            color_discrete_map=COLOR_MAP,
            category_orders={"Transaction Type": ["New Buy", "Addition", "Partial Sale", "Complete Sale"]},
            custom_data=["Company", "Trade Date", "Transaction Type",
                         "% Portfolio Order", "Perf Display"],
        )
        fig.update_traces(
            marker=dict(size=11, opacity=0.78,
                        line=dict(width=0.5, color="rgba(255,255,255,0.6)")),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Date: %{customdata[1]|%d %b %Y}<br>"
                "Type: %{customdata[2]}<br>"
                "Trade weight: %{customdata[3]:.2f}%<br>"
                "Performance: %{customdata[4]}"
                "<extra></extra>"
            ),
        )
        # Quadrant guide lines
        fig.add_hline(y=0, line_dash="dash", line_color="#30363D", opacity=0.7)
        fig.add_vline(x=0, line_dash="dash", line_color="#30363D", opacity=0.7)

        # Quadrant labels (paper coords)
        quadrant_font = dict(size=11, color="rgba(160,160,160,0.7)")
        fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.98,
                           text="Good buys<br>(bought, went up)",
                           showarrow=False, align="right",
                           font=quadrant_font, xanchor="right", yanchor="top")
        fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.02,
                           text="Poor buys<br>(bought, went down)",
                           showarrow=False, align="right",
                           font=quadrant_font, xanchor="right", yanchor="bottom")
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.98,
                           text="Premature sales<br>(sold, kept rising)",
                           showarrow=False, align="left",
                           font=quadrant_font, xanchor="left", yanchor="top")
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.02,
                           text="Avoided losses<br>(sold, then dropped)",
                           showarrow=False, align="left",
                           font=quadrant_font, xanchor="left", yanchor="bottom")

        fig.update_layout(
            height=620,
            hovermode="closest",
            xaxis_title="Trade weight (% of portfolio) — buys positive, sales negative",
            yaxis_title="Performance (%)" + (" – clipped" if clip_outliers else ""),
            yaxis=dict(tickformat=".0f", hoverformat=".0f"),
            xaxis=dict(tickformat=".2f", hoverformat=".2f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, title=None),
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)", zeroline=False)

        if log_y:
            fig.update_yaxes(type="log")
        else:
            # Force symmetric axes by adding invisible anchor points at corners
            x_abs = float(plot_data["Signed Weight"].abs().max() or 0) * 1.08
            y_abs = float(plot_data["Performance % (plot)"].abs().max() or 0) * 1.08
            if x_abs == 0: x_abs = 1
            if y_abs == 0: y_abs = 1
            fig.add_trace(go.Scatter(
                x=[-x_abs, x_abs, -x_abs, x_abs],
                y=[-y_abs, -y_abs, y_abs, y_abs],
                mode="markers",
                marker=dict(size=0.1, opacity=0),
                hoverinfo="skip",
                showlegend=False,
            ))

        st.plotly_chart(fig, use_container_width=True)

    # ── Breakdown by transaction type ──────────────────────────────────────
    if len(plot_data):
        st.subheader("Breakdown by transaction type")
        summary = (
            plot_data.groupby("Transaction Type")["Performance %"]
            .agg(["count", "mean", "median", "min", "max"])
            .rename(columns={"count": "Trades", "mean": "Mean %",
                             "median": "Median %", "min": "Min %", "max": "Max %"})
            .reindex(["New Buy", "Addition", "Partial Sale", "Complete Sale"])
            .dropna(how="all")
        )
        summary[["Mean %", "Median %", "Min %", "Max %"]] = (
            summary[["Mean %", "Median %", "Min %", "Max %"]].round(1)
        )
        st.dataframe(summary, use_container_width=True)

    # ── Underlying trade data table + download ─────────────────────────────
    with st.expander("📋 View underlying trade data"):
        if len(plot_data):
            show = plot_data[[
                "Trade Date", "Company", "Transaction Type",
                "% Portfolio Order", "Trade Value (£)", "Performance %"
            ]].copy()
            show["Trade Date"] = show["Trade Date"].dt.strftime("%Y-%m-%d")
            show["% Portfolio Order"] = show["% Portfolio Order"].round(2)
            show["Trade Value (£)"] = show["Trade Value (£)"].round(0)
            show["Performance %"] = show["Performance %"].round(0)
            show = show.sort_values("Trade Date", ascending=False).reset_index(drop=True)
            st.dataframe(show, use_container_width=True, height=400)

            st.download_button(
                "Download filtered data as CSV",
                show.to_csv(index=False).encode("utf-8"),
                file_name="trade_performance.csv",
                mime="text/csv",
            )
        else:
            st.info("No trades match the current filters.")

    st.caption(
        "Performance uses each company's total-return index. New Buys = first time entering a "
        "position. Additions = adding to an existing position. Partial / Complete Sales speak for "
        "themselves. Same-day trades for the same instrument are aggregated."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Portfolio vs Benchmark
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Portfolio vs Benchmark":
    st.header("Portfolio vs Benchmarks")
    bench_caption = f"Primary: **{benchmark_name}**"
    if HAS_BM2:
        bench_caption += f"  ·  Secondary: **{benchmark_2_name}**"
    bench_caption += "  ·  All rebased to 0% at inception"
    st.caption(bench_caption)

    # KPIs — primary benchmark
    port_itd = period_data["ITD"][1]
    port_1y = period_data["1Y"][1]
    port_ytd = period_data["YTD"][1]
    bm_itd = bm_period_returns["ITD"]
    bm_1y = bm_period_returns["1Y"]
    bm_ytd = bm_period_returns["YTD"]

    st.markdown(f"##### Alpha vs {benchmark_name}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio ITD", fmt_pct(port_itd), delta_color=delta_color(port_itd))
    c2.metric(f"{benchmark_name[:18]} ITD", fmt_pct(bm_itd), delta_color=delta_color(bm_itd))
    alpha_itd = (port_itd or 0) - (bm_itd or 0)
    c3.metric("Alpha ITD", fmt_pct(alpha_itd), delta_color=delta_color(alpha_itd))
    alpha_1y = (port_1y or 0) - (bm_1y or 0)
    c4.metric("Alpha 1Y", fmt_pct(alpha_1y), delta_color=delta_color(alpha_1y))
    alpha_ytd = (port_ytd or 0) - (bm_ytd or 0)
    c5.metric("Alpha YTD", fmt_pct(alpha_ytd), delta_color=delta_color(alpha_ytd))

    if HAS_BM2:
        st.markdown(f"##### Alpha vs {benchmark_2_name}")
        bm2_itd = bm2_period_returns["ITD"]
        bm2_1y = bm2_period_returns["1Y"]
        bm2_ytd = bm2_period_returns["YTD"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Portfolio ITD", fmt_pct(port_itd), delta_color=delta_color(port_itd))
        c2.metric(f"{benchmark_2_name[:18]} ITD", fmt_pct(bm2_itd), delta_color=delta_color(bm2_itd))
        alpha2_itd = (port_itd or 0) - (bm2_itd or 0)
        c3.metric("Alpha ITD", fmt_pct(alpha2_itd), delta_color=delta_color(alpha2_itd))
        alpha2_1y = (port_1y or 0) - (bm2_1y or 0)
        c4.metric("Alpha 1Y", fmt_pct(alpha2_1y), delta_color=delta_color(alpha2_1y))
        alpha2_ytd = (port_ytd or 0) - (bm2_ytd or 0)
        c5.metric("Alpha YTD", fmt_pct(alpha2_ytd), delta_color=delta_color(alpha2_ytd))

    st.divider()

    # Performance chart (% returns, rebased to 0 at inception)
    port_pct = (growth_100 / 100 - 1) * 100
    bm_pct = (bm_growth_100 / 100 - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_pct.index, y=port_pct.values, name="Portfolio",
                             line=dict(color="#EF4444", width=2.5)))
    fig.add_trace(go.Scatter(x=bm_pct.index, y=bm_pct.values, name=benchmark_name,
                             line=dict(color="#C9D1D9", width=1.8)))
    if HAS_BM2:
        bm2_pct = (bm2_growth_100 / 100 - 1) * 100
        fig.add_trace(go.Scatter(x=bm2_pct.index, y=bm2_pct.values, name=benchmark_2_name,
                                 line=dict(color="#58A6FF", width=1.8)))
    fig.add_hline(y=0, line_dash="dash", line_color="#30363D")
    fig.update_xaxes(tickformat="%b-%Y", dtick="M6")
    fig.update_yaxes(title_text="Return (%)")
    fig.update_layout(
        height=500, hovermode="x unified",
        title="Portfolio Performance vs Benchmarks",
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Period-by-period table — both benchmarks side by side
    st.subheader("Period-by-period comparison")
    rows = []
    for p in PERIODS:
        pr = period_data[p][1]
        br = bm_period_returns[p]
        alpha = (pr or 0) - (br or 0)
        row = {
            "Period": p,
            "Portfolio": fmt_pct(pr),
            benchmark_name: fmt_pct(br),
            f"Alpha vs {benchmark_name[:14]}": fmt_pct(alpha),
        }
        if HAS_BM2:
            br2 = bm2_period_returns[p]
            alpha2 = (pr or 0) - (br2 or 0)
            row[benchmark_2_name] = fmt_pct(br2)
            row[f"Alpha vs {benchmark_2_name[:14]}"] = fmt_pct(alpha2)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Snapshot
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Snapshot":
    st.header("Portfolio Snapshot")
    st.caption("Pick any historical date to see every holding's value, weight, cumulative contribution and P&L.")

    default_date = AS_OF.date()
    min_date = date_range[0].date()
    picked = st.date_input("Date", value=default_date, min_value=min_date, max_value=default_date)

    # Snap to nearest available business date
    picked_ts = pd.Timestamp(picked)
    valid = date_range[date_range <= picked_ts]
    if valid.empty:
        st.warning("Picked date is before the portfolio started.")
        st.stop()
    dt = valid[-1]

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio value", fmt_gbp(portfolio[dt]))
    c2.metric("Securities", fmt_gbp(D["securities"][dt]))
    c3.metric("Cash", fmt_gbp(cash[dt]))
    si_pl = D["sinc_pl"][dt]
    c4.metric("SI P&L", fmt_gbp(si_pl), delta_color=delta_color(si_pl))
    c5.metric("Growth of £100", f"{growth_100[dt]:.2f}")

    if dt != picked_ts:
        st.caption(f"(Displaying nearest available date: {dt.strftime('%d %b %Y')})")

    st.divider()

    # Holdings table (currently held on this date only)
    show_all = st.toggle("Show all ever-held (including zero-value)", value=False)

    co_cum_trade_flow = co_trade_flow.cumsum()
    port_val = portfolio[dt] if portfolio[dt] else 1
    rows = []
    for co in company_list:
        v = company_values.loc[dt, co]
        if not show_all and v <= 10: continue
        flow = co_cum_trade_flow.loc[dt, co] if co in co_cum_trade_flow.columns else 0
        pl = v + flow   # value today + net trade flow (neg buys + pos sells) = unrealised+realised P&L
        cc = company_cum_contrib.loc[dt, co]
        rows.append({
            "Company": co,
            "Value": fmt_gbp(v),
            "Weight": f"{v/port_val*100:.2f}%" if v > 0 else "—",
            "Cum contrib (ITD)": fmt_pct(cc),
            "P&L since inception": fmt_gbp(pl),
            "_val": v,
        })
    df = pd.DataFrame(rows).sort_values("_val", ascending=False).drop(columns=["_val"])
    st.dataframe(df, hide_index=True, use_container_width=True, height=600)

    # Total row
    held_rows = [r for r in rows if float(r["Weight"].rstrip("%")) > 0 if r["Weight"] != "—"]
    st.caption(f"{len(held_rows)} companies held on this date")
