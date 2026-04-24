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
trading_raw = D["trading_raw"]
date_range = D["date_range"]
instrument_info = D["instrument_info"]
co_trade_flow = D["co_trade_flow"]

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
        ["Dashboard", "Stock Detail", "Contributors", "Portfolio vs Benchmark", "Snapshot"],
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
        tri_clean = D["tri_clean"]
        daily_inst_value = D["daily_inst_value"]
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
# PAGE: Portfolio vs Benchmark
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Portfolio vs Benchmark":
    st.header("Portfolio vs Benchmark")
    st.caption(f"Benchmark: **{benchmark_name}**, both rebased to 0% at inception")

    # KPIs
    port_itd = period_data["ITD"][1]
    bm_itd = bm_period_returns["ITD"]
    port_1y = period_data["1Y"][1]
    bm_1y = bm_period_returns["1Y"]
    port_ytd = period_data["YTD"][1]
    bm_ytd = bm_period_returns["YTD"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio ITD", fmt_pct(port_itd), delta_color=delta_color(port_itd))
    c2.metric(f"{benchmark_name} ITD", fmt_pct(bm_itd), delta_color=delta_color(bm_itd))
    alpha_itd = (port_itd or 0) - (bm_itd or 0)
    c3.metric("Alpha ITD", fmt_pct(alpha_itd), delta_color=delta_color(alpha_itd))
    alpha_1y = (port_1y or 0) - (bm_1y or 0)
    c4.metric("Alpha 1Y", fmt_pct(alpha_1y), delta_color=delta_color(alpha_1y))
    alpha_ytd = (port_ytd or 0) - (bm_ytd or 0)
    c5.metric("Alpha YTD", fmt_pct(alpha_ytd), delta_color=delta_color(alpha_ytd))

    st.divider()

    # Performance chart (% returns, rebased to 0 at inception)
    port_pct = (growth_100 / 100 - 1) * 100
    bm_pct = (bm_growth_100 / 100 - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_pct.index, y=port_pct.values, name="Portfolio",
                             line=dict(color="#EF4444", width=2.5)))
    fig.add_trace(go.Scatter(x=bm_pct.index, y=bm_pct.values, name=benchmark_name,
                             line=dict(color="#C9D1D9", width=1.8)))
    fig.add_hline(y=0, line_dash="dash", line_color="#30363D")
    fig.update_xaxes(tickformat="%b-%Y", dtick="M6")
    fig.update_yaxes(title_text="Return (%)")
    fig.update_layout(
        height=500, hovermode="x unified",
        title="Portfolio Performance vs Benchmark",
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Period-by-period table
    st.subheader("Period-by-period comparison")
    rows = []
    for p in PERIODS:
        pr = period_data[p][1]
        br = bm_period_returns[p]
        alpha = (pr or 0) - (br or 0)
        rows.append({
            "Period": p,
            "Portfolio": fmt_pct(pr),
            "Benchmark": fmt_pct(br),
            "Alpha": fmt_pct(alpha),
            "Hit/Miss": "✓" if alpha > 0 else "✗",
        })
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
