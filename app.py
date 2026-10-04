"""Marine Operations - Submarine cable laying tracker.

Input : the monthly CSV report(s) (one snapshot per project per month).
Views : 1. Portfolio overview (selected month)   2. Project detail (monthly trend + forecast)
        3. Map (routes, ports, estimated position of the laying vessel)
Run   : streamlit run app.py
Without upload, the app falls back to SYNTHETIC files in ./sample_data.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Cable Laying Tracker", page_icon="🌊", layout="wide")

SAMPLE_DIR = Path(__file__).parent / "sample_data"
STATUS_COLORS = {"On track": "#2E9E6B", "At risk": "#E8A33D", "Late": "#D64550", "No end date": "#9AA5B1"}
S_DOMAIN, S_RANGE = list(STATUS_COLORS), list(STATUS_COLORS.values())

# Column-name hints (FR/EN), tried in this order; a column is claimed only once.
ALIASES = {
    "planned_end": ["fin", "end"],
    "project": ["projet", "project", "nom", "name"],
    "date": ["date", "mois", "month", "periode", "période"],
    "deployed": ["deploy", "déploy", "laid", "pos"],
    "remaining": ["restant", "remain", "reste"],
    "planned": ["prevu", "prévu", "planned", "total", "plan"],
}
NONE = "(none)"
# Optional location columns: (kind tokens, side tokens) - a column must match one of each.
GEO = {
    "start_name": (["port", "nom", "name"], ["depart", "start", "origin", "from"]),
    "start_lat": (["lat"], ["depart", "start", "origin", "from"]),
    "start_lon": (["lon"], ["depart", "start", "origin", "from"]),
    "end_name": (["port", "nom", "name"], ["arriv", "end", "dest"]),
    "end_lat": (["lat"], ["arriv", "end", "dest"]),
    "end_lon": (["lon"], ["arriv", "end", "dest"]),
}


# ------------------------------------------------------------- loading ----
def read_csv(src, name: str) -> pd.DataFrame:
    """Read a CSV with automatic separator detection (',' ';' tab) - everything as text."""
    df = pd.read_csv(src, sep=None, engine="python", dtype=str, encoding_errors="replace")
    df.columns = [c.strip() for c in df.columns]
    df["_source"] = name
    return df


def to_num(s: pd.Series) -> pd.Series:
    """Handle '1 234,5' and '1234.5' alike."""
    cleaned = (s.astype(str).str.replace("\u202f", "", regex=False).str.replace("\xa0", "", regex=False)
               .str.replace(" ", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(cleaned, errors="coerce")


def guess_columns(columns: list[str]) -> dict:
    claimed, guess = set(), {}
    for key, (kind, side) in GEO.items():
        match = next((c for c in columns if c not in claimed and c != "_source"
                      and any(k in c.lower() for k in kind) and any(x in c.lower() for x in side)), None)
        guess[key] = match
        if match:
            claimed.add(match)
    for key, hints in ALIASES.items():
        match = next((c for c in columns if c not in claimed and c != "_source"
                      and any(h in c.lower() for h in hints)), None)
        guess[key] = match
        if match:
            claimed.add(match)
    return guess


def load_raw() -> tuple[pd.DataFrame, bool]:
    uploads = st.sidebar.file_uploader("Monthly CSV report(s)", type="csv", accept_multiple_files=True,
                                       help="Drop one or several monthly files; they are stacked together.")
    if uploads:
        return pd.concat([read_csv(u, u.name) for u in uploads], ignore_index=True), False
    files = sorted(SAMPLE_DIR.glob("*.csv"))
    if not files:
        st.warning("Upload a monthly CSV to get started.")
        st.stop()
    st.sidebar.info("Showing SYNTHETIC sample data. Upload your own CSV to replace it.")
    return pd.concat([read_csv(f, f.name) for f in files], ignore_index=True), True


def clean(raw: pd.DataFrame, m: dict) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame({
        "project": raw[m["project"]].str.strip(),
        "date": pd.to_datetime(raw[m["date"]], dayfirst=True, errors="coerce"),
        "planned_km": to_num(raw[m["planned"]]),
        "deployed_km": to_num(raw[m["deployed"]]),
    })
    df["remaining_km"] = to_num(raw[m["remaining"]]) if m["remaining"] != NONE else np.nan
    df["planned_end"] = (pd.to_datetime(raw[m["planned_end"]], dayfirst=True, errors="coerce")
                         if m["planned_end"] != NONE else pd.NaT)
    for k in GEO:
        if m.get(k, NONE) == NONE:
            df[k] = np.nan if k.endswith(("lat", "lon")) else None
        else:
            df[k] = to_num(raw[m[k]]) if k.endswith(("lat", "lon")) else raw[m[k]].str.strip()
    n_in = len(df)
    df = df.dropna(subset=["project", "date", "planned_km", "deployed_km"])
    df = df.drop_duplicates(["project", "date"], keep="last")
    df["remaining_km"] = df["remaining_km"].fillna(df.planned_km - df.deployed_km)
    checks = {
        "Rows read": n_in,
        "Rows dropped (missing/invalid values)": n_in - len(df),
        "Deployed > planned": int((df.deployed_km > df.planned_km + 0.5).sum()),
        "Deployed + remaining ≠ planned": int(((df.deployed_km + df.remaining_km - df.planned_km).abs() > 1).sum()),
        "Deployed decreased vs previous month": int(
            (df.sort_values("date").groupby("project").deployed_km.diff() < -0.5).sum()),
    }
    return df.sort_values(["project", "date"]), checks


# ------------------------------------------------------------- geometry ----
def gc_path(lat1, lon1, lat2, lon2, frac=1.0, n=48):
    """Points [lon, lat] along the great circle from A towards B, up to `frac` of the way."""
    def vec(lat, lon):
        lat, lon = np.radians(lat), np.radians(lon)
        return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    a, b = vec(lat1, lon1), vec(lat2, lon2)
    omega = np.arccos(np.clip(a @ b, -1, 1))
    frac = float(np.clip(frac, 0, 1))
    if omega < 1e-9 or frac == 0:
        return [[lon1, lat1], [lon1, lat1]]
    pts = []
    for t in np.linspace(0, frac, max(3, int(n * frac))):
        v = (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)
        pts.append([float(np.degrees(np.arctan2(v[1], v[0]))), float(np.degrees(np.arcsin(np.clip(v[2], -1, 1))))])
    return pts


def hex_rgb(h: str) -> list[int]:
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


# ---------------------------------------------------------- KPI engine ----
def summarize(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for project, h in df[df.date <= as_of].groupby("project"):
        last = h.iloc[-1]
        win = h.tail(3)  # laying rate over the last (up to) 3 snapshots
        days = (win.date.iloc[-1] - win.date.iloc[0]).days
        rate = (win.deployed_km.iloc[-1] - win.deployed_km.iloc[0]) / days if days > 0 else np.nan
        remaining = max(last.remaining_km, 0)
        if remaining <= 0.5:
            forecast_end = last.date
        elif rate and rate > 0:
            forecast_end = last.date + pd.Timedelta(days=int(np.ceil(remaining / rate)))
        else:
            forecast_end = pd.NaT
        delay = (forecast_end - last.planned_end).days if pd.notna(forecast_end) and pd.notna(last.planned_end) else np.nan
        status = ("No end date" if np.isnan(delay) else "On track" if delay <= 7 else "At risk" if delay <= 21 else "Late")
        rows.append({
            "project": project, "report_date": last.date, "planned_km": last.planned_km,
            "deployed_km": last.deployed_km, "remaining_km": remaining,
            "progress": last.deployed_km / last.planned_km, "rate_km_day": rate,
            "planned_end": last.planned_end, "forecast_end": forecast_end,
            "delay_days": delay, "status": status, **{k: last[k] for k in GEO},
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- sidebar ----
st.sidebar.title("🌊 Marine Operations")
raw, is_sample = load_raw()
cols = [c for c in raw.columns if c != "_source"]
guess = guess_columns(cols)

with st.sidebar.expander("Column mapping", expanded=False):
    mapping = {}
    for key, label, optional in [("project", "Project", False), ("date", "Report date", False),
                                 ("planned", "Planned length (km)", False), ("deployed", "Deployed length (km)", False),
                                 ("remaining", "Remaining length (km)", True), ("planned_end", "Planned end date", True),
                                 ("start_name", "Start port (name)", True), ("start_lat", "Start latitude", True),
                                 ("start_lon", "Start longitude", True), ("end_name", "End port (name)", True),
                                 ("end_lat", "End latitude", True), ("end_lon", "End longitude", True)]:
        options = ([NONE] if optional else []) + cols
        default = guess[key] if guess[key] in options else options[0]
        mapping[key] = st.selectbox(label, options, index=options.index(default))

try:
    data, checks = clean(raw, mapping)
except Exception as exc:  # wrong mapping -> explain instead of crashing
    st.error(f"Could not read the file with this column mapping: {exc}")
    st.stop()
if data.empty:
    st.error("No valid rows. Check the column mapping in the sidebar.")
    st.stop()

dates = sorted(data.date.unique())
as_of = pd.Timestamp(st.sidebar.selectbox("Report month", dates[::-1], format_func=lambda d: pd.Timestamp(d).strftime("%B %Y")))
view = st.sidebar.radio("View", ["Portfolio overview", "Project detail", "Map"])
summary = summarize(data, as_of)
st.sidebar.caption("Data quality")
for k, v in checks.items():
    st.sidebar.caption(f"{'✅' if v == 0 or k == 'Rows read' else '⚠️'} {k}: {v}")

# ------------------------------------------------------------ portfolio ----
if view == "Portfolio overview":
    st.title("Cable laying - portfolio overview")
    st.caption(f"Report of {as_of:%B %Y} - {len(summary)} projects" + (" - synthetic data" if is_sample else ""))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Planned", f"{summary.planned_km.sum():,.0f} km")
    c2.metric("Deployed", f"{summary.deployed_km.sum():,.0f} km", f"{summary.deployed_km.sum() / summary.planned_km.sum():.0%}")
    c3.metric("Remaining", f"{summary.remaining_km.sum():,.0f} km")
    c4.metric("Late / at risk", f"{summary.status.isin(['Late', 'At risk']).sum()} / {len(summary)}")

    pick = st.multiselect("Filter by status", S_DOMAIN, default=S_DOMAIN)
    view_df = summary[summary.status.isin(pick)].sort_values("delay_days", ascending=False)
    color = alt.Color("status:N", scale=alt.Scale(domain=S_DOMAIN, range=S_RANGE),
                      legend=alt.Legend(title="Status", orient="bottom"))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Progress by project")
        st.altair_chart(alt.Chart(view_df).mark_bar().encode(
            y=alt.Y("project:N", sort=alt.EncodingSortField("delay_days", order="descending"), title=None),
            x=alt.X("progress:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="Progress"),
            color=color,
            tooltip=["project", alt.Tooltip("deployed_km:Q", format=",.0f"),
                     alt.Tooltip("remaining_km:Q", format=",.0f"), "delay_days", "status"],
        ).properties(height=380), width="stretch")
    with right:
        st.subheader("Forecast delay (days)")
        st.altair_chart(alt.Chart(view_df.dropna(subset=["delay_days"])).mark_bar().encode(
            y=alt.Y("project:N", sort=alt.EncodingSortField("delay_days", order="descending"),
                    title=None, axis=alt.Axis(labels=False)),
            x=alt.X("delay_days:Q", title="Days vs planned end"), color=color.legend(None),
        ).properties(height=380), width="stretch")

    st.subheader("Project table")
    st.dataframe(
        view_df.assign(planned_end=view_df.planned_end.dt.date, forecast_end=view_df.forecast_end.dt.date)[
            ["project", "status", "planned_km", "deployed_km", "remaining_km", "progress",
             "planned_end", "forecast_end", "delay_days"]],
        hide_index=True, width="stretch",
        column_config={
            "progress": st.column_config.ProgressColumn("Progress", format="percent", min_value=0, max_value=1),
            "planned_km": st.column_config.NumberColumn("Planned (km)", format="%d"),
            "deployed_km": st.column_config.NumberColumn("Deployed (km)", format="%d"),
            "remaining_km": st.column_config.NumberColumn("Remaining (km)", format="%d"),
            "delay_days": st.column_config.NumberColumn("Delay (days)", format="%d"),
        })

# ----------------------------------------------------------------- map ----
elif view == "Map":
    st.title("Cable routes and estimated vessel positions")
    st.caption(f"Report of {as_of:%B %Y}")
    geo = summary.dropna(subset=["start_lat", "start_lon", "end_lat", "end_lon"])
    if geo.empty:
        st.info("No location data found. Map the start/end latitude and longitude columns in the sidebar "
                "('Column mapping'), or add them to the CSV.")
        st.stop()
    pick = st.multiselect("Filter by status", S_DOMAIN, default=S_DOMAIN, key="map_status")
    geo = geo[geo.status.isin(pick)]

    planned_rows, laid_rows, ship_rows, port_rows = [], [], [], []
    for r in geo.itertuples():
        label = f"{r.start_name or 'A'} → {r.end_name or 'B'}"
        info = {"project": r.project, "route": label, "status": r.status,
                "progress_txt": f"{r.progress:.0%} laid - {r.remaining_km:,.0f} km remaining"}
        color = hex_rgb(STATUS_COLORS[r.status])
        laid = gc_path(r.start_lat, r.start_lon, r.end_lat, r.end_lon, r.progress)
        planned_rows.append({**info, "path": gc_path(r.start_lat, r.start_lon, r.end_lat, r.end_lon)})
        laid_rows.append({**info, "path": laid, "color": color})
        ship_rows.append({**info, "lon": laid[-1][0], "lat": laid[-1][1], "color": color})
        port_rows += [{"port": r.start_name or "Start", "lat": r.start_lat, "lon": r.start_lon},
                      {"port": r.end_name or "End", "lat": r.end_lat, "lon": r.end_lon}]

    layers = [
        pdk.Layer("PathLayer", pd.DataFrame(planned_rows), get_path="path", get_color=[150, 160, 175, 140],
                  width_min_pixels=2, pickable=True),
        pdk.Layer("PathLayer", pd.DataFrame(laid_rows), get_path="path", get_color="color",
                  width_min_pixels=4, pickable=True),
        pdk.Layer("ScatterplotLayer", pd.DataFrame(port_rows).drop_duplicates(), get_position=["lon", "lat"],
                  get_fill_color=[35, 45, 65], radius_min_pixels=4, pickable=False),
        pdk.Layer("ScatterplotLayer", pd.DataFrame(ship_rows), get_position=["lon", "lat"], get_fill_color="color",
                  get_line_color=[255, 255, 255], stroked=True, line_width_min_pixels=2,
                  radius_min_pixels=8, pickable=True),
    ]
    view_state = pdk.ViewState(latitude=float(geo[["start_lat", "end_lat"]].mean().mean()),
                               longitude=float(geo[["start_lon", "end_lon"]].mean().mean()), zoom=1.1)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state,
                             tooltip={"text": "{project}\n{route}\n{progress_txt}\nStatus: {status}"}),
                    height=560)
    st.caption("Grey = planned route - colored = cable already laid - large dot = estimated vessel position. "
               "The position is interpolated along the route from the % laid, not a real GPS track.")
    st.dataframe(geo.assign(route=geo.start_name.fillna("A") + " → " + geo.end_name.fillna("B"))[
        ["project", "route", "status", "progress", "remaining_km"]], hide_index=True, width="stretch",
        column_config={"progress": st.column_config.ProgressColumn("Progress", format="percent", min_value=0, max_value=1),
                       "remaining_km": st.column_config.NumberColumn("Remaining (km)", format="%d")})

# -------------------------------------------------------------- detail ----
else:
    project = st.sidebar.selectbox("Project", summary.project)
    row = summary[summary.project == project].iloc[0]
    hist = data[(data.project == project) & (data.date <= as_of)].copy()
    hist["laid_in_month_km"] = hist.deployed_km.diff().fillna(hist.deployed_km)

    st.title(project)
    st.caption(f"Report of {as_of:%B %Y}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Planned", f"{row.planned_km:,.0f} km")
    c2.metric("Deployed", f"{row.deployed_km:,.0f} km", f"{row.progress:.0%}")
    c3.metric("Remaining", f"{row.remaining_km:,.0f} km")
    delay_txt = "n/a" if np.isnan(row.delay_days) else f"{row.delay_days:+.0f} days"
    c4.metric("Forecast vs planned end", delay_txt, row.status, delta_color="off")

    st.subheader("Cumulative deployed length")
    line = alt.Chart(hist).mark_line(point=True, strokeWidth=3, color="#1F6FEB").encode(
        x=alt.X("date:T", title=None), y=alt.Y("deployed_km:Q", title="km", scale=alt.Scale(domain=[0, row.planned_km * 1.05])),
        tooltip=["date:T", alt.Tooltip("deployed_km:Q", format=",.0f")])
    target = alt.Chart(pd.DataFrame({"y": [row.planned_km]})).mark_rule(strokeDash=[6, 4], color="#9AA5B1").encode(y="y:Q")
    layers = [target, line]
    if pd.notna(row.forecast_end) and row.remaining_km > 0.5:
        fc = pd.DataFrame({"date": [row.report_date, row.forecast_end], "km": [row.deployed_km, row.planned_km]})
        layers.append(alt.Chart(fc).mark_line(strokeDash=[4, 4], color="#E8A33D", strokeWidth=3).encode(x="date:T", y="km:Q"))
    if pd.notna(row.planned_end):
        layers.append(alt.Chart(pd.DataFrame({"d": [row.planned_end]})).mark_rule(color="#D64550").encode(x="d:T"))
    st.altair_chart(alt.layer(*layers).properties(height=340), width="stretch")
    st.caption("Grey dashed = planned length - orange dashed = forecast at current rate - red = planned end date")

    st.subheader("Cable laid per month (km)")
    st.altair_chart(alt.Chart(hist).mark_bar(color="#1F6FEB", opacity=0.8).encode(
        x=alt.X("yearmonth(date):T", title=None), y=alt.Y("laid_in_month_km:Q", title="km"),
        tooltip=[alt.Tooltip("yearmonth(date):T", title="Month"), alt.Tooltip("laid_in_month_km:Q", format=",.0f")],
    ).properties(height=200), width="stretch")

    if pd.notna(row.forecast_end):
        end_txt = f"**{row.planned_end:%d %b %Y}** planned" if pd.notna(row.planned_end) else "no planned end date in the file"
        st.info(f"At the recent pace (~{row.rate_km_day:,.1f} km/day), laying completes around "
                f"**{row.forecast_end:%d %b %Y}** ({end_txt}).")
    else:
        st.info("Not enough history to forecast: at least two monthly reports are needed.")
