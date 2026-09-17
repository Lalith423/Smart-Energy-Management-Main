"""
Reusable dashboard components: KPI cards, status badges and Plotly figures.

The Streamlit page file (``app.py``) only arranges these; all figure
construction lives here so the UI code stays readable and the styling is
consistent across pages.

Version note: Streamlit renamed the chart-width argument in 1.49. The
``show_figure`` / ``show_dataframe`` helpers below pick the right keyword at
runtime, so the dashboard works on both older and newer releases without
emitting deprecation warnings.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
COLORS = {
    "primary": "#1f6feb",
    "accent": "#00b894",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "muted": "#6b7280",
    "grid": "#e5e7eb",
    "actual": "#1f6feb",
    "predicted": "#f39c12",
    "anomaly": "#e74c3c",
}

PLOT_TEMPLATE = "plotly_white"


def _streamlit_version() -> tuple[int, ...]:
    parts = []
    for chunk in str(getattr(st, "__version__", "0.0")).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts[:2])


_USE_WIDTH_KEYWORD = _streamlit_version() >= (1, 49)


def show_figure(fig: go.Figure) -> None:
    """Render a Plotly figure full-width on any supported Streamlit version."""
    if _USE_WIDTH_KEYWORD:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_dataframe(df: pd.DataFrame, **kwargs) -> None:
    """Render a dataframe full-width on any supported Streamlit version."""
    if _USE_WIDTH_KEYWORD:
        st.dataframe(df, width="stretch", **kwargs)
    else:
        st.dataframe(df, use_container_width=True, **kwargs)


def round_numeric(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Round only the numeric columns, leaving timestamps untouched."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def inject_css() -> None:
    """Global styling for KPI cards, badges and banners."""
    st.markdown(
        """
        <style>
        .kpi-grid { display:flex; flex-wrap:wrap; gap:14px; margin:6px 0 18px 0; }
        .kpi-card {
            flex:1 1 170px; background:#ffffff; border:1px solid #e5e7eb;
            border-left:4px solid #1f6feb; border-radius:10px; padding:14px 16px;
            box-shadow:0 1px 3px rgba(16,24,40,.06);
        }
        .kpi-label { font-size:.74rem; letter-spacing:.06em; text-transform:uppercase;
                     color:#6b7280; margin-bottom:6px; font-weight:600; }
        .kpi-value { font-size:1.6rem; font-weight:700; color:#111827; line-height:1.1; }
        .kpi-unit  { font-size:.85rem; font-weight:500; color:#6b7280; margin-left:4px; }
        .kpi-note  { font-size:.72rem; color:#9ca3af; margin-top:6px; }
        .status-row { display:flex; justify-content:space-between; padding:7px 2px;
                      border-bottom:1px dashed #e5e7eb; font-size:.88rem; }
        .dot { height:9px; width:9px; border-radius:50%; display:inline-block;
               margin-right:7px; }
        .mode-banner {
            border-radius:8px; padding:10px 14px; font-weight:600; font-size:.9rem;
            margin-bottom:12px; border:1px solid transparent;
        }
        .mode-sim  { background:#fff7ed; border-color:#fed7aa; color:#9a3412; }
        .mode-mqtt { background:#ecfdf5; border-color:#a7f3d0; color:#065f46; }
        .note-box  { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                     padding:10px 14px; font-size:.84rem; color:#475569; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# KPI cards & status
# --------------------------------------------------------------------------
def kpi_card(label: str, value: str, unit: str = "", note: str = "",
             color: str = COLORS["primary"]) -> str:
    """Return the HTML for a single KPI card."""
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    return (
        f'<div class="kpi-card" style="border-left-color:{color}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>'
        f"{note_html}"
        f"</div>"
    )


def kpi_row(cards: list[dict]) -> None:
    """Render a responsive row of KPI cards.

    Each dict accepts: label, value, unit, note, color.
    """
    html = "".join(
        kpi_card(
            c["label"], c["value"], c.get("unit", ""), c.get("note", ""),
            c.get("color", COLORS["primary"]),
        )
        for c in cards
    )
    st.markdown(f'<div class="kpi-grid">{html}</div>', unsafe_allow_html=True)


def status_row(label: str, ok: bool, text: str) -> str:
    """HTML for one line of the system-health panel."""
    color = COLORS["accent"] if ok else COLORS["muted"]
    return (
        f'<div class="status-row"><span>{label}</span>'
        f'<span><span class="dot" style="background:{color}"></span>{text}</span></div>'
    )


def system_health_panel(statuses: list[tuple[str, bool, str]]) -> None:
    """Render the Data / Model / Database / MQTT status list."""
    st.markdown("".join(status_row(*s) for s in statuses), unsafe_allow_html=True)


def mode_banner(mode: str) -> None:
    """Prominent SIMULATION / MQTT banner so synthetic data is never mistaken for real."""
    if mode.upper().startswith("MQTT"):
        st.markdown(
            '<div class="mode-banner mode-mqtt">MQTT MODE &mdash; readings received '
            "from a live broker</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="mode-banner mode-sim">SIMULATION MODE &mdash; all values are '
            "synthetic, generated by the software. No sensor is connected.</div>",
            unsafe_allow_html=True,
        )


def note(text: str) -> None:
    """Small muted explanatory box."""
    st.markdown(f'<div class="note-box">{text}</div>', unsafe_allow_html=True)


def _style(fig: go.Figure, ylabel: str = "", xlabel: str = "") -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        margin=dict(l=10, r=10, t=44, b=10),
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if ylabel:
        fig.update_yaxes(title_text=ylabel)
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    return fig


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def load_trend_chart(df: pd.DataFrame, title: str = "Active power over time") -> go.Figure:
    """Line chart of active power against time."""
    fig = px.line(df, x="timestamp", y="active_power", title=title,
                  color_discrete_sequence=[COLORS["primary"]])
    fig.update_traces(line_width=1.4, hovertemplate="%{y:.0f} W<extra></extra>")
    return _style(fig, "Active power (W)", "Time")


def energy_chart(df: pd.DataFrame, x: str = "timestamp", y: str = "energy_kwh",
                 title: str = "Energy consumption") -> go.Figure:
    """Bar chart of energy per period."""
    fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=[COLORS["accent"]])
    fig.update_traces(hovertemplate="%{y:.2f} kWh<extra></extra>")
    return _style(fig, "Energy (kWh)", "Period")


def hourly_profile_chart(hourly: pd.DataFrame) -> go.Figure:
    """Average daily load profile, with the peak hour highlighted."""
    peak_hour = int(hourly.loc[hourly["avg_power_w"].idxmax(), "hour"])
    colors = [COLORS["danger"] if h == peak_hour else COLORS["primary"]
              for h in hourly["hour"]]
    fig = go.Figure(
        go.Bar(x=hourly["hour"], y=hourly["avg_power_w"], marker_color=colors,
               hovertemplate="Hour %{x}:00 &mdash; %{y:.0f} W<extra></extra>")
    )
    fig.update_layout(title=f"Average load profile by hour (peak: {peak_hour}:00)")
    fig.update_xaxes(dtick=2)
    return _style(fig, "Average active power (W)", "Hour of day")


def weekday_profile_chart(df: pd.DataFrame) -> go.Figure:
    """Average load by day of week."""
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    tmp = df.copy()
    tmp["dow"] = pd.to_datetime(tmp["timestamp"]).dt.dayofweek
    agg = tmp.groupby("dow")["active_power"].mean().reset_index()
    agg["day"] = agg["dow"].map(lambda d: names[int(d)])
    fig = px.bar(agg, x="day", y="active_power", title="Average load by day of week",
                 color_discrete_sequence=[COLORS["primary"]])
    fig.update_traces(hovertemplate="%{y:.0f} W<extra></extra>")
    return _style(fig, "Average active power (W)", "Day")


def peak_chart(df: pd.DataFrame, peaks: pd.DataFrame) -> go.Figure:
    """Load curve with peak-period samples marked."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["active_power"], mode="lines",
                             name="Load", line=dict(color=COLORS["primary"], width=1.2)))
    if not peaks.empty:
        fig.add_trace(go.Scatter(x=peaks["timestamp"], y=peaks["active_power"],
                                 mode="markers", name="Peak period",
                                 marker=dict(color=COLORS["danger"], size=5)))
        fig.add_hline(y=float(peaks["threshold_w"].iloc[0]), line_dash="dash",
                      line_color=COLORS["danger"],
                      annotation_text="Peak threshold (90th percentile)")
    fig.update_layout(title="Peak periods")
    return _style(fig, "Active power (W)", "Time")


def actual_vs_predicted_chart(history: pd.DataFrame) -> go.Figure:
    """Overlay of measured and predicted load."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["timestamp"], y=history["actual"], name="Actual",
                             mode="lines", line=dict(color=COLORS["actual"], width=1.5)))
    fig.add_trace(go.Scatter(x=history["timestamp"], y=history["predicted"], name="Predicted",
                             mode="lines",
                             line=dict(color=COLORS["predicted"], width=1.5, dash="dot")))
    fig.update_layout(title="Actual vs predicted active power")
    return _style(fig, "Active power (W)", "Time")


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Recent measured load followed by the recursive forecast."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["timestamp"], y=history["active_power"],
                             name="Measured (recent)", mode="lines",
                             line=dict(color=COLORS["actual"], width=1.5)))
    fig.add_trace(go.Scatter(x=forecast["timestamp"], y=forecast["predicted_power_w"],
                             name="Forecast", mode="lines+markers",
                             line=dict(color=COLORS["predicted"], width=2, dash="dash"),
                             marker=dict(size=5)))
    if not history.empty:
        fig.add_vline(x=history["timestamp"].iloc[-1], line_dash="dot",
                      line_color=COLORS["muted"], annotation_text="now")
    fig.update_layout(title="Load forecast")
    return _style(fig, "Active power (W)", "Time")


def error_distribution_chart(history: pd.DataFrame) -> go.Figure:
    """Histogram of prediction error (predicted - actual)."""
    fig = px.histogram(history, x="error", nbins=50, title="Prediction error distribution",
                       color_discrete_sequence=[COLORS["primary"]])
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["danger"])
    return _style(fig, "Samples", "Error (W)")


def anomaly_timeline_chart(labelled: pd.DataFrame) -> go.Figure:
    """Load curve with flagged abnormal samples marked in red."""
    normal = labelled[~labelled["is_anomaly"]]
    abnormal = labelled[labelled["is_anomaly"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal["timestamp"], y=normal["active_power"], mode="lines",
                             name="Normal", line=dict(color=COLORS["primary"], width=1.1)))
    fig.add_trace(go.Scatter(x=abnormal["timestamp"], y=abnormal["active_power"],
                             mode="markers", name="Abnormal pattern",
                             marker=dict(color=COLORS["anomaly"], size=7, symbol="x")))
    fig.update_layout(title="Anomaly timeline")
    return _style(fig, "Active power (W)", "Time")


def anomaly_scatter_chart(labelled: pd.DataFrame) -> go.Figure:
    """Current vs voltage, coloured by anomaly status."""
    fig = px.scatter(
        labelled, x="voltage", y="current", color="anomaly_status",
        title="Operating points (current vs voltage)",
        color_discrete_map={"NORMAL": COLORS["primary"], "ANOMALY": COLORS["anomaly"]},
        opacity=0.65,
    )
    return _style(fig, "Current (A)", "Voltage (V)")


def cost_chart(daily_costs: pd.DataFrame, currency: str = "Rs.") -> go.Figure:
    """Estimated daily electricity cost."""
    fig = px.bar(daily_costs, x="timestamp", y="estimated_cost",
                 title=f"Estimated daily cost ({currency})",
                 color_discrete_sequence=[COLORS["accent"]])
    fig.update_traces(hovertemplate=f"{currency} %{{y:.2f}}<extra></extra>")
    return _style(fig, f"Estimated cost ({currency})", "Date")


def realtime_chart(readings: pd.DataFrame) -> go.Figure:
    """Live power trace for the real-time monitor page."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=readings["timestamp"], y=readings["active_power"],
                             mode="lines+markers", name="Active power",
                             line=dict(color=COLORS["primary"], width=2),
                             marker=dict(size=5)))
    if "is_anomaly" in readings.columns:
        flagged = readings[readings["is_anomaly"]]
        if not flagged.empty:
            fig.add_trace(go.Scatter(x=flagged["timestamp"], y=flagged["active_power"],
                                     mode="markers", name="Abnormal",
                                     marker=dict(color=COLORS["anomaly"], size=11,
                                                 symbol="x")))
    fig.update_layout(title="Live active power")
    return _style(fig, "Active power (W)", "Time")


def model_comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of MAE and RMSE per model."""
    tidy = comparison.melt(id_vars="model", value_vars=["mae", "rmse"],
                           var_name="metric", value_name="watts")
    tidy["metric"] = tidy["metric"].str.upper()
    fig = px.bar(tidy, x="model", y="watts", color="metric", barmode="group",
                 title="Model comparison (lower is better)",
                 color_discrete_sequence=[COLORS["primary"], COLORS["warning"]])
    return _style(fig, "Error (W)", "Model")
