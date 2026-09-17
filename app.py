"""
SMART ENERGY MONITORING & LOAD PREDICTION - Streamlit dashboard.

Run with:
    streamlit run app.py

Architecture note
-----------------
This file is a *rendering* layer only. It never trains a model and never
computes analytics inline:

    app.py  ->  src.models.predict     ->  models/best_model.pkl
            ->  src.anomaly.detector   ->  models/anomaly_model.pkl
            ->  src.utils.analytics    ->  data/processed/*.csv
            ->  src.database.db        ->  data/energy.db

Every data source is loaded defensively: if the dataset or a model is missing,
the page explains which command to run instead of crashing.
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard import components as ui
from src.anomaly.detector import AnomalyDetector, detector_is_available, summarise_anomalies
from src.data.preprocess import load_processed_data
from src.database import db
from src.iot.mqtt_client import get_mqtt_status
from src.iot.simulator import EnergySimulator, SimulatorSettings
from src.models import predict as predictor
from src.utils import analytics, config

st.set_page_config(
    page_title="Smart Energy Monitoring & Load Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "Dashboard Overview",
    "Historical Analysis",
    "Load Forecast",
    "Anomaly Detection",
    "Cost Estimation",
    "Real-Time Monitor",
    "System & Model Info",
]


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Processed dataset (cached for the session)."""
    return load_processed_data()


@st.cache_resource(show_spinner=False)
def load_detector() -> AnomalyDetector:
    """Trained Isolation Forest (cached across reruns)."""
    return AnomalyDetector.load()


@st.cache_data(show_spinner=False)
def labelled_dataset() -> pd.DataFrame:
    """Dataset with anomaly labels attached."""
    return load_detector().label_dataframe(load_dataset())


@st.cache_data(show_spinner=False)
def backtest(n_samples: int) -> pd.DataFrame:
    """Actual vs predicted over the most recent ``n_samples`` rows."""
    return predictor.predict_on_history(load_dataset().tail(n_samples))


def dataset_available() -> bool:
    try:
        load_dataset()
        return True
    except FileNotFoundError:
        return False


def missing_data_notice() -> None:
    st.error("Processed dataset not found.")
    st.code(
        "python -m src.data.generate_data\n"
        "python -m src.data.preprocess\n"
        "python -m src.models.train\n"
        "python -m src.anomaly.detector",
        language="bash",
    )
    st.stop()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar() -> tuple[str, float]:
    """Draw navigation, tariff input and the system-health panel."""
    st.sidebar.title("⚡ Smart Energy")
    st.sidebar.caption("Monitoring & Intelligent Load Prediction")

    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

    st.sidebar.markdown("---")
    tariff = st.sidebar.number_input(
        f"Electricity tariff ({config.CURRENCY_SYMBOL}/kWh)",
        min_value=0.0, max_value=100.0,
        value=float(st.session_state.get("tariff", config.DEFAULT_TARIFF_PER_KWH)),
        step=0.5,
        help="Used for all cost estimates. Estimated cost = energy (kWh) x tariff.",
    )
    st.session_state["tariff"] = tariff

    st.sidebar.markdown("---")
    st.sidebar.markdown("**System status**")

    data_ok = dataset_available()
    model_ok = predictor.model_is_available()
    detector_ok = detector_is_available()
    db_ok = db.database_is_available()
    mqtt = get_mqtt_status()

    ui.system_health_panel(
        [
            ("Dataset", data_ok, "Ready" if data_ok else "Missing"),
            ("ML model", model_ok, "Loaded" if model_ok else "Not trained"),
            ("Anomaly model", detector_ok, "Loaded" if detector_ok else "Not trained"),
            ("Database", db_ok, "Connected" if db_ok else "Not initialised"),
            ("MQTT", mqtt.connected, mqtt.label),
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data source: **synthetic development dataset**. "
        "No physical energy meter is connected."
    )
    return page, tariff


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_overview(df: pd.DataFrame, tariff: float) -> None:
    st.title("Dashboard Overview")
    st.caption("Latest recorded measurement and headline statistics")

    latest = df.iloc[-1]
    summary = analytics.consumption_summary(df)
    peaks = analytics.peak_analysis(df)

    ui.kpi_row(
        [
            {"label": "Voltage", "value": f"{latest['voltage']:.1f}", "unit": "V",
             "note": "RMS, nominal 230 V"},
            {"label": "Current", "value": f"{latest['current']:.2f}", "unit": "A",
             "note": "I = P / (V x PF)"},
            {"label": "Active power", "value": f"{latest['active_power']:.0f}", "unit": "W",
             "note": "P = V x I x PF", "color": ui.COLORS["accent"]},
            {"label": "Energy (this interval)", "value": f"{latest['energy']:.3f}",
             "unit": "kWh", "note": f"{config.SAMPLING_INTERVAL_MINUTES} min interval"},
            {"label": "Power factor", "value": f"{latest['power_factor']:.3f}", "unit": "",
             "note": "cos φ, dimensionless"},
            {"label": "Frequency", "value": f"{latest['frequency']:.2f}", "unit": "Hz",
             "note": "Nominal 50 Hz"},
        ]
    )

    st.markdown("#### Consumption at a glance")
    ui.kpi_row(
        [
            {"label": "Total energy", "value": f"{summary['total_energy_kwh']:,.1f}",
             "unit": "kWh", "note": f"over {summary['n_days']} days"},
            {"label": "Average daily", "value": f"{summary['avg_daily_kwh']:.2f}",
             "unit": "kWh/day"},
            {"label": "Peak load", "value": f"{peaks['max_load_w']:,.0f}", "unit": "W",
             "note": str(peaks["peak_timestamp"]), "color": ui.COLORS["warning"]},
            {"label": "Peak hour", "value": f"{peaks['peak_hour_of_day']:02d}:00",
             "unit": "", "note": "highest average load"},
            {"label": "Load factor",
             "value": f"{peaks['load_factor'] * 100:.1f}", "unit": "%",
             "note": "average / peak load"},
            {"label": "Estimated cost", "value": f"{analytics.estimate_cost(summary['avg_daily_kwh'] * 30, tariff):,.0f}",
             "unit": f"{config.CURRENCY_SYMBOL}/month",
             "note": f"at {config.CURRENCY_SYMBOL}{tariff:.2f}/kWh",
             "color": ui.COLORS["accent"]},
        ]
    )

    left, right = st.columns([3, 2])
    with left:
        ui.show_figure(ui.load_trend_chart(df.tail(24 * 14),
                                          "Active power - last 14 days"))
    with right:
        ui.show_figure(ui.hourly_profile_chart(analytics.hourly_consumption(df)))

    ui.note(
        "Active power is computed as P = V × I × PF and energy is integrated over the "
        f"{config.SAMPLING_INTERVAL_MINUTES}-minute sampling interval: E = P × Δt / 1000. "
        "All readings come from the synthetic development dataset."
    )


def page_historical(df: pd.DataFrame) -> None:
    st.title("Historical Analysis")
    st.caption("Load and energy behaviour across the full record")

    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
        view = df[mask]
    else:
        view = df

    if view.empty:
        st.warning("No samples in the selected range.")
        return

    summary = analytics.consumption_summary(view)
    peaks = analytics.peak_analysis(view)
    ui.kpi_row(
        [
            {"label": "Samples", "value": f"{len(view):,}", "unit": ""},
            {"label": "Energy", "value": f"{summary['total_energy_kwh']:,.1f}", "unit": "kWh"},
            {"label": "Max load", "value": f"{peaks['max_load_w']:,.0f}", "unit": "W",
             "color": ui.COLORS["warning"]},
            {"label": "Average load", "value": f"{peaks['avg_load_w']:,.0f}", "unit": "W"},
            {"label": "Min load", "value": f"{peaks['min_load_w']:,.0f}", "unit": "W"},
            {"label": "Daily trend", "value": f"{summary['trend_kwh_per_day']:+.3f}",
             "unit": "kWh/day²", "note": "linear fit on daily totals"},
        ]
    )

    tabs = st.tabs(["Load over time", "Energy consumption", "Usage profiles", "Peak analysis"])

    with tabs[0]:
        ui.show_figure(ui.load_trend_chart(view))
    with tabs[1]:
        period = st.radio("Aggregation", ["Daily", "Weekly", "Monthly"],
                          horizontal=True, key="hist_period")
        frame = {
            "Daily": analytics.daily_consumption,
            "Weekly": analytics.weekly_consumption,
            "Monthly": analytics.monthly_consumption,
        }[period](view)
        ui.show_figure(ui.energy_chart(frame, title=f"{period} energy consumption"))
        ui.show_dataframe(frame.tail(20), hide_index=True)
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            ui.show_figure(ui.hourly_profile_chart(analytics.hourly_consumption(view)))
        with col2:
            ui.show_figure(ui.weekday_profile_chart(view))
    with tabs[3]:
        ui.show_figure(ui.peak_chart(view, analytics.peak_periods(view)))
        col1, col2, col3 = st.columns(3)
        col1.metric("Daily peak (kWh)", f"{peaks['daily_peak']['energy_kwh']:.2f}",
                    help=str(peaks["daily_peak"]["timestamp"]))
        col2.metric("Weekly peak (kWh)", f"{peaks['weekly_peak']['energy_kwh']:.2f}",
                    help=str(peaks["weekly_peak"]["timestamp"]))
        col3.metric("Monthly peak (kWh)", f"{peaks['monthly_peak']['energy_kwh']:.2f}",
                    help=str(peaks["monthly_peak"]["timestamp"]))


def page_forecast(df: pd.DataFrame) -> None:
    st.title("Load Forecast")
    st.caption("Predictions produced by the saved model - no training happens here")

    if not predictor.model_is_available():
        st.error("No trained model found.")
        st.code("python -m src.models.train", language="bash")
        return

    meta = predictor.get_model_metadata()
    metrics = meta.get("metrics", {}).get(meta.get("best_model"), {})

    ui.kpi_row(
        [
            {"label": "Selected model", "value": meta.get("best_model", "?"), "unit": "",
             "note": meta.get("selection_rule", "")},
            {"label": "Test MAE", "value": f"{metrics.get('mae', float('nan')):.1f}",
             "unit": "W"},
            {"label": "Test RMSE", "value": f"{metrics.get('rmse', float('nan')):.1f}",
             "unit": "W"},
            {"label": "Test R²", "value": f"{metrics.get('r2', float('nan')):.4f}", "unit": ""},
        ]
    )

    horizon = st.slider("Forecast horizon (hours)", min_value=1, max_value=48, value=24)

    with st.spinner("Running recursive forecast..."):
        next_step = predictor.predict_next_hour(df)
        forecast = predictor.forecast(df, horizon=horizon)

    col1, col2, col3 = st.columns(3)
    col1.metric("Next-hour load", f"{next_step['predicted_power_w']:,.0f} W",
                help=str(next_step["timestamp"]))
    col2.metric(f"Energy over next {horizon} h",
                f"{forecast['predicted_energy_kwh'].sum():.2f} kWh")
    col3.metric("Forecast peak", f"{forecast['predicted_power_w'].max():,.0f} W",
                help=str(forecast.loc[forecast['predicted_power_w'].idxmax(), 'timestamp']))

    ui.show_figure(ui.forecast_chart(df.tail(72), forecast))

    st.markdown("#### Back-test: actual vs predicted")
    n_samples = st.select_slider("Samples to back-test", options=[168, 336, 720, 1440],
                                 value=720)
    history = backtest(int(n_samples))
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{history['abs_error'].mean():.1f} W")
    col2.metric("Max error", f"{history['abs_error'].max():.1f} W")
    col3.metric("Bias (mean error)", f"{history['error'].mean():+.1f} W")

    ui.show_figure(ui.actual_vs_predicted_chart(history))
    ui.show_figure(ui.error_distribution_chart(history))

    ui.note(
        "Multi-step forecasts are recursive: each prediction becomes the lag input for the "
        "next step, so error compounds with horizon. Treat long horizons as indicative "
        "rather than precise. The back-test above includes samples the model trained on "
        "when the selected window reaches back before "
        f"{meta.get('split_date', '')}; the headline metrics in the cards are from the "
        "held-out test period only."
    )


def page_anomaly(df: pd.DataFrame) -> None:
    st.title("Anomaly Detection")
    st.caption("Isolation Forest over active power, current, voltage and power factor")

    if not detector_is_available():
        st.error("No anomaly model found.")
        st.code("python -m src.anomaly.detector", language="bash")
        return

    labelled = labelled_dataset()
    stats = summarise_anomalies(labelled)

    ui.kpi_row(
        [
            {"label": "Samples scored", "value": f"{stats['total_samples']:,}", "unit": ""},
            {"label": "Abnormal patterns", "value": f"{stats['anomaly_count']:,}", "unit": "",
             "color": ui.COLORS["danger"]},
            {"label": "Anomaly rate", "value": f"{stats['anomaly_percentage']:.2f}",
             "unit": "%"},
            {"label": "Mean load - normal", "value": f"{stats['mean_power_normal']:,.0f}",
             "unit": "W"},
            {"label": "Mean load - abnormal", "value": f"{stats['mean_power_anomaly']:,.0f}",
             "unit": "W", "color": ui.COLORS["warning"]},
        ]
    )

    st.warning(
        "A flag means **abnormal consumption pattern** - a reading that does not resemble "
        "the rest of the data. It is not a confirmed equipment fault; proving that would "
        "require labelled failure data, which this project does not have."
    )

    tabs = st.tabs(["Timeline", "Operating points", "Event list"])
    with tabs[0]:
        window = st.select_slider("Samples to display", options=[336, 720, 1440, len(labelled)],
                                  value=720)
        ui.show_figure(ui.anomaly_timeline_chart(labelled.tail(int(window))))
    with tabs[1]:
        ui.show_figure(ui.anomaly_scatter_chart(labelled.sample(
            min(2000, len(labelled)), random_state=config.RANDOM_SEED)))
    with tabs[2]:
        events = labelled[labelled["is_anomaly"]].sort_values("anomaly_score")
        st.write(f"{len(events)} abnormal samples, most extreme first:")
        ui.show_dataframe(
            ui.round_numeric(
                events[["timestamp", "voltage", "current", "power_factor", "active_power",
                        "anomaly_score"]].head(50)
            ),
            hide_index=True,
        )


def page_cost(df: pd.DataFrame, tariff: float) -> None:
    st.title("Cost Estimation")
    st.caption("Estimated electricity cost = energy consumption × tariff")

    cost = analytics.cost_summary(df, tariff)
    symbol = config.CURRENCY_SYMBOL

    ui.kpi_row(
        [
            {"label": "Tariff", "value": f"{symbol}{cost['tariff_per_kwh']:.2f}",
             "unit": "/kWh", "note": "editable in the sidebar"},
            {"label": "Average daily use", "value": f"{cost['avg_daily_kwh']:.2f}",
             "unit": "kWh"},
            {"label": "Estimated daily cost", "value": f"{symbol}{cost['estimated_daily_cost']:,.2f}",
             "unit": "", "color": ui.COLORS["accent"]},
            {"label": "Estimated monthly cost",
             "value": f"{symbol}{cost['estimated_monthly_cost']:,.0f}", "unit": "",
             "note": "30 days", "color": ui.COLORS["accent"]},
            {"label": "Estimated yearly cost",
             "value": f"{symbol}{cost['estimated_yearly_cost']:,.0f}", "unit": "",
             "note": "365 days"},
            {"label": "Recorded period cost", "value": f"{symbol}{cost['total_cost']:,.0f}",
             "unit": "", "note": f"{cost['n_days']} days of data"},
        ]
    )

    st.error(
        "**Estimated electricity cost only.** A real utility bill also includes fixed "
        "charges, slab/telescopic tariff rates, taxes, duties and possibly demand charges. "
        "This figure is a simple energy × tariff product."
    )

    daily = analytics.daily_cost_table(df, tariff)
    ui.show_figure(ui.cost_chart(daily, symbol))

    st.markdown("#### Tariff sensitivity")
    rates = sorted({round(tariff * f, 2) for f in (0.5, 0.75, 1.0, 1.25, 1.5)})
    table = pd.DataFrame(
        {
            f"Tariff ({symbol}/kWh)": rates,
            "Estimated monthly cost": [
                analytics.estimate_cost(cost["avg_daily_kwh"] * 30, r) for r in rates
            ],
            "Estimated yearly cost": [
                analytics.estimate_cost(cost["avg_daily_kwh"] * 365, r) for r in rates
            ],
        }
    ).round(2)
    ui.show_dataframe(table, hide_index=True)


def page_realtime(tariff: float) -> None:
    st.title("Real-Time Monitor")

    mqtt = get_mqtt_status()
    mode = "MQTT" if mqtt.connected else "SIMULATION"
    ui.mode_banner(mode)

    if not mqtt.connected:
        st.caption(
            f"MQTT status: {mqtt.label}. Configure MQTT_BROKER in .env to ingest readings "
            "from a real ESP32/STM32 node; until then the simulator drives this page."
        )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        interval = st.number_input("Update interval (s)", 0.5, 10.0,
                                   float(config.SIMULATOR_INTERVAL_SECONDS), 0.5)
    with col2:
        store = st.checkbox("Store to SQLite", value=False,
                            help="Writes each generated reading into data/energy.db")
    with col3:
        auto = st.checkbox("Auto-refresh", value=False,
                           help="Continuously generates readings until unchecked")

    if "sim" not in st.session_state:
        st.session_state["sim"] = EnergySimulator(
            settings=SimulatorSettings(interval_seconds=interval)
        )
        st.session_state["readings"] = []

    simulator: EnergySimulator = st.session_state["sim"]
    simulator.settings.interval_seconds = interval

    button_col1, button_col2 = st.columns([1, 6])
    step = button_col1.button("Generate reading", type="primary")
    if button_col2.button("Clear buffer"):
        st.session_state["readings"] = []

    detector = load_detector() if detector_is_available() else None

    if step or auto:
        reading = simulator.generate_reading()
        if detector is not None:
            reading.update(detector.check_reading(reading))
        st.session_state["readings"].append(reading)
        st.session_state["readings"] = st.session_state["readings"][-200:]
        if store:
            db.initialize_database()
            db.insert_reading(reading)

    readings = st.session_state["readings"]
    if not readings:
        st.info("Press **Generate reading** or enable auto-refresh to start the stream.")
        return

    latest = readings[-1]
    status = latest.get("anomaly_status", "NOT SCORED")
    ui.kpi_row(
        [
            {"label": "Voltage", "value": f"{latest['voltage']:.1f}", "unit": "V"},
            {"label": "Current", "value": f"{latest['current']:.2f}", "unit": "A"},
            {"label": "Active power", "value": f"{latest['active_power']:,.0f}", "unit": "W",
             "color": ui.COLORS["accent"]},
            {"label": "Power factor", "value": f"{latest['power_factor']:.3f}", "unit": ""},
            {"label": "Frequency", "value": f"{latest['frequency']:.2f}", "unit": "Hz"},
            {"label": "Status", "value": status, "unit": "",
             "color": ui.COLORS["danger"] if status == "ANOMALY" else ui.COLORS["accent"],
             "note": "Isolation Forest" if detector else "anomaly model not trained"},
        ]
    )

    frame = pd.DataFrame(readings)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    ui.show_figure(ui.realtime_chart(frame))

    total_energy = float(frame["energy"].sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Readings in buffer", len(frame))
    col2.metric("Energy accumulated", f"{total_energy:.3f} kWh")
    col3.metric("Estimated cost",
                f"{config.CURRENCY_SYMBOL}{analytics.estimate_cost(total_energy, tariff):,.2f}")
    col4.metric("Abnormal in buffer",
                int(frame["is_anomaly"].sum()) if "is_anomaly" in frame else 0)

    ui.show_dataframe(
        ui.round_numeric(
            frame.tail(10)[["timestamp", "device_id", "voltage", "current", "power_factor",
                            "frequency", "active_power", "energy"]]
        ),
        hide_index=True,
    )

    if store:
        st.caption(f"Stored rows in database: {db.count_readings():,}")

    if auto:
        time.sleep(float(interval))
        st.rerun()


def page_system(df: pd.DataFrame) -> None:
    st.title("System & Model Info")

    if predictor.model_is_available():
        meta = predictor.get_model_metadata()
        st.markdown("#### Model")
        col1, col2, col3 = st.columns(3)
        col1.metric("Selected model", meta.get("best_model", "?"))
        col2.metric("Features", meta.get("n_features", 0))
        col3.metric("Trained at", str(meta.get("trained_at", ""))[:16])

        st.caption(
            f"Split strategy: {meta.get('split_strategy')} | "
            f"train {meta.get('train_period', ['', ''])[0][:10]} to "
            f"{meta.get('train_period', ['', ''])[1][:10]} | "
            f"test {meta.get('test_period', ['', ''])[0][:10]} to "
            f"{meta.get('test_period', ['', ''])[1][:10]}"
        )

        comparison = pd.DataFrame(meta.get("metrics", {})).T.reset_index(names="model")
        ui.show_dataframe(comparison.round(4), hide_index=True)
        ui.show_figure(ui.model_comparison_chart(comparison))

        with st.expander("Feature list"):
            st.write(meta.get("feature_names", []))
    else:
        st.warning("Model not trained yet - run `python -m src.models.train`.")

    st.markdown("#### Database")
    if db.database_is_available():
        summary = db.get_energy_summary()
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows stored", f"{summary['n_readings']:,}")
        col2.metric("Energy stored", f"{summary['total_energy_kwh']:.2f} kWh")
        col3.metric("Flagged abnormal", summary["anomaly_count"])
        recent = db.get_recent_readings(10)
        if not recent.empty:
            ui.show_dataframe(
                recent[["timestamp", "device_id", "voltage", "current", "active_power",
                        "anomaly_status"]],
                hide_index=True,
            )
    else:
        st.info("Database not initialised yet. It is created automatically on first write.")

    st.markdown("#### Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", f"{len(df):,}")
    col2.metric("From", str(df['timestamp'].min())[:16])
    col3.metric("To", str(df['timestamp'].max())[:16])

    st.markdown("#### MQTT")
    mqtt = get_mqtt_status()
    st.write(
        {
            "library_available": mqtt.library_available,
            "broker": mqtt.broker or "(not configured)",
            "port": mqtt.port,
            "topic": mqtt.topic,
            "status": mqtt.label,
        }
    )
    ui.note(
        "The dashboard is fully functional without MQTT. When a broker is configured, "
        "readings published by an ESP32/STM32 node on the topic above are validated, "
        "converted and stored through the same pipeline as simulated readings."
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    ui.inject_css()
    page, tariff = render_sidebar()

    if page == "Real-Time Monitor":
        page_realtime(tariff)
        return

    if not dataset_available():
        missing_data_notice()

    df = load_dataset()

    if page == "Dashboard Overview":
        page_overview(df, tariff)
    elif page == "Historical Analysis":
        page_historical(df)
    elif page == "Load Forecast":
        page_forecast(df)
    elif page == "Anomaly Detection":
        page_anomaly(df)
    elif page == "Cost Estimation":
        page_cost(df, tariff)
    elif page == "System & Model Info":
        page_system(df)

    st.markdown("---")
    st.caption(
        f"Smart Energy Monitoring & Load Prediction · synthetic development dataset · "
        f"rendered {datetime.now():%Y-%m-%d %H:%M}"
    )


if __name__ == "__main__":
    main()
