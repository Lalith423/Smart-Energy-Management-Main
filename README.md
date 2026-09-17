# Smart Energy Monitoring & Intelligent Load Prediction System

An end-to-end energy-monitoring platform that ingests electrical measurements, stores them,
forecasts future load with machine learning, flags abnormal consumption patterns, and presents
everything through an interactive engineering dashboard.

Built as a 3rd-year ECE project, engineered to production-repository standards: modular
packages, 124 automated tests, continuous integration, and no hardware required to run it.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-124%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Project Overview

The system models the full path an energy measurement takes, from sensing node to decision:

```
ESP32 / STM32 node  →  isolated V & I front-end  →  Wi-Fi  →  MQTT broker
                                                                   ↓
                                                          Python backend
                                                                   ↓
                              validation → preprocessing → feature engineering
                                                                   ↓
                                        ┌──────────────┬───────────────────┐
                                   ML forecasting   anomaly detection   SQLite
                                        └──────────────┴───────────────────┘
                                                                   ↓
                                                      Streamlit dashboard
```

Everything downstream of the broker is implemented and runs today. The hardware node is
**designed for but not built** — the MQTT contract is defined, validated and tested, so a
real node can be attached without changing the backend.

---

## Problem Statement

Most households and small facilities see their electricity consumption once a month, as a
single number on a bill. That is far too coarse to act on. Without hourly visibility you
cannot answer basic questions:

- When does my load actually peak, and could that load be shifted to a cheaper period?
- How much will I consume tomorrow, and what will it cost?
- Is today's consumption behaving abnormally compared with my own history?

Utilities solve this with SCADA and smart meters. This project builds the same capability at
student scale: monitor the electrical quantities, learn the consumption pattern, forecast
ahead, and surface anything unusual.

---

## Objectives

1. Monitor voltage, current, power factor and frequency.
2. Compute active power and integrate energy over the sampling interval.
3. Store readings in a persistent local database.
4. Validate incoming data and report — never silently discard — every problem found.
5. Engineer leakage-free time-series features.
6. Train and compare several regression models for load forecasting.
7. Select a model using a criterion defined in advance.
8. Detect abnormal consumption patterns without labelled fault data.
9. Estimate electricity cost from a user-supplied tariff.
10. Present everything through a dashboard that works with or without MQTT.

---

## Key Features

| Area | What it does |
|---|---|
| **Data pipeline** | Synthetic dataset generator, validation layer with a structured report, deterministic cleaning |
| **Feature engineering** | Calendar + cyclical encodings, lag features, shift-safe rolling statistics |
| **Forecasting** | Linear Regression, Random Forest, Gradient Boosting (optional XGBoost); recursive multi-step forecast |
| **Anomaly detection** | Isolation Forest over P, I, V and PF, with NORMAL / ANOMALY labelling |
| **Peak analysis** | Max/avg/min load, peak hour, daily/weekly/monthly peaks, load factor |
| **Cost estimation** | Adjustable tariff, daily/monthly/yearly estimates, sensitivity table |
| **Real-time monitor** | Deterministic sensor simulator with a clear SIMULATION MODE banner |
| **IoT layer** | Optional MQTT ingestion with JSON validation; the app runs fine with the broker down |
| **Persistence** | SQLite with a clean function-level API and indexed queries |
| **Quality** | 124 pytest tests, GitHub Actions on Python 3.10 / 3.11 / 3.12 |

---

## System Architecture

```
                         ┌───────────────────────────────┐
                         │      Streamlit dashboard      │
                         │           (app.py)            │
                         └───────────────┬───────────────┘
                                         │  renders only
       ┌──────────────┬──────────────────┼──────────────────┬──────────────┐
       ▼              ▼                  ▼                  ▼              ▼
 dashboard/     src/models/        src/anomaly/       src/utils/     src/database/
 components.py  predict.py         detector.py        analytics.py   db.py
       │              │                  │                  │              │
       │              ▼                  ▼                  ▼              ▼
       │     models/best_model.pkl  models/            data/processed/  data/energy.db
       │                            anomaly_model.pkl  energy_processed.csv
       │                                  ▲                  ▲              ▲
       │                                  │                  │              │
       │                          src/models/train.py        │       src/iot/
       │                                  ▲                  │       simulator.py
       │                                  │                  │       mqtt_client.py
       │                       src/features/feature_engineering.py
       │                                  ▲
       │                          src/data/preprocess.py
       │                                  ▲
       │                          src/data/generate_data.py
       └──────────────────────────────────┘
```

Two architectural rules are enforced throughout:

1. **The dashboard never trains.** It calls `src.models.predict`, which loads the saved
   artefact. Training is an offline step with its own command.
2. **Business logic never lives in the UI.** Analytics, database access and model calls are
   importable, testable modules; `app.py` only arranges their output.

---

## ECE Concepts Used

| Concept | Where it appears |
|---|---|
| **Active power** `P = V · I · PF` | `generate_data.py`, `preprocess.py`, `db.py`, `mqtt_client.py` |
| **Apparent vs active power** | Power factor is modelled explicitly; current is derived as `I = P / (V · PF)` |
| **Power factor behaviour** | PF improves with load: light load is dominated by inductive standby devices (PF ≈ 0.80), heavy load has a larger resistive share (PF → 0.97) |
| **Energy integration** | `E = P · Δt / 1000` kWh, where Δt is the sampling interval (60 min) |
| **Voltage regulation** | Supply voltage sags as feeder current rises — modelled as a load-dependent drop about the 230 V nominal |
| **Grid frequency** | 50 Hz nominal with a small excursion band (49.5–50.5 Hz) |
| **Load curve** | Night trough → morning shoulder → afternoon dip → evening peak, the classic domestic profile |
| **Load factor** | Average load ÷ peak load, reported on the dashboard |
| **Peak demand** | Peak hour, daily/weekly/monthly peaks, 90th-percentile peak periods |
| **Electrical safety** | Mains is never connected to an MCU directly — see [ESP32/STM32 Integration](#esp32stm32-integration) |
| **IoT / embedded comms** | MQTT publish–subscribe, JSON payload contract, constrained-device design (the node may omit derived fields) |

---

## Machine Learning Pipeline

```
processed CSV
     ↓
create_features()          calendar + cyclical + lag + rolling  (17 features)
     ↓
chronological_split()      Jan–Aug → train   |   Sep → test     (never shuffled)
     ↓
train 3–4 models           LinearRegression, RandomForest, GradientBoosting [, XGBoost]
     ↓
evaluate                   MAE, RMSE, R², MAPE on the held-out month
     ↓
select_best_model()        lowest test RMSE
     ↓
joblib.dump()              models/best_model.pkl + model_metadata.json + model_comparison.csv
     ↓
predict.py                 next-hour, recursive N-hour forecast, daily energy
```

**Target:** `active_power` in watts.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, NumPy |
| ML | scikit-learn (Linear Regression, Random Forest, Gradient Boosting, Isolation Forest); XGBoost optional |
| Persistence | SQLite (standard library `sqlite3`) |
| Model artefacts | joblib |
| IoT | paho-mqtt (optional), JSON |
| Dashboard | Streamlit, Plotly |
| Config | python-dotenv |
| Testing | pytest |
| CI | GitHub Actions |
| Notebooks | Jupyter, matplotlib, seaborn |

---

## Project Structure

```
Smart-Energy-Monitoring/
├── .github/workflows/python-tests.yml   CI: install, build pipeline, run tests
├── data/
│   ├── raw/          generated dataset (gitignored)
│   ├── processed/    cleaned dataset (gitignored)
│   └── sample/       500-row sample, committed
├── models/           best_model.pkl, anomaly_model.pkl, metadata (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/         generate_data.py, preprocess.py
│   ├── features/     feature_engineering.py
│   ├── models/       train.py, predict.py, evaluate.py
│   ├── anomaly/      detector.py
│   ├── database/     db.py
│   ├── iot/          simulator.py, mqtt_client.py
│   └── utils/        config.py, analytics.py
├── dashboard/        components.py   (KPI cards, status badges, Plotly figures)
├── tests/            test_data, test_features, test_model, test_anomaly,
│                     test_database, test_iot, test_analytics, conftest
├── screenshots/
├── app.py            Streamlit entry point
├── requirements.txt
├── pyproject.toml
├── .env.example
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Installation

Requires Python 3.10 or newer and Git.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Smart-Energy-Monitoring

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -e ".[notebooks]"   # Jupyter, matplotlib, seaborn
pip install -e ".[boosting]"    # XGBoost as a 4th candidate model
```

---

## Environment Setup

```bash
cp .env.example .env            # Windows: copy .env.example .env
```

Every setting has a working default in `src/utils/config.py`, so the project runs even with
no `.env` present.

```ini
MQTT_BROKER=                    # empty → the app runs in SIMULATION MODE
MQTT_PORT=1883
MQTT_TOPIC=energy/data
DEVICE_ID=ENERGY_NODE_01
DATABASE_PATH=data/energy.db
MODEL_PATH=models/best_model.pkl
TARIFF_PER_KWH=8.0
```

`.env` is listed in `.gitignore` and must never be committed. No credentials appear anywhere
in source.

---

## Dataset

> **Synthetic development dataset.** Generated by `src/data/generate_data.py` from a
> mathematical load model. It is **not** measured with a physical energy meter. It exists so
> the pipeline, models and dashboard can be developed and tested without hardware.

| Property | Value |
|---|---|
| Samples | 6,552 |
| Period | 2025-01-01 00:00 → 2025-09-30 23:00 |
| Sampling interval | 60 minutes |
| Columns | `timestamp, voltage, current, power_factor, frequency, active_power, energy` |
| Average daily consumption | 19.31 kWh |
| Peak load | 4,632 W |
| Reproducible | Yes — fixed seed |

**How it is generated.** Active power is built from multiplicative components, then the
electrical quantities a meter would report are derived from it:

```
P(t) = base_load × hour_factor(t) × day_factor(t) × season_factor(t) × noise
I    = P / (V × PF)
E    = P × Δt / 1000
```

- `hour_factor` encodes the domestic load curve (night trough ≈ 0.32, evening peak ≈ 1.60).
- `day_factor` raises weekend consumption by 12 %.
- `season_factor` adds yearly seasonality peaking in the hot months (cooling load).
- Noise is log-normal, keeping power strictly positive.
- ~40 abnormal episodes (surges and collapses) are injected **without labels**, so the
  Isolation Forest has genuinely unusual patterns to find.

Values are clipped to physically plausible bands, so no impossible readings are produced.

---

## Data Processing

`python -m src.data.preprocess`

The validator checks required columns, numeric dtypes, timestamp parsing, duplicate
timestamps, missing values, out-of-range values, and consistency with P = V·I·PF. It returns
a structured `ValidationReport` rather than a boolean.

Cleaning, in order:

1. Drop rows with unparseable timestamps
2. Sort chronologically
3. Drop duplicate timestamps (keep first)
4. Interpolate short gaps, then forward/back fill the edges
5. Clip physically implausible values into the valid band
6. Recompute `active_power` and `energy` so the physics holds

**Every action is recorded and printed.** Nothing is removed silently — that is a hard rule
of this project, and there is a test enforcing it.

---

## Model Training

```bash
python -m src.models.train
```

### Chronological splitting

Time-series data is **never shuffled**. A random split lets the model train on samples that
occur after the test samples, so it effectively sees the future and the reported score is
optimistic and meaningless.

```
2025-01-02 → 2025-08-31    5,808 samples   training
2025-09-01 → 2025-09-30      720 samples   testing (held out)
```

Notebook 03 quantifies the difference: a random split reports a higher R² than the honest
chronological split on identical data.

### Features (17)

| Group | Features |
|---|---|
| Calendar | `hour`, `day`, `month`, `day_of_week`, `is_weekend` |
| Cyclical | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| Lags | `lag_1`, `lag_2`, `lag_3`, `lag_24` |
| Rolling (shifted) | `rolling_mean_3`, `rolling_std_3`, `rolling_mean_24`, `rolling_std_24` |

**Why cyclical encoding?** Two reasons. Hour 23 and hour 0 are one hour apart in reality but
23 units apart as integers. And a tree can only split on values it saw in training — the test
month (September = 9) lies outside the training months (1–8), so a raw `month` integer forces
extrapolation, which tree ensembles cannot do. The sin/cos values for September fall inside
the range already observed, so the model interpolates instead. Making this change improved
every model.

### Leakage prevention

These columns are deliberately **excluded** from the feature matrix:

| Excluded | Reason |
|---|---|
| `active_power` | it is the target |
| `energy` | `E = P·Δt/1000` — a rescaled copy of the target |
| `voltage`, `current`, `power_factor`, `frequency` | measured at the *same instant* as the target and bound to it by P = V·I·PF; unavailable when forecasting a future hour |

Rolling statistics use `shift(1).rolling(w)`, never `rolling(w)` directly, so the current
sample cannot contribute to its own rolling mean.

Notebook 02 demonstrates the consequence of getting this wrong: adding `current` back into
the features lifts R² from **0.77 to 0.9987** — not a better forecaster, just a model reading
the answer off its input.

---

## Model Evaluation

All metrics are computed on the held-out September data and regenerated on every training
run. Nothing below is hard-coded.

| Model | MAE (W) | RMSE (W) | R² | MAPE (%) |
|---|---|---|---|---|
| Linear Regression | 112.27 | 188.39 | 0.7514 | 15.44 |
| **Random Forest** | **100.26** | **172.86** | **0.7907** | **14.08** |
| Gradient Boosting | 127.68 | 231.66 | 0.6241 | 18.41 |

**Selection criterion, defined before training:** lowest test **RMSE**. RMSE squares errors
before averaging, so a model that is usually close but occasionally very wrong is penalised.
For peak-load planning, the occasional large miss is the expensive one. Random Forest wins.

### Metrics explained

- **MAE** — mean absolute error in watts. "On average the forecast is off by 100 W."
- **RMSE** — root mean squared error in watts. Dominated by the largest misses.
- **R²** — share of load variance explained. 0.79 means 79 % explained; 0.0 would be no
  better than always predicting the mean.
- **MAPE** — mean absolute percentage error, computed only where the true value is
  comfortably non-zero.

### Baseline comparison

A forecasting model earns its place only by beating naive baselines:

| Approach | MAE (W) | RMSE (W) | R² |
|---|---|---|---|
| Persistence (`lag_1`, last hour repeated) | 168.05 | 257.29 | 0.5364 |
| Seasonal naive (`lag_24`, same hour yesterday) | 130.51 | 261.43 | 0.5214 |
| **Random Forest (selected)** | **100.26** | **172.86** | **0.7907** |

The model reduces RMSE by ~33 % against the better baseline.

### Honest limitations

- Residual error concentrates at the **evening peak** and during abnormal episodes — the
  hardest hours, and the ones a utility cares about most.
- Multi-step forecasts are **recursive**: each prediction becomes the lag input for the next,
  so error compounds with horizon. Short horizons are the trustworthy ones.
- No weather or occupancy inputs, which drive real load significantly.
- The dataset is synthetic. Real meter data is noisier, has missing periods and drifts.

---

## Anomaly Detection

```bash
python -m src.anomaly.detector
```

**Isolation Forest** over `active_power`, `current`, `voltage` and `power_factor`, scaled
with a `StandardScaler` so that watts (hundreds) do not dominate power factor (≈ 1).

**Why Isolation Forest?** It is unsupervised — there are no labelled faults — it handles
multivariate data, and the mechanism is simple to defend: random trees split on random
features at random thresholds; points far from the bulk of the data get isolated in very few
splits, so a *short average path length* means *anomalous*.

Result on the shipped dataset: **132 of 6,552 samples flagged (2.01 %)**, mean power 2,134 W
for flagged samples versus 777 W for normal ones.

> **Interpretation.** A flag means an **abnormal consumption pattern** — a reading that does
> not resemble the rest of the data. It is **not** evidence of equipment failure. Proving
> failure would require labelled fault data, which this project does not have. Because the
> anomalies are unlabelled, precision and recall cannot be measured.

---

## Dashboard

```bash
streamlit run app.py
```

Seven pages, navigated from the sidebar:

| Page | Contents |
|---|---|
| **Dashboard Overview** | Voltage, current, active power, energy, power factor, frequency; consumption and peak KPIs; 14-day load trend; hourly profile |
| **Historical Analysis** | Date-range filter; load over time; daily/weekly/monthly energy; hourly and weekday profiles; peak-period chart with 90th-percentile threshold |
| **Load Forecast** | Selected model and its test metrics; adjustable 1–48 h horizon; next-hour prediction; forecast chart; back-test with actual vs predicted and error distribution |
| **Anomaly Detection** | Anomaly count and percentage; timeline; V–I operating-point scatter; ranked event list |
| **Cost Estimation** | Tariff-driven daily/monthly/yearly estimates; per-day cost chart; tariff sensitivity table |
| **Real-Time Monitor** | Live simulated stream with SIMULATION MODE banner, anomaly scoring per reading, optional SQLite storage, auto-refresh |
| **System & Model Info** | Model metadata, comparison table and chart, feature list, database contents, MQTT status |

The sidebar always shows system health and the tariff control:

```
Dataset        ● Ready
ML model       ● Loaded
Anomaly model  ● Loaded
Database       ● Connected
MQTT           ● Offline
```

The dashboard never crashes when a component is missing — each page reports what is absent
and prints the command that creates it.

---

## MQTT Architecture

Expected payload published by a node:

```json
{
  "device_id": "ENERGY_NODE_01",
  "timestamp": "2026-09-17T10:30:00",
  "voltage": 230.5,
  "current": 2.8,
  "power_factor": 0.94,
  "frequency": 50.0
}
```

`active_power` and `energy` may be omitted — the backend derives them, so a constrained MCU
can send the bare minimum.

Ingestion path: **connect → subscribe → receive → parse JSON → validate → convert → store**.

Validation rejects malformed JSON, missing fields, non-numeric fields and physically
implausible values, with a human-readable reason for each. A rejected message is logged and
dropped; the subscriber loop keeps running.

**MQTT is entirely optional.** If `paho-mqtt` is not installed, if `MQTT_BROKER` is unset, or
if the broker is unreachable, `connect()` returns `False` instead of raising and the dashboard
continues in simulation mode. There are dedicated tests for all three cases, including an
unroutable broker address.

To try it with a local broker:

```bash
# terminal 1
mosquitto -p 1883

# .env
MQTT_BROKER=localhost

# terminal 2
python -m src.iot.mqtt_client
```

---

## ESP32/STM32 Integration

```
┌──────────────────────────────────────────────┐
│  Isolated measurement front-end              │
│  • Voltage: rated transformer / isolated      │
│    sensing module with correct CAT rating     │
│  • Current: clamp-type CT or Hall sensor      │
│  • Proper fusing, enclosure and creepage      │
└───────────────────┬──────────────────────────┘
                    │ safe low-voltage signals
                    ▼
        ESP32 / STM32  →  ADC sampling, RMS computation, PF estimation
                    │
                    ▼   Wi-Fi (ESP32) or Ethernet/Wi-Fi module (STM32)
              MQTT publish  →  topic: energy/data
                    │
                    ▼
        Python backend (this repository) — unchanged
```

> **Electrical safety.** Do not connect mains voltage directly to a microcontroller pin.
> Mains measurement requires properly rated isolated measurement hardware (isolation
> transformers or certified sensing modules, current transformers), correct fusing, adequate
> creepage and clearance, and an enclosure. Work on live mains should be done under
> supervision by someone qualified. This repository requires no hardware to run.

Because the backend consumes a JSON contract rather than a driver, swapping the simulator for
a real node changes nothing above the MQTT layer.

---

## Running the Application

```bash
# 1. Generate the synthetic development dataset
python -m src.data.generate_data

# 2. Validate and clean it
python -m src.data.preprocess

# 3. Train and compare models, save the winner
python -m src.models.train

# 4. Fit the anomaly detector
python -m src.anomaly.detector

# 5. Launch the dashboard
streamlit run app.py
```

Individual modules also run standalone for inspection:

```bash
python -m src.models.predict     # next-hour + 24 h forecast
python -m src.iot.simulator      # stream of simulated readings
python -m src.iot.mqtt_client    # MQTT status and payload validation demo
python -m src.database.db        # database summary
```

---

## Running Tests

```bash
pytest
```

124 tests across 7 files:

| File | Covers |
|---|---|
| `test_data.py` | Generation, reproducibility, plausible ranges, P = V·I·PF, validation findings, cleaning actions |
| `test_features.py` | Calendar and cyclical features, **lag direction**, **rolling-window leakage**, excluded columns, prediction-row alignment |
| `test_model.py` | Metric correctness, RMSE vs MAE behaviour, model selection, chronological split ordering, beating a mean baseline, forecast sanity |
| `test_anomaly.py` | Fit/predict contract, label consistency, score ordering, save/load round-trip, determinism |
| `test_database.py` | Schema creation, idempotency, derived fields, date queries, aggregation, transaction rollback |
| `test_iot.py` | Simulator determinism and physics, payload validation, graceful MQTT failure |
| `test_analytics.py` | Energy conservation across aggregations, peak logic, cost linearity |

Notable tests: `test_lag_values_come_from_the_past`, `test_rolling_features_exclude_the_current_sample`,
`test_chronological_split_never_shuffles` and `test_unreachable_broker_does_not_crash_the_app`
each guard a specific failure mode described in this README.

---

## GitHub Actions

`.github/workflows/python-tests.yml` runs on every push and pull request to `main`, across
Python 3.10, 3.11 and 3.12:

1. Check out the repository
2. Install Python with pip caching
3. Install `requirements.txt`
4. Build the pipeline artefacts (generate → preprocess → train → anomaly)
5. Run `pytest`
6. Verify the dashboard module imports cleanly

Step 4 means CI validates the entire project from a clean clone, not just the unit tests.

---

## Results

All figures below are produced by the commands in this README and regenerate on every run.

**Dataset:** 6,552 hourly samples · 19.31 kWh/day average · 4,632 W peak
**Forecasting:** Random Forest selected — MAE 100.26 W, RMSE 172.86 W, R² 0.7907
**Against baselines:** ~33 % lower RMSE than seasonal-naive
**Anomaly detection:** 132 samples flagged (2.01 %)
**Peak behaviour:** peak hour 19:00 · load factor 0.174
**Tests:** 124 passed

---

## Screenshots

Place PNGs in `screenshots/` after running the dashboard and they will render here.

| View | File |
|---|---|
| Dashboard Overview | `screenshots/01_overview.png` |
| Historical Analysis | `screenshots/02_historical.png` |
| Load Forecast | `screenshots/03_forecast.png` |
| Anomaly Detection | `screenshots/04_anomaly.png` |
| Cost Estimation | `screenshots/05_cost.png` |
| Real-Time Monitor | `screenshots/06_realtime.png` |

```markdown
![Dashboard Overview](screenshots/01_overview.png)
```

---

## Future Scope

- **Real energy meter** — replace the synthetic dataset with measurements from a certified
  meter or an isolated sensing front-end.
- **ESP32 / STM32 node** — build the hardware against the MQTT contract already defined here.
- **Edge AI** — run a quantised model on the node (TensorFlow Lite Micro) so forecasting
  survives a network outage.
- **Cloud deployment** — hosted broker, time-series database (InfluxDB/TimescaleDB) and a
  deployed dashboard.
- **Mobile application** — push notifications when an abnormal pattern is detected.
- **Advanced forecasting** — SARIMA, Prophet or LSTM/Temporal Fusion Transformer, plus
  weather and occupancy features; probabilistic forecasts with prediction intervals.
- **Appliance-level disaggregation (NILM)** — infer which appliances are running from the
  aggregate signal.
- **Three-phase support** — per-phase measurement, imbalance detection and reactive power.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

Built with scikit-learn, pandas, Streamlit and Plotly. The synthetic load model follows the
standard domestic load-curve shape described in power-systems literature.
#   S m a r t - E n e r g y - M a n a g e m e n t  
 