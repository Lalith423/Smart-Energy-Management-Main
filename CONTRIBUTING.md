# Contributing

Thanks for your interest in improving this project.

## Development setup

```bash
git clone <your-fork-url>
cd Smart-Energy-Monitoring

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

Build the artefacts once so every module has data to work with:

```bash
python -m src.data.generate_data
python -m src.data.preprocess
python -m src.models.train
python -m src.anomaly.detector
```

## Before opening a pull request

1. `pytest` passes locally.
2. New behaviour comes with a test.
3. Public functions have docstrings and type hints where they help.
4. No secrets, credentials or `.env` files are committed.
5. No generated artefacts (`data/raw/*.csv`, `models/*.pkl`, `*.db`) are committed —
   they are reproducible from the scripts and are listed in `.gitignore`.

## Style

- Line length 100, `black`-compatible formatting.
- Keep modules single-purpose: data, features, models, anomaly, database, iot, utils.
- Business logic belongs in `src/`; `app.py` and `dashboard/` only render.
- Never shuffle time-series data, and never build a feature from information that
  would be unavailable at prediction time.

## Reporting issues

Include the command you ran, the full traceback, your Python version and your OS.
