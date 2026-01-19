# Mimic Backend (FastAPI)

Bare-bones Polymarket copy-trading signal backend.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

## Config (env)

- `MIMIC_DB_PATH` (default: `./mimic.sqlite3`)
- `MIMIC_N_WALLETS` (default: `50`)
- `MIMIC_TRADES_LIMIT` (default: `100`)
- `MIMIC_REFRESH_INTERVAL_SEC` (default: `60`)
- `MIMIC_ENABLE_PRICING` (default: `true`)

## API

- `GET /health`
- `GET /refresh` (optional query: `n_wallets`, `trades_limit`)
- `GET /state`
- `GET /market/{conditionId}`

