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

### Paper Trading

- `MIMIC_PAPER_ENABLED` (default: `false`)
- `MIMIC_PAPER_WINDOW_DAYS` (default: `14`)
- `MIMIC_PAPER_MIN_CONFIDENCE` (default: `0.6`)
- `MIMIC_PAPER_MIN_PARTICIPANTS` (default: `2`)
- `MIMIC_PAPER_MIN_TOTAL_PARTICIPANTS` (default: `2`)
- `MIMIC_PAPER_WEIGHTED_CONSENSUS_MIN` (default: `0.0`)
- `MIMIC_PAPER_MIN_WEIGHTED_PARTICIPANTS` (default: `0.0`)
- `MIMIC_PAPER_ENTRY_MIN` (default: `0.0`)
- `MIMIC_PAPER_ENTRY_MAX` (default: `1.0`)
- `MIMIC_PAPER_REQUIRE_TIGHT_BAND` (default: `false`)
- `MIMIC_PAPER_EV_MIN` (default: `-1.0`)
- `MIMIC_PAPER_BET_SIZING` (default: `bankroll`)
- `MIMIC_PAPER_BASE_BET` (default: `100.0`)
- `MIMIC_PAPER_MAX_BET` (default: `500.0`)
- `MIMIC_PAPER_STARTING_BANKROLL` (default: `200.0`)
- `MIMIC_PAPER_BET_FRACTION` (default: `0.02`)

## API

- `GET /health`
- `GET /refresh` (optional query: `n_wallets`, `trades_limit`)
- `GET /state`
- `GET /market/{conditionId}`
- `GET /paper/state`
- `GET /paper/trades` (optional query: `status`, `limit`)
