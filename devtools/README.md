# Mimic Devtools

## Frontend

From `devtools/`:

```bash
npm --prefix frontend install
npm run dev
```

Or directly:

```bash
cd frontend
npm install
npm run dev
```

## Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

