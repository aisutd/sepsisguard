# SepsisGuard

## Description

SepsisGuard is an AI-powered early warning system that monitors ICU vitals and lab values to predict sepsis risk hours before standard clinical assessment. It combines temporal modeling with explainable, verified natural-language reasoning to help clinical teams understand the factors behind each risk score.

## Planned Technologies

- Python, JavaScript, and SQL
- React and FastAPI
- PyTorch Forecasting, Temporal Fusion Transformer, SHAP, and XGBoost
- Hugging Face models and PostgreSQL

## Docker Setup

Make sure Docker Desktop is installed and running.

```bash
docker compose up -d
docker compose down
docker compose ps
```

This is a generic starter configuration. The team can add project-specific dependencies and startup commands later.

## How to run the app

You need two terminals, both starting in the `sepsisguard/` folder.

**1. Backend** (FastAPI, http://localhost:8000):
```powershell
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

**2. Frontend** (React + Vite, http://localhost:5173):
```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The frontend reads everything from the backend,
so start the backend first. If it isn't running, the screen says so.
