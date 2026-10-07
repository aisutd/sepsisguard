from fastapi import FastAPI 
from confidence import score_confidence
app= FastAPI()

@app.post("/confidence")
def confidence(row:dict):
    return score_confidence(row)

@app.get("/health")
def health():
    return {"status": "ok"}
