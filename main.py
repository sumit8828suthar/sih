from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from typing import List, Dict

from model_serve import router as model_serve

app = FastAPI()

# Add CORSMiddleware to allow requests from specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # You can allow specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.include_router(model_serve)

@app.get("/")
def read_root():
    return {"Data": "Model Server v0.1"}