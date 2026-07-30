"""FastAPI application exposing ASGAE training and validation endpoints."""

from fastapi import FastAPI
from pydantic import BaseModel
from app.utils import run_training, Run_Test
import torch
import sys, os
# Docker image with PyTorch3D installed:
# https://hub.docker.com/r/gwangjin/pytorch3d
#
# Add application directories to the import path.
sys.path.append("/app")
sys.path.append("/app/app")

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

app = FastAPI()

@app.get("/")
def read_root():
    """Return a minimal health-check response."""
    return {"status": "ok"}

class Params(BaseModel):
    """Validated request body shared by training and validation endpoints.

    ``k_order`` controls the Chebyshev polynomial order. The input-channel
    count is inferred from the documented RGB mesh format.
    """
    db_path: str
    batch_size: int
    test_batch_size: int
    k_order: int
    embedding_dim: int
    optimizer: str
    learning_rate: float
    epochs: int
    model_path: str
    train_set_percentage: float
    val_set_percentage: float
    test_set_percentage: float
    full_test_dataset: bool
    verbose: bool

@app.post("/train")
async def train(params: Params):
    """Train a model using the supplied dataset and hyperparameters."""
    print("Training started with parameters:", params)
    print("Training started.")
    result = run_training(params)
    return result

@app.post("/validate")
async def validate(params: Params):
    """Evaluate a saved model using the supplied dataset and split settings."""
    result = Run_Test(params)
    return result

@app.get("/gpu")
def check_gpu():
    """Report whether PyTorch can use CUDA in the running container."""
    return {"gpu_available": torch.cuda.is_available()}
