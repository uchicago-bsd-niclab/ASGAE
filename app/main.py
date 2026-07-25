
from fastapi import FastAPI
from pydantic import BaseModel
from app.utils import run_training, Run_Test
import torch
import sys, os
# Docker image with PyTorch3D installed:
# https://hub.docker.com/r/gwangjin/pytorch3d
#
# Add /app and /app/app to sys.path
sys.path.append("/app")
sys.path.append("/app/app")

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok"}

class Params(BaseModel):
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
    print("Training started with parameters:", params)
    print("Taining start....")
    result = run_training(params)
    return result

@app.post("/validate")
async def validate(params: Params):
    result = Run_Test(params)
    return result

@app.get("/gpu")
def check_gpu():
    return {"gpu_available": torch.cuda.is_available()}
