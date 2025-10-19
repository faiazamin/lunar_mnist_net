import mlflow.pytorch
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import MNISTNet

app  = FastAPI(title="MNISTNet API")

model = MNISTNet()
model.load_state_dict(torch.load("model/model.pt"))
model.eval

mlflow.search_experiments("inference_monitoring")

class InputData(BaseModel):
    features: list[float]


@app.post("/predict")
def post(input: InputData):
    x = torch.tensor([input.features], type = torch.float32)
    with torch.no_grad:
        logits = model(x)
        pred = int(torch.argmax(logits))
    mlflow.log_metric("prediction", pred)
    mlflow.log_params("input length", len(x))
    return {"prediction": pred}






