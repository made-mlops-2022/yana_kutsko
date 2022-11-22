import pickle
import pandas as pd
import uvicorn

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, HTTPException
from starlette.responses import PlainTextResponse

from online_inference.server.src.model_data import ModelData

app = FastAPI()
model = None


@app.on_event("startup")
def load_model():
    with open("../../../ml_project/models/logreg.pkl", "rb") as file:
        global model
        model = pickle.load(file)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(f"Data is invalid: {exc}")
    return PlainTextResponse(f"Data is invalid: {exc}", status_code=400)


@app.get("/health")
async def check_health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model was not loaded")


@app.post("/predict")
async def predict(data: ModelData):
    x_pred = pd.DataFrame.from_dict({k: [v] for k, v in jsonable_encoder(data).items()})
    y_pred = pd.Series(model.predict(x_pred))
    return {"result": y_pred.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
