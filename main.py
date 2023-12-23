from fastapi import FastAPI
import uvicorn
from FastAPI.Utils.load_json import *
from FastAPI.API.collector import *
from FastAPI.API.predictor import *

app = FastAPI()


@app.get("/test")
def read_root():
    return {"Hello": "World"}

@app.get("/predict")
def predict():
    arg = load_json("./FastAPI/hypter/predict.json")
    if arg["status"] == True:
        predictor(arg)
    else:
        collector(arg)

    return {"Hello": "World"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

