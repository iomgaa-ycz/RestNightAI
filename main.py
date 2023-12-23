from fastapi import FastAPI
import uvicorn
from FastAPI.Utils.load_json import *
from FastAPI.API.collector import *
from FastAPI.API.predictor import *
from FastAPI.Class.PAA import *
from LMDB.controller.lmdb_controller import LMDBManager

app = FastAPI()

# 加载lmdb数据库
arg = load_json("./FastAPI/hypter/lmdb.json")
db_manager = LMDBManager(arg["lmdb_path"])
write_queue = db_manager.create()


@app.get("/test")
def read_root():
    return {"Hello": "World"}

@app.get("/predict")
def predict(data: PAAInputData):
    arg = load_json("./FastAPI/hypter/predict.json")

    pressure_datas = data.PressureMap
    ID = data.ID

    if arg["status"] == True:
        predictor(arg)
    else:
        collector(arg)

    return {"Hello": "World"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

