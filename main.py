from fastapi import FastAPI
import uvicorn
from FastAPI.Utils.load_json import *
from FastAPI.API.collector import *
from FastAPI.API.predictor import *
from FastAPI.Class.PAA import *
from FastAPI.Utils.preprocess_pressure_img import *
from LMDB.controller.lmdb_controller import LMDBManager

app = FastAPI()

# 加载lmdb数据库
arg = load_json("./FastAPI/hypter/lmdb.json")
arg = load_json("./FastAPI/hypter/predict.json", arg)
db_manager = LMDBManager(arg["lmdb_path"])
write_queue = db_manager.create()


@app.get("/test")
def read_root():
    return {"Hello": "World"}

@app.get("/predict")
def predict(data: PAAInputData):
    """
    预测函数，根据输入的数据进行预测。

    Args:
        data (PAAInputData): 输入的数据对象，包含压力图和ID。

    Returns:
        dict: 预测结果，示例为{"Hello": "World"}。
    """
    begin_time = time.time()

    # 读取预测参数
    arg = load_json("./FastAPI/hypter/predict.json")

    # 读取输入数据
    pressure_datas = data.PressureMap
    ID = data.ID

    # 预处理
    pressure_datas = base64_to_image_list(pressure_datas,ID)
    pressure_datas = preprocess(pressure_datas)

    # 如果为True则为预测，否则为采集
    if arg["status"] == True:
        predictor(arg, pressure_datas, ID, write_queue)
    else:
        collector(arg, pressure_datas, ID, write_queue)

    end_time = time.time()
    predict_time = end_time - begin_time
    print("Predict time: ", predict_time, "seconds")

    return {"Hello": "World"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
