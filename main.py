from fastapi import FastAPI
import uvicorn
from FastAPI.Utils.load_json import *
from FastAPI.Utils.utils import *
from FastAPI.API.collector import *
from FastAPI.API.predictor import *
from FastAPI.Class.PAA import *
from FastAPI.Class.CollectClass import *
from FastAPI.Utils.preprocess_pressure_img import *
from LMDB.controller.lmdb_controller import LMDBManager
from datetime import datetime,timedelta
from FastAPI.Model.Onbed_model import *
from FastAPI.Model.Pose_model import *
from FastAPI.Model.Action_model import *
from FastAPI.Utils.dataloader import *
from FastAPI.Utils.train_Onbed import *
from FastAPI.Utils.train_Pose import *
from FastAPI.Utils.train_Action import *
import wandb
import zipfile
import os

os.environ['WANDB_BASE_URL'] = "http://192.168.1.121:1123"
os.environ['WANDB_API_KEY'] = "local-00f57935806148c0ce6b0c5623b2b826ad2ee681"

app = FastAPI()

# 加载lmdb数据库
arg = load_json("./FastAPI/hypter/lmdb.json")
arg = load_json("./FastAPI/hypter/predict.json", arg)
db_manager = LMDBManager(arg["lmdb_path"])
write_queue = db_manager.create()

# 保存标注
collect_label = {}
action_list = ["正卧（一级）", "正卧（二级）", "俯卧（一级）", "俯卧（二级）", "左侧卧（一级）", "左侧卧（二级）", "右侧卧（一级）", "右侧卧（二级）","坐床头", "坐床边", "坐中间", "手掌", "站立", "三级体动"]
index = 0

model_Onbed = Onbed_model(hypers=arg).cuda()
model_Pose = Pose_model(hypers=arg).cuda()
model_Action = Action_model(hypers=arg).cuda()
data_collect = np.zeros((20, 160, 320))

# 缓冲池
buffer_pool = {}


@app.get("/test")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
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
        global data_collect,model_Onbed
        action,data_collect = predictor(arg, pressure_datas, ID, write_queue,data_collect,model_Onbed)
    else:
        global buffer_pool
        buffer_pool = collector(arg, pressure_datas, ID, write_queue, collect_label,buffer_pool)

    predict_time = time.time() - begin_time
    print("Predict time: ", predict_time, "seconds")

    return {"Hello": "World"}

@app.post("/begin_collect")
def begin_collect(data: CCInputData):
    time = datetime.strptime(data.Time, "%Y/%m/%d %H:%M:%S")
    time = time + timedelta(seconds=1)
    
    ID = data.ID
    action = data.Action
    collect_label[ID] = CCRecordData(ID=ID, action=action, begin_time=time, end_time=None)
    return {"Hello": "World"}

@app.post("/finish_collect")
def finish_collect(data: CCInputData):
    time = datetime.strptime(data.Time, "%Y/%m/%d %H:%M:%S")
    time = time - timedelta(seconds=1)

    ID = data.ID
    action = data.Action
    collect_label[ID].end_time = time
    return {"Hello": "World"}

@app.get("/collect_action")
def collect_action():
    global index
    if index >= len(action_list):
        index = 0
        action = action_list[index]
        index += 1
        return {"action": action_list[index]}
    else:
        action = action_list[index]
        index += 1
        return {"action": action}

@app.get("/get_database_key_number")
def get_database_key_number():
    return {"number":db_manager.get_second_list_length()}

@app.get("/clean_database")
def clean_database():
    db_manager.clear_databases()
    return {"Hello": "World"}

@app.get("/train_Onbed")
def train_Onbed():

    run = wandb.init(project='RestNightAI', entity='iomgaa')
    artifact = run.use_artifact('iomgaa/RestNightAI/Onbed_data:v2', type='dataset')
    artifact_dir = artifact.download()

    # 解压zip压缩包
    extract_dir = os.path.join(artifact_dir, "database")
    with zipfile.ZipFile(artifact_dir+"/database.zip", 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    

    # 读取预测参数
    arg = load_json("./FastAPI/hypter/train_Onbed.json")
    arg["lmdb_path"] = extract_dir
    Database_name = arg["Database_name"]
    missing_databases = db_manager.check_databases(Database_name)# 检查数据库是否存在

    # 初始化模型
    model = Onbed_model(arg)
    model = initialize_layers(model)
    model = model.to(arg["device"])

    # 初始化优化器与学习率衰减器
    optimizer = torch.optim.Adam(model.parameters(), lr=arg["lr"], weight_decay=arg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg["step_size"], gamma=arg["gamma"])

    # 训练
    for epoch in range(arg["epochs"]):
        train_dataset = PressureDataset(db_manager, None, phase="train",db_name="train")
        train_loader = DataLoader(train_dataset, batch_size=arg["batch_size"], shuffle=True, num_workers=arg["num_workers"])
        train_loss_l2, train_loss_ssim, rate = train_Onbed(model, train_loader, optimizer, arg)
        val_dataset = PressureDataset(db_manager, None, phase="val",db_name="val")
        val_loader = DataLoader(val_dataset, batch_size=arg["batch_size"], shuffle=True, num_workers=arg["num_workers"])
        val_loss_l2, val_loss_ssim, accuracy = val_Onbed(model, val_loader, optimizer, arg, rate)
        
        scheduler.step()
        print("epoch: ", epoch, "train_loss_l2: ", train_loss_l2, "train_loss_ssim: ", train_loss_ssim, "val_loss_l2: ", val_loss_l2, "val_loss_ssim: ", val_loss_ssim, "accuracy: ", accuracy)
        
        # Log metrics to Wandb
        wandb.log({"train_loss_l2": train_loss_l2, "train_loss_ssim": train_loss_ssim, "val_loss_l2": val_loss_l2, "val_loss_ssim": val_loss_ssim, "accuracy": accuracy})

    print(missing_databases)

@app.get("/train_SleepPose")
def train_SleepPose():
    run = wandb.init(project='RestNightAI', entity='iomgaa')
    artifact = run.use_artifact('iomgaa/RestNightAI/Sleep_Pose_data:v3', type='dataset')
    artifact_dir = artifact.download()

    # 解压zip压缩包
    extract_dir = os.path.join(artifact_dir)
    with zipfile.ZipFile(artifact_dir+"/database.zip", 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # 读取预测参数
    arg = load_json("./FastAPI/hypter/train_Pose.json")
    arg["lmdb_path"] = extract_dir+"/database"
    Database_name = arg["Database_name"]
    db_manager = LMDBManager(arg["lmdb_path"])
    missing_databases = db_manager.check_databases(Database_name)# 检查数据库是否存在

    # 初始化模型
    model = Pose_model(arg)
    model = initialize_layers(model)
    model = model.to(arg["device"])

    # 初始化优化器与学习率衰减器
    params = [
    {'params': [p for n, p in model.named_parameters() if 'bias' not in n], 'weight_decay': arg["weight_decay"]},
    {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': arg["bias_decay"]}
    ]
    optimizer = torch.optim.Adam(params=params, lr=arg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg["step_size"], gamma=arg["gamma"])

    # 训练
    for epoch in range(arg["epochs"]):
        train_dataset = PressureDataset(db_manager, None, phase="train",db_name="train", mode="Pose")
        train_loader = DataLoader(train_dataset, batch_size=arg["batch_size"], shuffle=True, num_workers=arg["num_workers"])
        train_loss, train_acc = train_Pose(model, train_loader, optimizer, arg)
        val_dataset = PressureDataset(db_manager, None, phase="val",db_name="val", mode="Pose")
        val_loader = DataLoader(val_dataset, batch_size=arg["batch_size"], shuffle=True, num_workers=arg["num_workers"])
        val_loss, val_acc = val_Pose(model, val_loader, arg)
        scheduler.step()
        print("epoch: ", epoch, "train_loss: ", train_loss, "train_acc: ", train_acc, "val_loss: ", val_loss, "val_acc: ", val_acc)

        # Log metrics to Wandb
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

@app.get("/train_SleepAction")
def train_SleepAction():
    run = wandb.init(project='RestNightAI', entity='iomgaa')
    artifact = run.use_artifact('iomgaa/RestNightAI/Sleep_Action_data:v0', type='dataset')
    artifact_dir = artifact.download()

    # 解压zip压缩包
    extract_dir = os.path.join(artifact_dir)
    with zipfile.ZipFile(artifact_dir+"/database.zip", 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # 读取预测参数
    arg = load_json("./FastAPI/hypter/train_Action.json")
    arg["lmdb_path"] = extract_dir+"/database"
    Database_name = arg["Database_name"]
    db_manager = LMDBManager(arg["lmdb_path"])
    missing_databases = db_manager.check_databases(Database_name)

    # 初始化模型
    model = Action_model(arg)
    model = initialize_layers(model)
    model = model.to(arg["device"])

    # 初始化优化器与学习率衰减器
    params = [
    {'params': [p for n, p in model.named_parameters() if 'bias' not in n], 'weight_decay': arg["weight_decay"]},
    {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': arg["bias_decay"]}
    ]
    optimizer = torch.optim.Adam(params=params, lr=arg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg["step_size"], gamma=arg["gamma"])

    # 训练
    for epoch in range(arg["epochs"]):
        train_dataset = PressureDataset(db_manager, None, phase="train",db_name="train", mode="Action",args=arg)
        train_loader = DataLoader(train_dataset, batch_size=arg["batch_size"], shuffle=True, num_workers=arg["num_workers"])
        train_loss, train_acc = train_Action(model, train_loader, optimizer, arg)
        val_dataset = PressureDataset(db_manager, None, phase="val",db_name="val", mode="Action",args=arg)
        val_loader = DataLoader(val_dataset, batch_size=arg["batch_size"], shuffle=True, num_workers=arg["num_workers"])
        val_loss, val_acc = val_Action(model, val_loader, arg)
        scheduler.step()
        print("epoch: ", epoch, "train_loss: ", train_loss, "train_acc: ", train_acc, "val_loss: ", val_loss, "val_acc: ", val_acc)

        # Log metrics to Wandb
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

