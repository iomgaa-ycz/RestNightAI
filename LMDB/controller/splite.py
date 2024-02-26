import lmdb
import wandb
import json
from lmdb_controller import LMDBManager
import os
import shutil

# os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_BASE_URL'] = "http://192.168.1.121:1141"
os.environ['WANDB_API_KEY'] = "local-9a5876dc995accd0691a161ba6967e414a9c6b28"

def split_databases(db_path, train_names=["yuchengzhang"], val_names=["baishuhang"]):
    
    val_data = []
    train_data = []
    data_dict = {}
    lmdb_manager = LMDBManager(db_path=db_path,max_dbs=20)
    lmdb_manager.second_db = lmdb_manager.env.open_db("train".encode('utf-8'))
    lmdb_manager.hypter_db = lmdb_manager.env.open_db("val".encode('utf-8'))
    lmdb_manager.clear_databases()
    keys = lmdb_manager.get_keys()
    if len(keys) == 0:
        print("The database has already been splited!")
    lmdb_manager.clear_databases()
    for name in train_names:
        lmdb_manager.second_db = lmdb_manager.env.open_db(name.encode('utf-8'))
        keys = lmdb_manager.get_keys()
        for key in keys:
            value_json = lmdb_manager.read(key)
            value = json.loads(value_json)
            if value["action"] not in ["正卧（一级）", "正卧（二级）", "俯卧（一级）", "俯卧（二级）", "左侧卧（一级）", "左侧卧（二级）", "右侧卧（一级）", "右侧卧（二级）"]:
                continue
            else:
                train_data.append(key)
                data_dict[key] = name

    for name in val_names:
        lmdb_manager.second_db = lmdb_manager.env.open_db(name.encode('utf-8'))
        keys = lmdb_manager.get_keys()
        for key in keys:
            value_json = lmdb_manager.read(key)
            value = json.loads(value_json)
            if value["action"] not in ["正卧（一级）", "正卧（二级）", "俯卧（一级）", "俯卧（二级）", "左侧卧（一级）", "左侧卧（二级）", "右侧卧（一级）", "右侧卧（二级）"]:
                continue
            else:
                val_data.append(key)
                data_dict[key] = name
    # print("Number of elements in val: ", len(val_data))
    # print("Number of elements in train: ", len(train_data))
    
    
    # Write train_data to "train" database
    lmdb_manager.hypter_db = lmdb_manager.env.open_db("train".encode('utf-8'))
    db_name = "yuchengzhang"
    lmdb_manager.second_db = lmdb_manager.env.open_db(db_name.encode('utf-8'))
    for key in train_data:
        if db_name != data_dict[key]:
            lmdb_manager.second_db = lmdb_manager.env.open_db(data_dict[key].encode('utf-8'))
            db_name = data_dict[key]
        value_json = lmdb_manager.read(key)
        lmdb_manager.add_data_to_db("hypter_db",key, value_json)
    
    # Write val_data to "val" database
    lmdb_manager.hypter_db = lmdb_manager.env.open_db("val".encode('utf-8'))
    db_name = "yuchengzhang"
    lmdb_manager.second_db = lmdb_manager.env.open_db(db_name.encode('utf-8'))
    for key in val_data:
        if db_name != data_dict[key]:
            lmdb_manager.second_db = lmdb_manager.env.open_db(data_dict[key].encode('utf-8'))
            db_name = data_dict[key]
        value_json = lmdb_manager.read(key)
        lmdb_manager.add_data_to_db("hypter_db",key, value_json)
    
    lmdb_manager.second_db = lmdb_manager.env.open_db("train".encode('utf-8'))
    keys = lmdb_manager.get_keys()
    print("Number of elements in train: ", len(keys))
    lmdb_manager.second_db = lmdb_manager.env.open_db("val".encode('utf-8'))
    keys = lmdb_manager.get_keys()
    print("Number of elements in val: ", len(keys))
    
    lmdb_manager.env.close()




# split_databases(db_path="./LMDB/database", train_names=['aiyubo' ,'Zhaozhiheng','gaoyuqi','caiwentao','liujunjie','fanghaolun','shunjian','wangjin','yuchengzhang'], val_names=["baishuhang"])
run = wandb.init(project="RestNightAI", job_type="load-data")
raw_data = wandb.Artifact(
            "Sleep_Pose_data", type="dataset",
            description="睡姿检测数据集，包含十个人的数据。白书航的数据作为验证集，其他数据作为训练集。",
            metadata={"subject": {
                "train":['aiyubo' ,'Zhaozhiheng','gaoyuqi','caiwentao','liujunjie','fanghaolun','shunjian','wangjin','yuchengzhang'],
                "val": ["baishuhang"]},
                      "type": "Sleep Pose"})
raw_data.add_file("./LMDB/database.zip")
run.log_artifact(raw_data)
print("Done")