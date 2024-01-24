import lmdb
import wandb
import json
from lmdb_controller import LMDBManager
import os
import shutil

# os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_BASE_URL'] = "http://192.168.1.121:1123"
os.environ['WANDB_API_KEY'] = "local-00f57935806148c0ce6b0c5623b2b826ad2ee681"

def split_databases(db_path, train_names=["yuchengzhang"], val_names=["baishuhang"]):
    
    val_data = []
    train_data = []
    data_dict = {}
    lmdb_manager = LMDBManager(db_path=db_path)
    lmdb_manager.second_db = lmdb_manager.env.open_db("train".encode('utf-8'))
    lmdb_manager.hypter_db = lmdb_manager.env.open_db("val".encode('utf-8'))
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
    print("Number of elements in val: ", len(val_data))
    print("Number of elements in train: ", len(train_data))
    
    
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
    
    lmdb_manager.env.close()




split_databases(db_path="./LMDB/database", train_names=["yuchengzhang","shunjian","fanghaolun"], val_names=["baishuhang"])
run = wandb.init(project="RestNightAI", job_type="load-data")
raw_data = wandb.Artifact(
            "Sleep_Pose_data", type="dataset",
            description="睡姿检测数据集，包含余承璋与白书航两个人的数据。余承璋的数据作为训练集，白书航的数据作为验证集。",
            metadata={"subject": ["yuchengzhang", "baishuhang"],
                      "type": "Sleep Pose"})
raw_data.add_file("./LMDB/database.zip")
run.log_artifact(raw_data)
print("Done")