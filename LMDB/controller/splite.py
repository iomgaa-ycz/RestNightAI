import lmdb
import wandb
import json
from lmdb_controller import LMDBManager
import os
import shutil

# os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_BASE_URL'] = "http://192.168.1.121:1123"
os.environ['WANDB_API_KEY'] = "local-00f57935806148c0ce6b0c5623b2b826ad2ee681"

def split_databases(db_path, db_names):
    
    val_data = []
    train_data = []
    data_dict = {}
    lmdb_manager = LMDBManager(db_path=db_path)
    # lmdb_manager.second_db = lmdb_manager.env.open_db("train".encode('utf-8'))
    # lmdb_manager.hypter_db = lmdb_manager.env.open_db("val".encode('utf-8'))
    # lmdb_manager.clear_databases()
    for name in db_names:
        lmdb_manager.second_db = lmdb_manager.env.open_db(name.encode('utf-8'))
        keys = lmdb_manager.get_keys()
        for key in keys:
            value_json = lmdb_manager.read(key)
            value = json.loads(value_json)
            if value["action"] not in ["正卧（一级）", "正卧（二级）", "俯卧（一级）", "俯卧（二级）", "左侧卧（一级）", "左侧卧（二级）", "右侧卧（一级）", "右侧卧（二级）"]:
                val_data.append(key)
                data_dict[key] = name
                # continue
            else:
                val_data.append(key)
                data_dict[key] = name

    # num_keys = len(val_data) + len(train_data)
    # print("Total number of elements: ", num_keys)
    print("Number of elements in val: ", len(val_data))
    print("Number of elements in train: ", len(train_data))
    # Calculate the number of elements to be sent to "val" database
    # val_count = num_keys - int(num_keys * 0.3)
    
    
    # Move remaining elements to "val" database if count is not reached
    # if len(val_data) < val_count:
    #     remaining_count = val_count - len(val_data)
    #     for key in train_data:
    #         val_data.append(key)
    #         train_data.remove(key)
    #         remaining_count -= 1
    #         if remaining_count == 0:
    #             break
    
    # Write train_data to "train" database
    lmdb_manager.hypter_db = lmdb_manager.env.open_db("train".encode('utf-8'))
    db_name = "yuchengzhang"
    for key in train_data:
        if db_name != data_dict[key]:
            lmdb_manager.second_db = lmdb_manager.env.open_db(data_dict[key].encode('utf-8'))
            db_name = data_dict[key]
        value_json = lmdb_manager.read(key)
        lmdb_manager.add_data_to_db("hypter_db",key, value_json)
    
    # Write val_data to "val" database
    lmdb_manager.hypter_db = lmdb_manager.env.open_db("val".encode('utf-8'))
    db_name = "yuchengzhang"
    for key in val_data:
        if db_name != data_dict[key]:
            lmdb_manager.second_db = lmdb_manager.env.open_db(data_dict[key].encode('utf-8'))
            db_name = data_dict[key]
        value_json = lmdb_manager.read(key)
        lmdb_manager.add_data_to_db("hypter_db",key, value_json)
    
    lmdb_manager.env.close()




# split_databases(db_path="./LMDB/database", db_names=["baishuhang"])
run = wandb.init(project="RestNightAI", job_type="load-data")
raw_data = wandb.Artifact(
            "Onbed_data", type="dataset",
            description="在床检测数据集，包含余承璋与白书航两个人的数据。余承璋的正例作为训练集，白书航的正例和负例作为验证集。",
            metadata={"subject": ["yuchengzhang", "baishuhang"],
                      "type": "Onbed"})
raw_data.add_file("./LMDB/database.zip")
run.log_artifact(raw_data)
print("Done")