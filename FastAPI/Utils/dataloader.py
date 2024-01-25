from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from datetime import datetime

class PressureDataset(Dataset):
    def __init__(self, lmdb_manager, keys=None, phase = "train",db_name="yuchengzhang", mode="Onbed"):
        self.lmdb_manager = lmdb_manager
        self.db_name = db_name
        self.lmdb_manager.second_db = self.lmdb_manager.env.open_db(db_name.encode('utf-8'))
        if keys is None:
            self.keys = self.lmdb_manager.get_keys()
        else:
            self.keys = keys
        self.phase = phase
        self.mode = mode

    def __getitem__(self, index):
        key = self.keys[index]
        value_json = self.lmdb_manager.read(key)
        value = json.loads(value_json)
        x = value["data"]
        x = np.array(x)
        n,w,h = x.shape
        x = x[0,:,:]
        x = x.reshape(1,w,h)

        # x归一化
        x = x.astype(np.float32)

        # x中为0的元素变为-0.0001
        x[x == 0] = -0.001

        if self.phase == "train":
            # 添加噪音
            # noise = np.random.normal(0, 0.001, x.shape)
            # x = x + noise

            # 在 w 和 h 维度上进行最多正负10的移动
            shift_w = np.random.randint(-2, 2)  # 随机选择从-10到10的整数
            shift_h = np.random.randint(-2, 2)  # 随机选择从-10到10的整数
            x = np.roll(x, shift_w, axis=0)  # 在 w 维度上移动
            x = np.roll(x, shift_h, axis=1)  # 在 h 维度上移动
        

        if self.mode == "Onbed":
            if value["action"] in ["正卧（一级）", "正卧（二级）", "俯卧（一级）", "俯卧（二级）", "左侧卧（一级）", "左侧卧（二级）", "右侧卧（一级）", "右侧卧（二级）"]:
                pose = 1
            else:
                pose = 0
        elif self.mode == "Pose":
            if value["action"] in ["正卧（一级）", "正卧（二级）"]:
                pose = 0
            elif value["action"] in ["俯卧（一级）", "俯卧（二级）"]:
                pose = 1
            elif value["action"] in ["左侧卧（一级）", "左侧卧（二级）"]:
                pose = 2
            elif value["action"] in ["右侧卧（一级）", "右侧卧（二级）"]:
                pose = 3

        return x,pose

    def __len__(self):
        return len(self.keys)