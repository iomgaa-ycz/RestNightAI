from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from datetime import datetime

class PressureDataset(Dataset):
    def __init__(self, lmdb_manager, keys=None, phase = "train",db_name="yuchengzhang"):
        self.lmdb_manager = lmdb_manager
        self.lmdb_manager.second_db = self.lmdb_manager.env.open_db(db_name.encode('utf-8')) #设置数据库
        if keys is None:
            self.keys = self.lmdb_manager.get_keys()
        else:
            self.keys = keys
        self.phase = phase

    def __getitem__(self, index):
        key = self.keys[index]
        value_json = self.lmdb_manager.read(key)
        value = json.loads(value_json)
        x = value["data"]
        x = np.array(x)
        n,w,h = x.shape
        x = x.reshape(n, 1, w, h)

        # x归一化
        min = np.min(x)
        max = np.max(x)
        x = (x - min) / (max - min)
        x = x.astype(np.float32)

        # x中为0的元素变为-0.0001
        x[x == 0] = -0.001

        if self.phase == "train":
            # 添加噪音
            noise = np.random.normal(0, 0.001, x.shape)
            x = x + noise

            # 在 w 和 h 维度上进行最多正负10的移动
            shift_w = np.random.randint(-20, 21)  # 随机选择从-10到10的整数
            shift_h = np.random.randint(-20, 21)  # 随机选择从-10到10的整数
            x = np.roll(x, shift_w, axis=2)  # 在 w 维度上移动
            x = np.roll(x, shift_h, axis=3)  # 在 h 维度上移动

        pose = value["pose"]
        action = value["action"]
        OnBed = value["OnBed"]
        y = {
            "pose": pose,
            "action": action,
            "OnBed": OnBed
        }
        return x,y

    def __len__(self):
        return len(self.keys)