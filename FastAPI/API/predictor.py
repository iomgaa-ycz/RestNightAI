from FastAPI.Model.Onbed_model import *
from FastAPI.Model.Pose_model import *
from FastAPI.Model.Action_model import *
from FastAPI.Utils.load_json import *
import numpy as np
import torch

def predictor(arg, pressure_datas, ID, write_queue,data_collect,model_Onbed):
    data_collect[4:20, :, :] = data_collect[0:16, :, :]
    data_collect[0:4, :, :] = pressure_datas

    total_sum = pressure_datas.sum()
    variance = pressure_datas.var()
    
    if total_sum < arg["threshold_UnBed"]:
        return "未在床上",data_collect
    
    pressure_datas = torch.tensor(pressure_datas)
    pressure_datas = pressure_datas.unsqueeze(0)
    
    Onbed = model_Onbed(pressure_datas.cuda())
    
    if Onbed < arg["threshold_Onbed"]:
        return "未检测到人体",data_collect
    
    if variance < arg["threshold_Pose"]:
        Pose = model_Pose(pressure_datas.cuda())
        #获得Pose的最大索引
        Pose = Pose.argmax()
        if Pose == 0:
            return "正卧",data_collect
        elif Pose == 1:
            return "俯卧",data_collect
        elif Pose == 2:
            return "左侧卧",data_collect
        elif Pose == 3:
            return "右侧卧",data_collect
    else:   
        data_collect = torch.tensor(data_collect.cuda())
        Action = model_Action(data_collect)
        #获得Action的最大索引
        Action = Action.argmax()
        if Action == 0:
            return "一级体动",data_collect
        elif Action == 1:
            return "二级体动",data_collect
        elif Action == 2:
            return "三级体动",data_collect
