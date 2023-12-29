import json
from datetime import datetime,timedelta

# FILEPATH: /home/iomgaa/RestfulNightAI/FastAPI/API/collector.py
def collector(arg, pressure_datas, ID, write_queue, collect_label, buffer_pool):

    # 将ID与pressure_datas放入缓冲池
    buffer_pool[ID] = pressure_datas.tolist()

    # 检查collect_label是否为空
    if len(collect_label) == 0:
        return buffer_pool

    # 遍历缓冲池
    for key in list(buffer_pool.keys()):
        for label_key in list(collect_label.keys()):
            # 检查缓冲池中的元素是否满足条件
            key_time = datetime.strptime(key, "%Y-%m-%dT%H:%M:%S")
            if collect_label[label_key].end_time != None and collect_label[label_key].begin_time < key_time < collect_label[label_key].end_time :
                # 构建json对象
                json_data = {
                    'ID': ID,
                    'action': collect_label[label_key].action,
                    'data': buffer_pool[key]
                }
                # 执行write_queue.put
                write_queue.put((key, json.dumps(json_data)))
                # 从缓冲池中删除这条元素
                print(len(buffer_pool))
                del buffer_pool[key]
    return buffer_pool

