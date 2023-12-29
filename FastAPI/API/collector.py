def collector(arg, pressure_datas, ID, write_queue):
    #向write_queue写入数据
    write_queue.put((ID,pressure_datas.tobytes()))

