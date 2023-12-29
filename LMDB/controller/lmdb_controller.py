# 导入所需库
import lmdb
import multiprocessing
import threading
import time
import json
import threading
import pickle
import datetime


# 创建 LMDBManager 类，继承 multiprocessing.Process 
class LMDBManager(multiprocessing.Process):
   
    # 初始化函数
    def __init__(self, db_path, map_size=200*1024*1024*1024, max_dbs=10, sync_interval=5*60):
        # 调用父类的构造函数
        super(LMDBManager, self).__init__()

        # LMDB 数据库路径
        self.db_path = db_path 

        # LMDB 数据库最大存储空间
        self.map_size = map_size 

        # 最大数据库数量
        self.max_dbs = max_dbs 

        # 创建进程间通信队列
        self.write_queue = multiprocessing.Queue() 

        # 实例化 LMDB 环境
        self.env = lmdb.open(self.db_path, map_size=self.map_size, max_dbs=self.max_dbs) 

        # 创建并打开命名为 'pressure' 的数据库
        self.second_db = self.env.open_db('pressure'.encode('utf-8')) 

        # 创建并打开命名为 'hypter' 的数据库
        self.hypter_db = self.env.open_db('hypter'.encode('utf-8')) 

        # 创建进程停止事件
        self.stop_event = multiprocessing.Event() 

        # 用于通知 LMDB 线程停止的事件
        self.flag = multiprocessing.Event() 

        # 创建共享列表用于存储数据包 ID
        self.manager = multiprocessing.Manager()
        self.second_list = self.manager.list(self.get_keys())

        # 创建 dict 用于存储每 20 帧的数据包
        self.record_dict = {} 

        # 数据同步间隔，单位为秒
        self.sync_interval = sync_interval  

        # 同步定时器
        self.sync_timer = None  

    # 创建 LMDB 线程
    def create(self):
        self.lmdb_thread = multiprocessing.Process(target=self.start)
        self.lmdb_thread.start()
        return self.write_queue

    # 开始 LMDB 线程
    def start(self):
        print("Starting LMDB thread")
        self.start_sync_timer()

        # 在接收到停止标志前，持续从队列中读取并写入数据
        while not self.flag.is_set():
            if not self.write_queue.empty():
                key, value_json = self.write_queue.get()
                with self.env.begin(db=self.second_db, write=True) as txn:
                    txn.put(key.encode('utf-8'), value_json)
                    self.second_list.append(key)
                    self.process_record_dict()

            time.sleep(0.1)  # 防止忙等待，添加小延迟
            if self.write_queue.empty() and self.stop_event.is_set():
                self.flag.set()

            self.process_record_dict()

    # 读取键值为 key 的数据
    def read(self, key):
        with self.env.begin(write=False) as txn:
            value = txn.get(key.encode('utf-8'))
            if value is not None:
                return value.decode('utf-8')
        return None

    # 获取数据库中所有键
    def get_keys(self):
        with self.env.begin(db=self.second_db, write=False) as txn:
            return [key.decode('utf-8') for key, _ in txn.cursor()]

    # 停止 LMDB 线程函数
    def stop(self):
        while not self.write_queue.empty():
            time.sleep(0.1)

        self.stop_sync_timer()
        self.stop_event.set()

        if self.lmdb_thread.is_alive():
            self.lmdb_thread.join()

        if self.env:
            self.env.close()

    # 处理 record_dict ，每次插入后检查当前长度是否大于等于 5，并检查时间差是否小于 7，满足条件则插入 record_dict
    def process_record_dict(self):
        if len(self.second_list) >= 5:
            last_five_seconds = self.second_list[-5:]
            first_time = datetime.datetime.strptime(last_five_seconds[0], "%Y-%m-%dT%H:%M:%S")
            last_time = datetime.datetime.strptime(last_five_seconds[0], "%Y-%m-%dT%H:%M:%S")
            time_diff = (last_time - first_time).total_seconds()
            if time_diff < 7:
                self.record_dict[last_five_seconds[0]] = last_five_seconds

    # 开启数据同步定时器
    def start_sync_timer(self):
        self.sync_timer = threading.Timer(self.sync_interval, self.sync_to_db)
        self.sync_timer.start()

    # 停止数据同步定时器
    def stop_sync_timer(self):
        if self.sync_timer is not None:
            self.sync_timer.cancel()
            self.sync_timer = None

    # 同步数据到 LMDB 数据库
    def sync_to_db(self):
        with self.env.begin(db=self.hypter_db,write=True) as txn:
            txn.put('second_list'.encode('utf-8'), pickle.dumps(self.second_list))
            txn.put('record_dict'.encode('utf-8'), pickle.dumps(self.record_dict))

        self.start_sync_timer()  # 完成同步后重新启动定时器


    def get_second_list_length(self):
        return len(self.second_list)
        
    # 清空 pressure 和 hyper 两个数据库
    def clear_databases(self):
        with self.env.begin(write=True) as txn:
            txn.drop(self.second_db, delete=False)
            txn.drop(self.hypter_db, delete=False)
