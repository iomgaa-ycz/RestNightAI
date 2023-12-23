import lmdb
import multiprocessing
import threading
import time
import json

class LMDBManager(multiprocessing.Process):
    def __init__(self, db_path, map_size=200*1024*1024*1024, max_dbs=10):
        super(LMDBManager, self).__init__()
        self.db_path = db_path
        self.map_size = map_size
        self.max_dbs = max_dbs
        self.write_queue = multiprocessing.Queue()
        self.env = lmdb.open(self.db_path, map_size=self.map_size, max_dbs=self.max_dbs)
        self.stop_event = multiprocessing.Event()

    def create(self):
        # Set up the LMDB environment
        self.lmdb_thread = multiprocessing.Process(target=self.start)
        self.lmdb_thread.start()
        return self.write_queue

    def start(self):
        print("Starting LMDB thread")
        while not self.stop_event.is_set():
            if not self.write_queue.empty():
                key, value_json = self.write_queue.get()
                with self.env.begin(write=True) as txn:
                    txn.put(key.encode('utf-8'), value_json.encode('utf-8'))
            time.sleep(0.1)  # Small delay to prevent busy waiting

    def read(self, key):
        with self.env.begin(write=False) as txn:
            value = txn.get(key.encode('utf-8'))
            if value is not None:
                return value.decode('utf-8')
        return None

    def get_keys(self):
        with self.env.begin(write=False) as txn:
            return [key.decode('utf-8') for key, _ in txn.cursor()]

    def stop(self):
        # 等待所有写操作完成
        while not self.write_queue.empty():
            time.sleep(0.1)

        # 设置事件通知其他进程停止
        self.stop_event.set()

        # 等待lmdb_thread进程结束
        if self.lmdb_thread.is_alive():
            self.lmdb_thread.join()

        # 关闭LMDB环境
        if self.env:
            self.env.close()
