import subprocess
import json
import pickle
from collections import OrderedDict
from opensora.npu_config import npu_config

import sys
import os

class SuppressStdout:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SuppressStdout, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# 创建单例


class ObsConnection:
    """
    AK, SK, STS_TOKEN临时密钥有效时效云计算网站最长为24h
    buckets & object: https://uconsole.ccaicc.com/#/mgt/modelarts -> 对象控制台
    keys & tokens: https://uconsole.ccaicc.com/#/mgt/modelarts -> 对象控制台 -> 获取访问密匙(AK 和 SK)
    """
    def __init__(self):
        with open(f"{npu_config.work_path}/scripts/train_data/key.json", "r") as f:
            key = json.load(f)
        self.AK = key["AK"]
        self.SK = key["SK"]
        self.endpoint = key["EP"]
        self.bucket = "sora"
        self.suppress_stdout = SuppressStdout()
    
    def connect(self, obs):
        config_command = [
            obs, 'config',
            '-i=' + self.AK,
            '-k=' + self.SK,
            '-e=' + self.endpoint
        ]
        result = subprocess.run(config_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to configure obsutil: {result.stderr}")
        else:
            print("Successfully configured obsutil")

class VirtualDisk:
    """
    :param storage_dir: 内存虚拟磁盘的挂载点路径。
    :param size: 内存虚拟磁盘的大小，例如 '1G'。
    :param obs: linux 系统里面obs具体位置
    :param connection: 抽象出obs连接管理
    """
    def __init__(self, storage_dir, size="1G", obs="/home/opensora/obsutil_linux_arm64_5.5.12/obsutil"):
        self.obs = obs
        self.connection = ObsConnection()
        self.connection.connect(obs)
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_dir = storage_dir
        self.size = self._convert_size_to_bytes(size)
        if not self.is_tmpfs_mounted():
            self.create_ramdisk()
        else:
            print(f"{self.storage_dir} is already mounted as tmpfs.")
        self.index_file = os.path.join(self.storage_dir, 'index.pkl')
        self.index = self.load_index()
        self.lru = OrderedDict()
        self.current_size = self.get_total_storage_size()  # 初始化时计算总大小
    
    def _convert_size_to_bytes(self, size):
        unit = size[-1].upper()
        size_value = int(size[:-1])
        if unit == 'K':
            return size_value * 1024
        elif unit == 'M':
            return size_value * 1024 ** 2
        elif unit == 'G':
            return size_value * 1024 ** 3
        else:
            raise ValueError("Invalid size unit. Use K, M, or G.")

    """
    创建并挂载一个 tmpfs 类型的内存虚拟磁盘。
    """
    def create_ramdisk(self):
        try:
            # 如果挂载点目录不存在，创建它
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)     
            # 挂载 tmpfs 到挂载点
            subprocess.run(['sudo', 'mount', '-t', 'tmpfs', '-o', f'size={self.size}', 'tmpfs', self.storage_dir], check=True)
            print(f"Successfully mounted tmpfs on {self.storage_dir} with size {self.size}.")
        
        except subprocess.CalledProcessError as e:
            print(f"Failed to mount tmpfs: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def load_index(self):
        """
        加载索引文件。
        :return: 索引字典。
        """
        if os.path.exists(self.index_file):
            with open(self.index_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_index(self):
        """
        保存索引文件。
        """
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.index, f)
 
    """
    取消挂载内存虚拟磁盘。

    :param storage_dir: 内存虚拟磁盘的挂载点路径。
    """
    def unmount_ramdisk(self):
        try:
            # 确保没有进程在使用挂载点后取消挂载
            subprocess.run(['sudo', 'umount', self.storage_dir], check=True)
            print(f"Successfully unmounted tmpfs from {self.storage_dir}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to unmount tmpfs: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    """
    检查挂载点是否已经被挂载为 tmpfs。
    :param storage_dir: 挂载点路径。
    :return: 如果已挂载为 tmpfs，返回 True；否则返回 False。
    """
    def is_tmpfs_mounted(self):
        try:
            result = subprocess.run(['mountpoint', '-q', self.storage_dir], check=False)
            if result.returncode == 0:
                return True
            return False
        except Exception as e:
            print(f"An error occurred while checking if tmpfs is mounted: {e}")
            return False

    def get_data(self, key):
        """
        获取存储在本地磁盘上的数据。如果数据不存在，通过 obsutil 从远端获取并存储。
        :param key: 数据的唯一键。
        :return: 数据。
        """
        # if key in self.index:
        #     data_file = self.index[key]
        #     if os.path.exists(data_file):
        #         self.lru.move_to_end(key)
        #         with open(data_file, 'rb') as f:
        #             # print(f"Successfully get {key} from local")
        #             return pickle.load(f)


        # 如果数据不存在，使用 obsutil 从远端获取
        object_name = key  # 假设 key 对应于远端对象名称
        local_path = os.path.join(self.storage_dir, key)

        with self.connection.suppress_stdout:
            self.download_and_convert_to_pickle(self.connection.bucket, object_name, local_path)

        # 保存数据的位置
        # self.index[key] = local_path
        # self.save_index()
        # self.lru[key] = local_path
        #
        # file_size = os.path.getsize(local_path)
        # self.current_size += file_size

        # self.ensure_storage_limit()

        return local_path

    def del_data(self, local_path):
        os.remove(local_path)

    def download_and_convert_to_pickle(self, bucket, object_name, local_path):
        """
        使用 obsutil 从 OBS 下载文件并转换为 pickle 格式存储到本地路径。
        :param bucket: OBS 存储桶名称。
        :param object_name: OBS 中的对象名称。
        :param local_path: 本地文件路径。
        """
        # try:
            # 下载文件到local_path路径
        subprocess.run([self.obs, 'cp', f'obs://{bucket}/{object_name}', local_path], check=True)
            # print(f"Successfully downloaded obs://{bucket}/{object_name} to {local_path}.")

        # except subprocess.CalledProcessError as e:
        #     print(f"Failed to download obs://{bucket}/{object_name} to {local_path}: {e}")

    def ensure_storage_limit(self):
        """
        确保存储总大小不超过虚拟磁盘大小，超出时根据LRU策略删除最旧的文件。
        """
        while self.current_size > self.size:
            oldest_key, oldest_path = self.lru.popitem(last=False)
            file_size = os.path.getsize(oldest_path)
            os.remove(oldest_path)
            del self.index[oldest_key]
            self.save_index()
            print(f"Removed {oldest_key} to free up {file_size} bytes.")
            self.current_size -= file_size

    def get_total_storage_size(self):
        """
        获取当前所有存储文件的总大小。
        :return: 总大小（字节）。
        """
        total_size = 0
        for path in self.lru.values():
            if os.path.exists(path):
                total_size += os.path.getsize(path)
        return total_size