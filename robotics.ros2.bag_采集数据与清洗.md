---
id: mdf1km43txpt2m2yb2r7xsz
title: Bag_采集数据与清洗
desc: ''
updated: 1752243948974
created: 1752166751031
---

## 格式

### 目录结构

ROS2 的 bag 文件本质上就是 SQLite3 数据库。使用 ros2 bag record 命令录制数据时，ROS2 会创建一个文件夹（通常以 .db3 结尾）。比如：

```bash
my_bag/
├── metadata.yaml
└── rosbag2_2023_07_11-10_30_45/
    ├── metadata.yaml
    └── rosbag2_2023_07_11-10_30_45_0.db3
```

顶层 metadata.yaml 包含整个 bag 的元信息。数据库则包含实际内容。

### 数据库表结构

db3 文件的核心表如下：


topics 表：

| id  | name               | type                            | serialization_format |
| --- | ------------------ | ------------------------------- | -------------------- |
| 1   | /joint_states      | sensor_msgs/msg/JointState      | cdr                  |
| 2   | /camera/compressed | sensor_msgs/msg/CompressedImage | cdr                  |

2. messages 表存储所有消息内容：

| id  | topic_id | timestamp           | data (BLOB) |
| --- | -------- | ------------------- | ----------- |
| 1   | 1        | 1689053445000000000 | 0x...       |
| 2   | 2        | 1689053445001000000 | 0x...       |

3. schema 表（可选）。存储消息类型的定义（ROS IDL）。

任何 SQLite 工具（如 DB Browser for SQLite）都可直接查看数据。

### Python 脚本访问 db

直接访问数据库处理：

```py
import sqlite3
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# 连接到 bag 数据库
conn = sqlite3.connect('rosbag2_2023_07_11-10_30_45_0.db3')
cursor = conn.cursor()

# 获取关节状态话题的元数据
cursor.execute("SELECT id, type FROM topics WHERE name = '/joint_states'")
topic_id, msg_type = cursor.fetchone()
msg_class = get_message(msg_type)

# 查询所有关节状态消息
cursor.execute("""
    SELECT timestamp, data 
    FROM messages 
    WHERE topic_id = ? 
    ORDER BY timestamp
""", (topic_id,))

# 反序列化并处理数据
for timestamp, blob_data in cursor.fetchall():
    joint_state = deserialize_message(blob_data, msg_class)
    print(f"时间: {timestamp}, 位置: {joint_state.position}")
    
conn.close()
```

## Recording and playing back data

`ros2 bag` 记录系统的数据，主要涉及发布的话题、服务和动作。使用如下：

```bash
ros2 bag record {{topic_name1, topic_name2...}} # 单个或多个话题
```

还可以指定参数：
- `-a` 记录系统所有话题。
- `-o` 选项指定 bag 文件名；

查看文件细节：

```bash
ros2 bag info {{bag_file_name}}
```

通常可以看到，由 sqlite3 保存内容。


重播：

```bash
ros2 bag play {{bag_file_name}}
```

## 将 bag 清洗到 numpy

流程通常如下：
1. **数据采集**：
   - 使用  `ros2 bag record`  命令或编程方式（通过  `rosbag2`   API ）记录所需的话题。
   - 确保记录的话题和消息类型正确，并且有足够的数据量。
2. **数据转换**：
   - 将 bag 文件转换为适合机器学习处理的格式（如 NumPy 数组、 HDF 5、 TFRecord 等）。
   - 对于图像数据，通常需要解码（如将 CompressedImage 转换为 OpenCV 图像）。
3. **数据清洗**：
   - 去除无效数据（如 NaN 值、异常值）。
   - 处理缺失值（如插值或丢弃）。
   - 对于时间序列数据，可能需要进行时间对齐（因为不同话题的消息时间戳可能不完全同步）。
4. **数据预处理**：
   - 图像数据：缩放、归一化、增强（旋转、裁剪等）。
   - 关节状态数据：归一化、差分处理（获取速度、加速度）等。
5. **数据集构建**：
   - 将数据划分为训练集、验证集和测试集。
   - 创建 PyTorch 的 `Dataset` 和 `DataLoader` 类，以便高效地加载和迭代数据。

比如，我们收集到了关节角数据，格式为 JointState ；收集到相机数据 CompressedImage ，我们可以清洗。使用 `rosbag2_py` 来读取 bag 文件，然后分别处理这两种消息。rosbag2_py 通常随 ROS2 SDK 安装，或者 `sudo apt install -y rosbag2`

### 编写 bag 记录的采集节点

```bash
sudo apt install python3-rosbag2-py python3-sensor-msgs-py
pip install numpy opencv-python-headless
```

```py
# 自定义采集脚本 (collect_data.py)
import rclpy
from rclpy.node import Node
from rosbag2_py import Recorder, StorageOptions, RecordOptions
from sensor_msgs.msg import JointState, CompressedImage
import time

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        # 配置存储参数
        storage_options = StorageOptions(
            uri='robot_data',
            storage_id='sqlite3'
        )
        record_options = RecordOptions()
        record_options.all = False
        record_options.topics = ['/joint_states', '/camera/compressed']
        
        # 创建记录器
        self.recorder = Recorder()
        self.recorder.init(storage_options, record_options)
        
        # 状态监控
        self.get_logger().info("开始数据采集...")
        self.start_time = time.time()
        
    def stop_recording(self):
        self.recorder.shutdown()
        duration = time.time() - self.start_time
        self.get_logger().info(f"采集完成! 时长: {duration:.2f}秒")

def main(args=None):
    rclpy.init(args=args)
    collector = DataCollector()
    
    try:
        # 运行采集（Ctrl+C停止）
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.stop_recording()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

数据转换：

```py
# advanced_converter.py
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import numpy as np
import cv2
import pandas as pd
import h5py
import os
from tqdm import tqdm
from scipy import interpolate

class BagConverter:
    def __init__(self, bag_path, output_dir):
        self.bag_path = bag_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建HDF5文件
        self.h5_file = h5py.File(os.path.join(output_dir, 'robot_dataset.h5'), 'w')
        self.image_group = self.h5_file.create_group('images')
        self.joint_group = self.h5_file.create_group('joints')
        
        # 元数据
        self.metadata = {
            'joint_topics': [],
            'image_topics': [],
            'num_samples': 0
        }
    
    def process_bag(self):
        # 连接数据库
        conn = sqlite3.connect(os.path.join(self.bag_path, 'rosbag2.db'))
        cursor = conn.cursor()
        
        # 获取所有话题
        cursor.execute("SELECT DISTINCT topic_name FROM messages")
        topics = [row[0] for row in cursor.fetchall()]
        
        # 分离关节和图像话题
        joint_topics = [t for t in topics if 'joint' in t.lower()]
        image_topics = [t for t in topics if 'image' in t.lower() or 'camera' in t.lower()]
        
        # 处理关节数据
        joint_data = {}
        for topic in tqdm(joint_topics, desc="处理关节数据"):
            cursor.execute("SELECT timestamp, data FROM messages WHERE topic_name = ? ORDER BY timestamp", (topic,))
            timestamps, positions = [], []
            
            for row in cursor.fetchall():
                timestamp, blob_data = row
                msg = deserialize_message(blob_data, get_message('sensor_msgs/msg/JointState'))
                timestamps.append(timestamp)
                positions.append(msg.position)
            
            # 转换为DataFrame
            df = pd.DataFrame(positions, index=pd.to_datetime(timestamps, unit='ns'))
            joint_data[topic] = df
            
            # 保存到HDF5
            self.joint_group.create_dataset(topic.replace('/', '_'), data=df.values)
        
        # 处理图像数据
        image_data = {}
        for topic in tqdm(image_topics, desc="处理图像数据"):
            cursor.execute("SELECT timestamp, data FROM messages WHERE topic_name = ? ORDER BY timestamp", (topic,))
            timestamps, images = [], []
            
            for row in cursor.fetchall():
                timestamp, blob_data = row
                msg = deserialize_message(blob_data, get_message('sensor_msgs/msg/CompressedImage'))
                
                # 转换为numpy数组
                np_arr = np.frombuffer(msg.data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    timestamps.append(timestamp)
                    images.append(img)
            
            # 保存到HDF5
            image_array = np.stack(images)
            self.image_group.create_dataset(topic.replace('/', '_'), data=image_array, 
                                          compression="gzip", compression_opts=9)
            
            image_data[topic] = {
                'timestamps': timestamps,
                'images': image_array
            }
        
        conn.close()
        
        # 时间同步与对齐
        self.align_data(joint_data, image_data)
        
        # 保存元数据
        self.metadata['joint_topics'] = list(joint_data.keys())
        self.metadata['image_topics'] = list(image_data.keys())
        self.metadata['num_samples'] = min(len(v) for v in joint_data.values())
        
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f)
        
        self.h5_file.close()
    
    def align_data(self, joint_data, image_data):
        """使用插值对齐不同频率的数据"""
        # 创建统一时间轴
        all_timestamps = []
        for data in joint_data.values():
            all_timestamps.extend(data.index.values)
        for data in image_data.values():
            all_timestamps.extend(data['timestamps'])
        
        master_timeline = np.unique(np.sort(all_timestamps))
        
        # 对齐关节数据
        aligned_joints = {}
        for topic, df in joint_data.items():
            f = interpolate.interp1d(
                df.index.astype(np.int64), 
                df.values.T, 
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            aligned = f(master_timeline.astype(np.int64))
            aligned_joints[topic] = aligned.T
        
        # 对齐图像数据 (需要特殊处理)
        aligned_images = {}
        for topic, data in image_data.items():
            # 使用最近邻插值
            indices = np.searchsorted(
                np.array(data['timestamps']), 
                master_timeline, 
                side="left"
            )
            # 确保不越界
            indices = np.clip(indices, 0, len(data['images'])-1)
            aligned_images[topic] = data['images'][indices]
        
        # 保存对齐后的数据
        aligned_group = self.h5_file.create_group('aligned')
        for topic, data in aligned_joints.items():
            aligned_group.create_dataset(f"joints/{topic}", data=data)
        
        for topic, data in aligned_images.items():
            aligned_group.create_dataset(f"images/{topic}", data=data, 
                                       compression="gzip", compression_opts=9)

if __name__ == "__main__":
    converter = BagConverter(
        bag_path="/path/to/your/bag",
        output_dir="processed_data"
    )
    converter.process_bag()
```

### 数据清洗

移除异常值

```py
def remove_outliers(data, std_threshold=3):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    mask = np.all(np.abs(data - mean) < std_threshold * std, axis=1)
    return data[mask]
```

缺失值处理：

```py
def interpolate_missing(data):
    # 线性插值
    df = pd.DataFrame(data)
    df.interpolate(method='linear', inplace=True)
    df.bfill(inplace=True)  # 后向填充
    df.ffill(inplace=True)  # 前向填充
    return df.values
```

图像预处理：

```py
def preprocess_image(img):
    # 1. 去畸变
    # 2. ROI裁剪
    # 3. 尺寸调整
    # 4. 归一化
    img = cv2.undistort(img, camera_matrix, dist_coeffs)
    img = img[y:y+h, x:x+w]  # ROI裁剪
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img
```

### 使用 zarr 的版本

```py
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import numpy as np
import cv2
import pandas as pd
import zarr
import os
from tqdm import tqdm
from scipy import interpolate
import json
import sqlite3
import time
from numcodecs import Blosc

class BagToZarrConverter:
    def __init__(self, bag_path, output_dir, chunk_size=100):
        self.bag_path = bag_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建 Zarr 存储
        self.store = zarr.DirectoryStore(os.path.join(output_dir, 'robot_dataset.zarr'))
        self.root = zarr.group(store=self.store, overwrite=True)
        
        # 创建元数据存储
        self.metadata = {
            'bag_path': bag_path,
            'creation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'joint_topics': [],
            'image_topics': [],
            'num_samples': 0,
            'chunk_size': chunk_size
        }
        
        # 压缩配置
        self.compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    
    def process_bag(self):
        """处理 ROS2 bag 文件并转换为 Zarr 格式"""
        # 连接数据库
        conn = sqlite3.connect(os.path.join(self.bag_path, 'rosbag2.db'))
        cursor = conn.cursor()
        
        # 获取所有话题
        cursor.execute("SELECT DISTINCT topic_name FROM messages")
        topics = [row[0] for row in cursor.fetchall()]
        
        # 分离关节和图像话题
        joint_topics = [t for t in topics if 'joint' in t.lower()]
        image_topics = [t for t in topics if 'image' in t.lower() or 'camera' in t.lower()]
        
        # 处理关节数据
        joint_data = {}
        for topic in tqdm(joint_topics, desc="处理关节数据"):
            cursor.execute("SELECT timestamp, data FROM messages WHERE topic_name = ? ORDER BY timestamp", (topic,))
            rows = cursor.fetchall()
            
            if not rows:
                continue
                
            # 解析第一条消息确定关节数量
            _, first_blob = rows[0]
            first_msg = deserialize_message(first_blob, get_message('sensor_msgs/msg/JointState'))
            num_joints = len(first_msg.position)
            
            # 创建 Zarr 数组
            arr = self.root.zeros(
                f'raw/joints/{topic.replace("/", "_")}',
                shape=(len(rows), num_joints),
                chunks=(self.chunk_size, None),
                dtype='float32',
                compressor=self.compressor
            )
            
            # 存储时间戳
            timestamps = self.root.zeros(
                f'raw/timestamps/joints/{topic.replace("/", "_")}',
                shape=(len(rows),),
                chunks=(self.chunk_size,),
                dtype='int64'
            )
            
            # 填充数据
            positions = []
            ts_list = []
            for i, row in enumerate(rows):
                timestamp, blob_data = row
                msg = deserialize_message(blob_data, get_message('sensor_msgs/msg/JointState'))
                
                # 确保关节数量一致
                if len(msg.position) != num_joints:
                    print(f"警告: 话题 {topic} 在索引 {i} 处关节数不一致")
                    # 使用最后一次有效位置填充
                    if positions:
                        positions.append(positions[-1])
                    else:
                        positions.append([0.0] * num_joints)
                else:
                    positions.append(msg.position)
                
                ts_list.append(timestamp)
            
            arr[:] = np.array(positions, dtype='float32')
            timestamps[:] = np.array(ts_list, dtype='int64')
            joint_data[topic] = arr
            self.metadata['joint_topics'].append(topic)
        
        # 处理图像数据
        image_data = {}
        for topic in tqdm(image_topics, desc="处理图像数据"):
            cursor.execute("SELECT timestamp, data FROM messages WHERE topic_name = ? ORDER BY timestamp", (topic,))
            rows = cursor.fetchall()
            
            if not rows:
                continue
                
            # 解析第一条消息确定图像尺寸
            _, first_blob = rows[0]
            first_msg = deserialize_message(first_blob, get_message('sensor_msgs/msg/CompressedImage'))
            np_arr = np.frombuffer(first_msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"无法解码话题 {topic} 的第一张图像，跳过")
                continue
                
            height, width, channels = img.shape
            
            # 创建 Zarr 数组
            arr = self.root.zeros(
                f'raw/images/{topic.replace("/", "_")}',
                shape=(len(rows), height, width, channels),
                chunks=(self.chunk_size, None, None, None),
                dtype='uint8',
                compressor=self.compressor
            )
            
            # 存储时间戳
            timestamps = self.root.zeros(
                f'raw/timestamps/images/{topic.replace("/", "_")}',
                shape=(len(rows),),
                chunks=(self.chunk_size,),
                dtype='int64'
            )
            
            # 填充数据
            ts_list = []
            for i, row in enumerate(rows):
                timestamp, blob_data = row
                msg = deserialize_message(blob_data, get_message('sensor_msgs/msg/CompressedImage'))
                
                np_arr = np.frombuffer(msg.data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f"无法解码话题 {topic} 在索引 {i} 处的图像")
                    # 使用最后一张有效图像
                    if i > 0:
                        arr[i] = arr[i-1]
                    else:
                        # 创建黑色图像
                        arr[i] = np.zeros((height, width, channels), dtype='uint8')
                else:
                    # 检查尺寸是否一致
                    if img.shape != (height, width, channels):
                        img = cv2.resize(img, (width, height))
                    arr[i] = img
                
                ts_list.append(timestamp)
            
            timestamps[:] = np.array(ts_list, dtype='int64')
            image_data[topic] = arr
            self.metadata['image_topics'].append(topic)
        
        conn.close()
        
        # 时间同步与对齐
        self.align_data(joint_data, image_data)
        
        # 保存元数据
        self.metadata['num_samples'] = self.root['aligned/joints'].shape[0]
        self.root.attrs['metadata'] = json.dumps(self.metadata)
        
        # 保存为单独文件
        with open(os.path.join(self.output_dir, 'dataset_metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"转换完成! 总样本数: {self.metadata['num_samples']}")
    
    def align_data(self, joint_data, image_data):
        """使用插值对齐不同频率的数据"""
        # 创建统一时间轴
        all_timestamps = []
        
        # 收集所有时间戳
        for topic, arr in joint_data.items():
            ts_arr = self.root[f'raw/timestamps/joints/{topic.replace("/", "_")}']
            all_timestamps.extend(ts_arr[:])
        
        for topic, arr in image_data.items():
            ts_arr = self.root[f'raw/timestamps/images/{topic.replace("/", "_")}']
            all_timestamps.extend(ts_arr[:])
        
        # 创建主时间轴
        all_timestamps = np.array(all_timestamps)
        master_timeline = np.unique(np.sort(all_timestamps))
        
        # 存储主时间轴
        self.root.zeros(
            'aligned/timestamps',
            shape=master_timeline.shape,
            chunks=(self.chunk_size,),
            dtype='int64'
        )[:] = master_timeline
        
        # 对齐关节数据
        aligned_joints = {}
        for topic, arr in joint_data.items():
            raw_timestamps = self.root[f'raw/timestamps/joints/{topic.replace("/", "_")}'][:]
            raw_data = arr[:]
            
            # 创建插值函数
            f = interpolate.interp1d(
                raw_timestamps.astype(np.int64), 
                raw_data.T, 
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            # 执行插值
            aligned_data = f(master_timeline.astype(np.int64)).T
            
            # 创建对齐后的数组
            aligned_arr = self.root.zeros(
                f'aligned/joints/{topic.replace("/", "_")}',
                shape=aligned_data.shape,
                chunks=(self.chunk_size, None),
                dtype='float32',
                compressor=self.compressor
            )
            aligned_arr[:] = aligned_data
            aligned_joints[topic] = aligned_arr
        
        # 对齐图像数据
        aligned_images = {}
        for topic, arr in image_data.items():
            raw_timestamps = self.root[f'raw/timestamps/images/{topic.replace("/", "_")}'][:]
            raw_data = arr[:]
            
            # 使用最近邻插值
            indices = np.searchsorted(raw_timestamps, master_timeline, side="left")
            
            # 处理边界情况
            indices = np.where(indices >= len(raw_data), len(raw_data) - 1, indices)
            indices = np.where(indices < 0, 0, indices)
            
            # 创建对齐后的数组
            aligned_arr = self.root.zeros(
                f'aligned/images/{topic.replace("/", "_")}',
                shape=(len(master_timeline),) + raw_data.shape[1:],
                chunks=(self.chunk_size, None, None, None),
                dtype='uint8',
                compressor=self.compressor
            )
            
            # 分块填充数据以减少内存使用
            for i in range(0, len(master_timeline), self.chunk_size):
                end = min(i + self.chunk_size, len(master_timeline))
                chunk_indices = indices[i:end]
                aligned_arr[i:end] = raw_data[chunk_indices]
            
            aligned_images[topic] = aligned_arr
        
        # 添加对齐标记
        self.root.attrs['aligned'] = True
        self.root.attrs['master_timeline_length'] = len(master_timeline)
        
        return aligned_joints, aligned_images

    @staticmethod
    def create_dataset_from_bags(bag_paths, output_dir, chunk_size=100):
        """从多个bag文件创建单个Zarr数据集"""
        # 创建主数据集
        main_store = zarr.DirectoryStore(os.path.join(output_dir, 'combined_dataset.zarr'))
        main_root = zarr.group(store=main_store, overwrite=True)
        
        # 初始化元数据
        combined_metadata = {
            'bag_paths': bag_paths,
            'creation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'chunk_size': chunk_size,
            'datasets': []
        }
        
        # 处理每个bag文件
        for i, bag_path in enumerate(bag_paths):
            print(f"处理bag {i+1}/{len(bag_paths)}: {bag_path}")
            converter = BagToZarrConverter(
                bag_path=bag_path,
                output_dir=os.path.join(output_dir, f"temp_{i}"),
                chunk_size=chunk_size
            )
            converter.process_bag()
            
            # 将临时数据集追加到主数据集
            temp_store = zarr.DirectoryStore(os.path.join(output_dir, f"temp_{i}", 'robot_dataset.zarr'))
            temp_root = zarr.open(store=temp_store, mode='r')
            
            # 复制对齐后的数据
            for group in ['aligned/joints', 'aligned/images', 'aligned/timestamps']:
                if group not in temp_root:
                    continue
                
                # 递归复制
                source = temp_root[group]
                dest_path = f"bag_{i}/{group}"
                
                if group not in main_root:
                    main_root.create_group(os.path.dirname(dest_path))
                
                zarr.copy(source, main_root, dest_path)
            
            # 保存元数据
            combined_metadata['datasets'].append({
                'bag_path': bag_path,
                'num_samples': temp_root.attrs.get('num_samples', 0)
            })
            
            # 清理临时文件
            # shutil.rmtree(os.path.join(output_dir, f"temp_{i}"))
        
        # 保存组合元数据
        main_root.attrs['metadata'] = json.dumps(combined_metadata)
        with open(os.path.join(output_dir, 'combined_metadata.json'), 'w') as f:
            json.dump(combined_metadata, f, indent=2)
        
        print(f"组合数据集创建完成! 总样本数: {sum(d['num_samples'] for d in combined_metadata['datasets'])}")
        return main_root

if __name__ == "__main__":
    # 单个bag文件转换
    converter = BagToZarrConverter(
        bag_path="/path/to/your/rosbag",
        output_dir="zarr_dataset",
        chunk_size=128  # 根据数据集大小调整
    )
    converter.process_bag()
    
    # 多个bag文件合并
    # BagToZarrConverter.create_dataset_from_bags(
    #     bag_paths=[
    #         "/path/to/bag1",
    #         "/path/to/bag2",
    #         "/path/to/bag3"
    #     ],
    #     output_dir="combined_zarr_dataset"
    # )
```

## Ref and Tag

[Recording and playing back data](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html)

[rosbag2](https://github.com/ros2/rosbag2)

[Recording a bag from a node (Python)](https://docs.ros.org/en/humble/Tutorials/Advanced/Recording-A-Bag-From-Your-Own-Node-Py.html#)