# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import numpy as np
from cv2 import transform
from httpx import get
from torch.utils.data import DataLoader, Dataset


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC,
    )
    return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      uniform_dequantization: If `True`, add uniform dequantization to images.
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      train_ds, eval_ds, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = (
        config.training.batch_size if not evaluation else config.eval.batch_size
    )
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch sizes ({batch_size} must be divided by"
            f"the number of devices ({jax.device_count()})"
        )

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1

    # Create dataset builders for each dataset.
    if config.data.dataset == "CIFAR10":
        dataset_builder = tfds.builder("cifar10")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            # Added to train grayscale models
            # img = tf.image.rgb_to_grayscale(img)
            return tf.image.resize(
                img, [config.data.image_size, config.data.image_size], antialias=True
            )

    elif config.data.dataset == "SVHN":
        dataset_builder = tfds.builder("svhn_cropped")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(
                img, [config.data.image_size, config.data.image_size], antialias=True
            )

    elif config.data.dataset == "CELEBA":
        dataset_builder = tfds.builder("celeb_a")
        train_split_name = "train"
        eval_split_name = "validation"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = central_crop(img, 140)
            img = resize_small(img, config.data.image_size)
            return img

    elif config.data.dataset == "LSUN":
        dataset_builder = tfds.builder(f"lsun/{config.data.category}")
        train_split_name = "train"
        eval_split_name = "validation"

        if config.data.image_size == 128:

            def resize_op(img):
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = resize_small(img, config.data.image_size)
                img = central_crop(img, config.data.image_size)
                return img

        else:

            def resize_op(img):
                img = crop_resize(img, config.data.image_size)
                img = tf.image.convert_image_dtype(img, tf.float32)
                return img

    elif config.data.dataset in ["FFHQ", "CelebAHQ"]:
        dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
        train_split_name = eval_split_name = "train"

    else:
        raise NotImplementedError(f"Dataset {config.data.dataset} not yet supported.")

    # Customize preprocess functions for each dataset.
    if config.data.dataset in ["FFHQ", "CelebAHQ"]:

        def preprocess_fn(d):
            sample = tf.io.parse_single_example(
                d,
                features={
                    "shape": tf.io.FixedLenFeature([3], tf.int64),
                    "data": tf.io.FixedLenFeature([], tf.string),
                },
            )
            data = tf.io.decode_raw(sample["data"], tf.uint8)
            data = tf.reshape(data, sample["shape"])
            data = tf.transpose(data, (1, 2, 0))
            img = tf.image.convert_image_dtype(data, tf.float32)
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (
                    tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
                ) / 256.0
            return dict(image=img, label=None)

    else:

        def preprocess_fn(d):
            """Basic preprocessing function scales data to [0, 1) and randomly flips."""
            img = resize_op(d["image"])
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (
                    tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
                ) / 256.0

            return dict(image=img, label=d.get("label", None))

    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)
        if isinstance(dataset_builder, tfds.core.DatasetBuilder):
            dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(
                split=split, shuffle_files=True, read_config=read_config
            )
        else:
            ds = dataset_builder.with_options(dataset_options)
        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size)

    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    return train_ds, eval_ds, dataset_builder


from pathlib import Path


class fastmri_knee(Dataset):
    """Simple pytorch dataset for fastmri knee singlecoil dataset"""

    def __init__(self, root, is_complex=False):
        self.root = root
        self.data_list = list(root.glob("*/*.npy"))
        self.is_complex = is_complex

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        fname = self.data_list[idx]
        if not self.is_complex:
            data = np.load(fname)
        else:
            data = np.load(fname).astype(np.complex64)
        data = np.expand_dims(data, axis=0)
        return data


class AAPM(Dataset):
    def __init__(self, root, sort):
        self.root = root
        self.data_list = list(root.glob("full_dose/*.npy"))
        self.sort = sort
        if sort:
            self.data_list = sorted(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        fname = self.data_list[idx]
        data = np.load(fname)
        data = np.expand_dims(data, axis=0)
        return data


class Object5(Dataset):
    def __init__(self, root, slice, fast=False):
        """
        slice - range of the 2000 _volumes_ that you want,
        but the dataset will return images, so will be 256 times longer

        fast - set to true to get a tiny version of the dataset
        """
        if fast:
            self.NUM_SLICES = 10
        else:
            self.NUM_SLICES = 256

        self.root = root
        self.data_list = list(root.glob("*.npz"))

        if len(self.data_list) == 0:
            raise ValueError(f"No npz files found in {root}")

        self.data_list = sorted(self.data_list)[slice]

    def __len__(self):
        return len(self.data_list) * self.NUM_SLICES

    def __getitem__(self, idx):
        vol_index = idx // self.NUM_SLICES
        slice_index = idx % self.NUM_SLICES
        fname = self.data_list[vol_index]
        data = np.load(fname)["x"][slice_index]
        data = np.expand_dims(data, axis=0)
        return data


class fastmri_knee_infer(Dataset):
    """Simple pytorch dataset for fastmri knee singlecoil dataset"""

    def __init__(self, root, sort=True, is_complex=False):
        self.root = root
        self.data_list = list(root.glob("*/*.npy"))
        self.is_complex = is_complex
        if sort:
            self.data_list = sorted(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        fname = self.data_list[idx]
        if not self.is_complex:
            data = np.load(fname)
        else:
            data = np.load(fname).astype(np.complex64)
        data = np.expand_dims(data, axis=0)
        return data, str(fname)


class fastmri_knee_magpha(Dataset):
    """Simple pytorch dataset for fastmri knee singlecoil dataset"""

    def __init__(self, root):
        self.root = root
        self.data_list = list(root.glob("*/*.npy"))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        fname = self.data_list[idx]
        data = np.load(fname).astype(np.float32)
        return data


class fastmri_knee_magpha_infer(Dataset):
    """Simple pytorch dataset for fastmri knee singlecoil dataset"""

    def __init__(self, root, sort=True):
        self.root = root
        self.data_list = list(root.glob("*/*.npy"))
        if sort:
            self.data_list = sorted(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        fname = self.data_list[idx]
        data = np.load(fname).astype(np.float32)
        return data, str(fname)


# v2 12/20/2025
# 将此函数替换到 datasets.py 中
def get_CTSpine1K_dataset(config, train_val):
    import json
    import lmdb
    import torch
    import numpy as np
    from pathlib import Path
    from torch.utils.data import Dataset

    class MultiVolumeLMDBDataset(Dataset):
        def __init__(self, lmdb_path, patients_list=None, transform=None):
            self.lmdb_path = str(lmdb_path)
            self.transform = transform
            self.keys = []
            
            # 将 patients_list 转为 set 提高查询速度
            target_patients = set(patients_list) if patients_list else None
            
            print(f"Scanning LMDB: {self.lmdb_path} ...")
            # 预先扫描数据库，建立符合 dataset_split 的索引
            env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    key_str = key.decode("ascii")
                    
                    # 1. 跳过存储 shape 的辅助键
                    if key_str.endswith("_shape"):
                        continue
                    
                    # 2. 提取 Patient ID (格式: PatientID_SliceID)
                    # 例如: volume-covid19-A-0215_ct_001 -> volume-covid19-A-0215_ct
                    # 使用 rsplit 确保只切分最后一个下划线
                    pid = key_str.rsplit("_", 1)[0]
                    
                    # 3. 筛选逻辑：只保留 info.json 中存在的病人
                    if target_patients is not None:
                        if pid not in target_patients:
                            continue
                            
                    self.keys.append(key_str)
            env.close()
            print(f"Dataset loaded. Found {len(self.keys)} slices for {train_val}.")

            self.env = None
            self.txn = None

        def _init_db(self):
            # 多进程 DataLoader 必须在每个 worker 中重新打开 env
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.txn = self.env.begin(write=False)

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, index):
            if self.env is None:
                self._init_db()

            # 获取当前样本的 Key (string -> bytes)
            key_str = self.keys[index]
            key_bytes = key_str.encode('ascii')
            shape_key_bytes = (key_str + "_shape").encode('ascii')

            # 1. 读取数据
            byte_data = self.txn.get(key_bytes)
            shape_data = self.txn.get(shape_key_bytes)

            # 2. 反序列化与形状还原
            if shape_data:
                shape = np.frombuffer(shape_data, dtype=np.int32)
                img = np.frombuffer(byte_data, dtype=np.float32).reshape(tuple(shape)).copy()
            else:
                # 容错：如果没有 shape key，尝试按正方形 reshape
                flat_data = np.frombuffer(byte_data, dtype=np.float32)
                size = int(np.sqrt(len(flat_data)))
                img = flat_data.reshape((size, size)).copy()

            # 3. 增加 Channel 维度: (H, W) -> (1, H, W)
            img = torch.from_numpy(img).unsqueeze(0)

            # 4. 数据增强 (如果定义了 transform)
            if self.transform:
                img = self.transform(img)

            return img

    # --- 配置读取 ---
    # 读取 dataset_split.json
    with open(config.data.json, "r") as f:
        datalist = json.load(f)
    
    # 获取对应 split 的病人列表 (train 或 validation)
    patients = datalist[train_val]
    
    # 构造 LMDB 完整路径
    # 例如: /home/public/CTSpine1K/data/data_lmdb/CT_AX.lmdb
    lmdb_root = Path(config.data.root)
    lmdb_filename = f"{config.data.seq}_{config.data.orientation}.lmdb"
    lmdb_path = lmdb_root / lmdb_filename

    # CT数据已归一化且尺寸统一，通常不需要额外的 CenterCrop
    transform = None 

    return MultiVolumeLMDBDataset(
        lmdb_path=lmdb_path, 
        patients_list=patients, 
        transform=transform
    )


def create_dataloader(configs, evaluation=False, sort=True):
    shuffle = True if not evaluation else False
    dataset_name = str(configs.data.dataset).strip()
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "object5":
        train_dataset = Object5(Path(configs.data.root), slice(None, 1800))
        val_dataset = Object5(Path(configs.data.root), slice(1800, None))
    elif dataset_name_lower == "object5fast":
        train_dataset = Object5(Path(configs.data.root), slice(None, 1), fast=True)
        val_dataset = Object5(Path(configs.data.root), slice(1, 2), fast=True)
    elif dataset_name_lower == "aapm":
        train_dataset = AAPM(Path(configs.data.root) / f"train", sort=False)
        val_dataset = AAPM(Path(configs.data.root) / f"test", sort=True)
    elif configs.data.is_multi:
        train_dataset = fastmri_knee(
            Path(configs.data.root) / f"knee_multicoil_{configs.data.image_size}_train"
        )
        val_dataset = fastmri_knee_infer(
            Path(configs.data.root) / f"knee_{configs.data.image_size}_val", sort=sort
        )
    elif configs.data.is_complex:
        if configs.data.magpha:
            train_dataset = fastmri_knee_magpha(
                Path(configs.data.root)
                / f"knee_complex_magpha_{configs.data.image_size}_train"
            )
            val_dataset = fastmri_knee_magpha_infer(
                Path(configs.data.root)
                / f"knee_complex_magpha_{configs.data.image_size}_val"
            )
        else:
            train_dataset = fastmri_knee(
                Path(configs.data.root)
                / f"knee_complex_{configs.data.image_size}_train",
                is_complex=True,
            )
            val_dataset = fastmri_knee_infer(
                Path(configs.data.root) / f"knee_complex_{configs.data.image_size}_val",
                is_complex=True,
            )
    elif dataset_name_lower == "fastmri_knee":
        train_dataset = fastmri_knee(
            Path(configs.data.root) / f"knee_{configs.data.image_size}_train"
        )
        val_dataset = fastmri_knee_infer(
            Path(configs.data.root) / f"knee_{configs.data.image_size}_val", sort=sort
        )
    elif dataset_name_lower == "ctspine1k":
        train_dataset = get_CTSpine1K_dataset(configs, "train")
        val_dataset = get_CTSpine1K_dataset(configs, "validation")
    else:
        supported = ["Object5", "Object5Fast", "AAPM", "fastmri_knee", "CTSpine1K"]
        raise ValueError(
            f"Dataset '{dataset_name}' not recognized. Supported datasets: {supported}"
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=configs.training.batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=configs.training.batch_size,
        # shuffle=False,
        shuffle=True,
        drop_last=True,
    )
    return train_loader, val_loader


def create_dataloader_regression(configs, evaluation=False):
    shuffle = True if not evaluation else False
    train_dataset = fastmri_knee(
        Path(configs.root) / f"knee_{configs.image_size}_train"
    )
    val_dataset = fastmri_knee_infer(
        Path(configs.root) / f"knee_{configs.image_size}_val"
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=configs.batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, val_loader
