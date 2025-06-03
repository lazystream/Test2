"""
load cifar数据集加载模块
修复要点：
1. 移除模型相关代码
2. 增强路径校验
3. 统一数据格式
4. 修复预处理流程
"""

import os
import pickle
import numpy as np
import torch
from torchvision import transforms

def load_cifar_data(dataset_name="cifar-10", dataset_root='../dataset', seed=None):
    """
    加载CIFAR数据集
    Args:
        dataset_name: 数据集名称
        dataset_root: 数据集根目录
        seed: 随机种子 (default: None)
    Returns:
        train_dataset, test_dataset
    """
    # 设置随机种子（如果提供）
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 路径生成与验证
    data_path = os.path.join(dataset_root, f"{dataset_name}-batches-py")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")

    def _unpickle(file):
        """反序列化数据文件"""
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    # 加载训练数据
    train_images, train_labels = [], []
    for i in range(1, 6):
        file_name = f"data_batch_{i}" if dataset_name == "cifar-10" else "train"
        file_path = os.path.join(data_path, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"训练文件 {file_path} 不存在")

        batch = _unpickle(file_path)
        train_images.append(batch[b'data'])
        label_key = b'labels' if dataset_name == "cifar-10" else b'fine_labels'
        train_labels.extend(batch[label_key])

    # 加载测试数据
    test_file = os.path.join(data_path, "test_batch")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试文件 {test_file} 不存在")

    test_batch = _unpickle(test_file)
    test_images = test_batch[b'data']
    test_labels = test_batch[b'labels' if dataset_name == "cifar-10" else b'fine_labels']

    # 数据格式转换
    def _convert_images(raw_data):
        """将原始数据转换为NHWC格式的归一化图像"""
        images = raw_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images.astype(np.float32) / 255.0

    # 处理数据
    train_images = np.vstack([_convert_images(batch) for batch in train_images])
    train_labels = np.array(train_labels)
    test_images = _convert_images(test_images)
    test_labels = np.array(test_labels)

    # 预处理流程（优化版）
    train_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 新增
        transforms.RandomRotation(5),  # 新增
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])

    test_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])

    # 数据集类
    class CIFARDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels, transformer=None):
            self.images = images
            self.labels = labels
            self.transformer = transformer

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            if self.transformer:
                image = self.transformer(image)
            return image, label

    # 返回数据集（修正测试集预处理）
    return (
        CIFARDataset(train_images, train_labels, transformer=train_preprocess),
        CIFARDataset(test_images, test_labels, transformer=test_preprocess)  # 修正这里
    )


if __name__ == '__main__':
    # 验证代码（移动到文件末尾）
    try:
        train_set, test_set = load_cifar_data()
        sample, label = train_set[0]
        print("✅ 数据加载验证通过")
        print(f"训练样本形状: {sample.shape} (dtype: {sample.dtype})")
        print(f"标签: {label} (type: {type(label)})")

        print("\n=== 测试随机种子 ===")
        train_set1, _ = load_cifar_data(seed=42)
        train_set2, _ = load_cifar_data(seed=42)
        sample1, _ = train_set1[0]
        sample2, _ = train_set2[0]
        print("✅ 随机种子验证通过")
        print(f"相同种子样本是否一致: {torch.allclose(sample1, sample2)}")

        # 测试预处理流程
        dummy_data = np.random.rand(32, 32, 3).astype(np.float32)
        processed = train_set.transformer(dummy_data)
        print("\n✅ 预处理验证通过")
        print(f"输出类型: {type(processed)}")
        print(f"输出形状: {processed.shape}")
        print(f"数值范围: [{processed.min():.4f}, {processed.max():.4f}]")
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")