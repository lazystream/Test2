"""
utils.py

"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

checkpoint_path='../trained_model/adversarial_cw_pure_best.pth'

def load_model_and_data(train_ratio=0.8, mode='test',seed = 42):
    """
    改进的模型和数据加载函数，支持不同运行模式

    参数:
        train_ratio (float): 训练集划分比例，默认0.8
        mode (str): 运行模式，'test'仅返回测试集，'train'返回完整数据加载器

    返回:
        根据mode返回不同数据:
        - mode='test': (model, test_loader)
        - mode='train': (model, train_loader, val_loader, test_loader)
    """
    # 设置随机种子（如果seed不是None）
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 设置Python和NumPy随机种子
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 模型加载部分 ---
    model = models.resnet18(num_classes=10)
    # 修改网络结构适配CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除最大池化层
    model.fc = nn.Sequential(
        nn.Dropout(0.1),  # 添加dropout防止过拟合
        nn.Linear(model.fc.in_features, 10)
    )

    # 加载预训练权重

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # 固定为评估模式

    # --- 数据加载部分 ---
    # CIFAR-10官方推荐的标准归一化参数
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    # 训练集变换（含数据增强）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 测试集变换（仅标准化）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 加载完整数据集
    full_train_set = datasets.CIFAR10(
        root='../dataset',
        train=True,
        download=True,
        transform=train_transform
    )

    test_set = datasets.CIFAR10(
        root='../dataset',
        train=False,
        download=True,
        transform=test_transform
    )

    # 创建测试集加载器（所有模式都需要）
    test_loader = DataLoader(
        test_set,
        batch_size=100,
        shuffle=False,  # 测试集必须关闭shuffle
        num_workers=2
    )

    if mode == 'test':
        return model, test_loader

        # 划分训练集和验证集
        train_size = int(train_ratio * len(full_train_set))
        val_size = len(full_train_set) - train_size

        train_set, val_set = torch.utils.data.random_split(
            full_train_set,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed) if seed is not None else None
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_set,
            batch_size=256,  # 增大batch size
            shuffle=True,
            num_workers=8,  # 增加worker数量
            pin_memory=True,
            persistent_workers=True,  # 保持worker进程
            prefetch_factor=4,  # 预加载批次
            generator=torch.Generator().manual_seed(seed) if seed is not None else None
        )

    val_loader = DataLoader(
        val_set,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    return model, train_loader, val_loader, test_loader


def generate_report(results):
    """生成详细测试报告"""
    if not results:  # 添加空结果检查
        return ["警告：没有可用的测试结果"]
    report = ["对抗训练验证报告", "=" * 50, "\n"]

    # 新增参数记录段
    report.append("\n=== 攻击参数配置 ===")
    for attack_name in results:
        if attack_name == 'clean': continue

        report.append(f"\n◆ {attack_name.upper()}参数组:")
        attack_data = results[attack_name]
        for i, (config_key, config) in enumerate(attack_data.items(), 1):
            params = config['params']
            param_str = ",  ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                                    for k, v in params.items()])
            report.append(f"  组{i}: {param_str}")

    # 基础测试结果
    report.append("基础测试结果:")
    report.append("-" * 50)
    if 'clean' in results and 'clean_baseline' in results['clean']:
        report.append(f"干净样本准确率: {results['clean']['clean_baseline']['accuracy']:.2f}%")
    else:
        report.append("干净样本数据缺失")

    # 各攻击方法结果
    for attack_name in ['fgsm', 'pgd', 'cw']:
        if attack_name in results:
            report.append(f"\n{attack_name.upper()}攻击结果:")
            report.append("-" * 50)
            for param_str, data in results[attack_name].items():
                params = data['params']
                # 对PGD特殊处理（兼容新旧参数命名）
                if attack_name == 'pgd':
                    # 使用get()方法避免KeyError，兼容epsilon/eps两种参数名
                    epsilon = params.get('epsilon', params.get('eps', 0))
                    # 兼容alpha/eps_iter两种参数名
                    alpha = params.get('alpha', params.get('eps_iter', 0))
                    norm_type = params.get('norm', 'Linf')  # 新增norm参数显示
                    param_str = (
                        f"norm={norm_type}, eps={epsilon:.3f}, "
                        f"step={alpha:.3f}, iter={params['iterations']}"
                    )
                elif attack_name == 'cw':
                    cw_params = {
                        'c': params.get('c', 1e-4),  # 约束系数
                        'steps': params.get('steps', 1000),  # 迭代次数
                        'lr': params.get('lr', 0.01),  # 学习率
                        'binary_search_steps': params.get('binary_search_steps', 9),  # 二分搜索次数
                        'abort_early': params.get('abort_early', False)  # 提前终止
                    }
                    param_str = (
                        f"c={params['c']:.1e}, steps={params['steps']}, "
                        f"lr={params['lr']:.3f}, binary_search={params['binary_search_steps']}, "
                        f"early_stop={params['abort_early']}"
                    )
                report.append(
                    f"参数: {param_str} | 准确率: {data['accuracy']:.2f}% | "
                    f"耗时: {data['time']:.2f}s"
                )

    # 添加总结段落
    report.append("\n总结:")
    report.append("-" * 50)
    try:
        min_acc = min(
            v['accuracy']
            for k in results.keys()
            if k != 'clean'
            for v in results[k].values()
        )
        report.append(f"最低对抗准确率: {min_acc:.2f}% (鲁棒性指标)")
    except (ValueError, KeyError):
        report.append("无法计算最低对抗准确率")

    return report


def get_model_name() -> str:
    """获取当前模型名称（基于预训练权重文件路径）

    参数:
        checkpoint_path (str): 预训练权重路径，默认为clean_best1.pth

    返回:
        str: 去除路径和扩展名的纯模型名称
    """
    return os.path.splitext(os.path.basename(checkpoint_path))[0]