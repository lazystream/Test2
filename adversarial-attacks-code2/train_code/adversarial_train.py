"""
adversarial_train.py - 对抗训练代码
实现基于ResNet18的CIFAR-10分类模型的对抗训练

支持FGSM、PGD和CW三种对抗训练方法
# FGSM组合训练
python adversarial_train.py --seed 3 --attack-method fgsm

# PGD组合训练
python adversarial_train.py --seed 3 --attack-method pgd

# CW组合训练
python adversarial_train.py --seed 3 --attack-method cw

# FGSM纯对抗训练
python adversarial_train.py --seed 3 --attack-method fgsm --pure-adv

# PGD纯对抗训练
python adversarial_train.py --seed 3 --attack-method pgd --pure-adv

# CW纯对抗训练
python adversarial_train.py --seed 3 --attack-method cw --pure-adv
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
import time
import argparse
import os
from load_cifar import load_cifar_data
from torchattacks import CW  # 用于CW攻击

# 导入攻击方法
from attack_methods import AttackMethods


def build_resnet18(num_classes=10):
    """与clean_train.py相同的模型构建函数"""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    def _make_layer(block, planes, blocks, stride=1):
        layers = []
        layers.append(block(64, planes, stride))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    model.layer1 = _make_layer(models.resnet.BasicBlock, 64, 2)
    model.fc = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def generate_adversarial_examples(model, data, target, attack_method, attack_params):
    """
    生成对抗样本的通用函数
    Args:
        model: 目标模型
        data: 原始输入数据
        target: 真实标签
        attack_method: 攻击方法 ('fgsm', 'pgd', 'cw')
        attack_params: 攻击参数字典
    Returns:
        对抗样本
    """
    # 保存原始模型状态
    training = model.training
    model.eval()  # 确保模型在评估模式

    # 确保输入数据需要梯度
    data = data.clone().detach().requires_grad_(True)
    target = target.clone().detach()

    # 生成对抗样本
    try:
        if attack_method == 'fgsm':
            adv_data = AttackMethods.fgsm(model, data, target, **attack_params)
        elif attack_method == 'pgd':
            adv_data = AttackMethods.pgd(model, data, target, **attack_params)
        elif attack_method == 'cw':
            # 特殊处理CW攻击
            with torch.enable_grad():  # 显式启用梯度
                attack = CW(model, **attack_params)
                adv_data = attack(data, target)
        else:
            raise ValueError(f"未知的攻击方法: {attack_method}")
    finally:
        # 确保恢复模型原始状态
        model.train(training)

    return adv_data.detach()  # 返回时断开计算图


def adversarial_train(model, device, train_loader, optimizer, criterion, epoch, scaler,
                     attack_method, attack_params, pure_adv=False,save_dir=None):
    """
    对抗训练函数
    增强的对抗训练函数，支持两种模式
    Args:
        pure_adv: 如果为True则使用纯对抗训练，False使用组合训练
    Args:
        model: 要训练的模型
        device: 训练设备
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epoch: 当前epoch数
        attack_method: 使用的对抗攻击方法 ('fgsm', 'pgd', 'cw')
        attack_params: 对抗攻击的参数
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    torch.backends.cudnn.benchmark = True

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)

        # 确保数据在生成对抗样本前设置梯度
        data = data.clone().detach().requires_grad_(True)

        # 生成对抗样本
        if attack_method == 'cw':
            # CW需要特殊处理模型状态
            with torch.enable_grad():
                model.eval()
                attack = CW(model, **attack_params)
                adv_data = attack(data, target)
                model.train()
        else:
            # 常规攻击方法
            model.eval()
            adv_data = generate_adversarial_examples(model, data, target,
                                                     attack_method, attack_params)
            model.train()

        # 根据训练模式选择损失计算方式
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            # 所有攻击方法共用同一套损失计算逻辑
            clean_output = model(data) if not pure_adv else None
            adv_output = model(adv_data)

            loss = criterion(adv_output, target)  # 基础对抗损失
            if not pure_adv:
                loss += criterion(clean_output, target)  # 组合训练时添加干净样本损失

            # 在autocast区域外转换数据类型
            output_for_acc = adv_output.float() if adv_output.dtype != torch.float32 else adv_output  # 统一使用对抗样本输出计算准确率

        # 反向传播和优化
        scaler.scale(loss).backward()

        # 解缩梯度以进行监控和裁剪
        scaler.unscale_(optimizer)

        # 梯度监控应在此处（裁剪前）
        if attack_method == 'cw' and (progress_bar.n % 10 == 0):
            # 计算裁剪前梯度范数
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            pre_clip_grad_norm = torch.norm(torch.stack([g.norm() for g in grads])) if grads else 0.0

            # 执行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 计算裁剪后梯度范数
            post_clip_grad_norm = torch.norm(torch.stack([g.norm() for g in grads])) if grads else 0.0

            # 保持旧变量兼容
            grad_norm = pre_clip_grad_norm

            # 打印监控信息
            print(f"【梯度监控】裁剪前: {pre_clip_grad_norm:.4f} | 裁剪后: {post_clip_grad_norm:.4f}")

            # 写入日志
            log_content = (
                f"[Epoch {epoch} | Batch {progress_bar.n}]\n"
                f"Pre-Clip: {pre_clip_grad_norm:.4f}\n"
                f"Post-Clip: {post_clip_grad_norm:.4f}\n"
                "------------------------\n"
            )
            with open(os.path.join(save_dir, f'adversarial_{attack_method}_best_log.txt'), 'a') as f:
                f.write(log_content)

        # 执行梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 可添加裁剪后监控（可选）
        if attack_method == 'cw' and (progress_bar.n % 10 == 0):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
            print(f"【梯度监控】裁剪后梯度范数: {grad_norm:.4f}")

        # 执行优化器更新
        scaler.step(optimizer)
        scaler.update()

        # 计算统计信息
        train_loss += loss.item()
        _, predicted = output_for_acc.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        progress_bar.set_postfix({
            'Loss': f"{train_loss / (progress_bar.n + 1):.3f}",
            'Acc': f"{100. * correct / total:.2f}%",
            'Mode': 'PureAdv' if pure_adv else 'Combined'
        })
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保CUDA操作完成
            torch.cuda.empty_cache()  # 及时释放碎片化显存

    # 清理缓存
    torch.cuda.empty_cache()
    print(f"Epoch {epoch} LR: {optimizer.param_groups[0]['lr']:.6f}")


def test(model, device, test_loader, criterion):
    """与clean_train.py相同的测试函数"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy


def save_training_log(save_dir, filename, log_data):
    """与clean_train.py相同的日志保存函数"""
    log_path = os.path.join(save_dir, filename)
    with open(log_path, 'w') as f:
        for key, value in log_data.items():
            f.write(f"{key}: {value}\n")


def get_default_attack_params(method):
    """获取不同攻击方法的默认参数"""
    if method == 'fgsm':
        return {'epsilon': 8 / 255}
    elif method == 'pgd':
        return {'epsilon': 8 / 255, 'alpha': 2 / 255, 'iterations': 10}
    elif method == 'cw':
        return {
            'c': 1e-3,  # 增大约束系数（加快收敛）
            'steps': 100,  # 减少迭代次数（原100）
            'lr': 0.02,  # 增大学习率（原0.01）
        }
    else:
        raise ValueError(f"未知的攻击方法: {method}")


def save_training_log(save_dir, filename, log_data):
    """增强的训练日志保存函数，记录所有关键训练参数"""
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    log_path = os.path.join(save_dir, filename)

    # 添加分隔线和对齐格式
    with open(log_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("对抗训练详细日志\n")
        f.write("=" * 50 + "\n\n")

        # 按类别分组记录信息
        f.write("[训练配置]\n")
        f.write("-" * 50 + "\n")
        for key in ['batch_size', 'epochs', 'patience', 'training_mode',
                    'training_device', 'model_architecture']:
            if key in log_data:
                f.write(f"{key:<25}: {log_data[key]}\n")

        f.write("\n[优化器配置]\n")
        f.write("-" * 50 + "\n")
        for key in ['optimizer', 'initial_lr', 'momentum', 'weight_decay',
                    'scheduler', 'current_lr']:
            if key in log_data:
                f.write(f"{key:<25}: {log_data[key]}\n")

        f.write("\n[对抗训练参数]\n")
        f.write("-" * 50 + "\n")
        for key in ['attack_method', 'attack_params', 'pure_adv']:
            if key in log_data:
                f.write(f"{key:<25}: {log_data[key]}\n")

        f.write("\n[随机种子]\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'python_random_seed':<25}: {log_data.get('python_random_seed', 'N/A')}\n")
        f.write(f"{'numpy_random_seed':<25}: {log_data.get('numpy_random_seed', 'N/A')}\n")
        f.write(f"{'torch_random_seed':<25}: {log_data.get('torch_random_seed', 'N/A')}\n")
        f.write(f"{'cuda_random_seed':<25}: {log_data.get('cuda_random_seed', 'N/A')}\n")
        f.write(f"{'cudnn_deterministic':<25}: {log_data.get('cudnn_deterministic', 'N/A')}\n")
        f.write(f"{'cudnn_benchmark':<25}: {log_data.get('cudnn_benchmark', 'N/A')}\n")

        f.write("\n[训练结果]\n")
        f.write("-" * 50 + "\n")
        for key in ['best_accuracy', 'best_epoch', 'final_accuracy',
                    'total_training_time', 'early_stop', 'early_stop_epoch']:
            if key in log_data:
                f.write(f"{key:<25}: {log_data[key]}\n")

        f.write("\n[环境信息]\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'torch_version':<25}: {torch.__version__}\n")
        f.write(f"{'cuda_available':<25}: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"{'cuda_version':<25}: {torch.version.cuda}\n")
            f.write(f"{'gpu_name':<25}: {torch.cuda.get_device_name(0)}\n")

def main():
    # 参数设置
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Adversarial Training')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, help='Learning rate step gamma')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--save-model', action='store_true', default=False, help='For saving the current Model')
    parser.add_argument('--dataset-root', type=str, default='../dataset', help='path to dataset')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--attack-method', type=str, default='pgd',
                        choices=['fgsm', 'pgd', 'cw'],
                        help='Adversarial training method (default: pgd)')
    parser.add_argument('--pure-adv', action='store_true',
                        help='Use pure adversarial training (default: combined training)')
    args = parser.parse_args()

    # 设置随机种子（在所有操作之前）
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)

    # CUDA随机种子（如果使用GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # 多GPU情况
        torch.backends.cudnn.deterministic = True  # 保证卷积结果确定
        torch.backends.cudnn.benchmark = False  # 关闭benchmark以获得确定性

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 设备设置
    torch.backends.cudnn.benchmark = True  # 启用CuDNN自动优化
    torch.set_float32_matmul_precision('high')  # 加速矩阵运算

    # 加载数据集
    train_dataset, test_dataset = load_cifar_data(
        dataset_name="cifar-10",
        dataset_root=args.dataset_root,
        seed=args.seed  # 传递种子参数
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 构建模型
    model = build_resnet18().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)  # 调小初始LR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)  # 余弦退火

    # 获取攻击参数
    attack_params = get_default_attack_params(args.attack_method)

    # 训练循环
    best_acc = 0.0
    last_acc = 0.0
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trained_model')
    scaler = torch.amp.GradScaler(enabled=True)  # 混合精度训练

    # 训练循环前记录开始时间
    start_time = time.time()

    # 初始化训练日志
    training_log = {
        'batch_size': args.batch_size,
        'initial_lr': 0.05,
        'epochs': args.epochs,
        'patience': args.patience,
        'best_accuracy': 0.0,
        'final_accuracy': 0.0,
        'training_device': str(device),
        'model_architecture': 'ResNet18-modified',
        'optimizer': 'SGD',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'scheduler': 'CosineAnnealingLR',
        'attack_method': args.attack_method,
        'attack_params': str(attack_params),
        'training_mode':'pure_adversarial' if args.pure_adv else 'combined',
        # 新增随机种子记录
        'python_random_seed': args.seed,
        'numpy_random_seed': args.seed,
        'torch_random_seed': args.seed,
        'cuda_random_seed': args.seed if torch.cuda.is_available() else 'N/A',
        'cudnn_deterministic': str(torch.backends.cudnn.deterministic),
        'cudnn_benchmark': str(torch.backends.cudnn.benchmark),

    }

    # 恢复训练支持和早停机制
    no_improve = 0
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"恢复训练：从epoch {start_epoch}开始，历史最佳准确率 {best_acc:.2f}%")

    scaler = torch.amp.GradScaler(enabled=True)  # 混合精度训练初始化

    for epoch in range(start_epoch, args.epochs + 1):
        # 使用对抗训练
        adversarial_train(model, device, train_loader, optimizer, criterion,
                          epoch, scaler, args.attack_method, attack_params, args.pure_adv, save_dir=save_dir)

        current_acc = test(model, device, test_loader, criterion)
        scheduler.step()

        # 更新训练日志
        training_log['current_epoch'] = epoch
        training_log['current_lr'] = optimizer.param_groups[0]['lr']
        training_log['final_accuracy'] = current_acc

        if current_acc > best_acc:
            best_acc = current_acc
            no_improve = 0
            training_log['best_accuracy'] = best_acc
            training_log['best_epoch'] = epoch

            # 保存最佳模型
            model_suffix = "_pure_best.pth" if args.pure_adv else "_best.pth"
            model_name = f'adversarial_{args.attack_method}{model_suffix}'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }, os.path.join(save_dir, model_name))
            log_suffix = "_pure_best_log.txt" if args.pure_adv else "_best_log.txt"
            save_training_log(save_dir, f'adversarial_{args.attack_method}{log_suffix}', training_log)
        elif current_acc > last_acc:
            no_improve = 0
        else:
            no_improve += 1

        last_acc = current_acc

        if no_improve >= args.patience:
            print(f"\n早停触发：连续{args.patience}轮准确率未提升")
            training_log['early_stop'] = True
            training_log['early_stop_epoch'] = epoch
            break

        # 计算总训练时间
        training_time = time.time() - start_time
        training_log[
            'total_training_time'] = f"{training_time // 3600:.0f}h {(training_time % 3600) // 60:.0f}m {training_time % 60:.2f}s"

        # 最终模型保存和日志记录
        if args.save_model:
            model_suffix = "_pure_final.pth" if args.pure_adv else "_final.pth"
            model_name = f'cifar10_adversarial_{args.attack_method}{model_suffix}'
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))
            save_training_log(save_dir, 'final_model_log.txt', training_log)

        # 最佳模型日志也记录训练时间
        training_log['best_accuracy'] = best_acc
        save_training_log(save_dir, f'adversarial_{args.attack_method}_best_log.txt', training_log)


if __name__ == '__main__':
    main()