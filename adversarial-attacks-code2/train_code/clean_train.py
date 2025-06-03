"""
clean_train.py - 不添加扰动的干净训练代码
实现基于ResNet18的CIFAR-10分类模型训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
import time
import argparse
import os

# 导入您提供的CIFAR-10数据加载模块
from load_cifar import load_cifar_data

from torch.optim.lr_scheduler import CosineAnnealingLR


def build_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # 新增：更激进的残差块修改
    def _make_layer(block, planes, blocks, stride=1):
        layers = []
        layers.append(block(64, planes, stride))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    model.layer1 = _make_layer(models.resnet.BasicBlock, 64, 2)
    model.fc = nn.Sequential(
        nn.Dropout(0.1),  # 新增dropout
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def train(model, device, train_loader, optimizer, criterion, epoch,scaler):
    """
    训练函数
    Args:
        model: 要训练的模型
        device: 训练设备
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epoch: 当前epoch数
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    torch.backends.cudnn.benchmark = True
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        progress_bar.set_postfix({
            'Loss': f"{train_loss / (progress_bar.n + 1):.3f}",
            'Acc': f"{100. * correct / total:.2f}%"
        })

    # 移到epoch结束后
    torch.cuda.empty_cache()
    print(f"Epoch {epoch} LR: {optimizer.param_groups[0]['lr']:.6f}")


def test(model, device, test_loader, criterion):
    """
    测试函数
    Args:
        model: 要测试的模型
        device: 测试设备
        test_loader: 测试数据加载器
        criterion: 损失函数
    Returns:
        测试准确率
    """
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
    """保存训练日志到txt文件"""
    log_path = os.path.join(save_dir, filename)
    with open(log_path, 'w') as f:
        for key, value in log_data.items():
            f.write(f"{key}: {value}\n")

def main():
    # 参数设置
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Clean Training')
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
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设备设置后立即配置cudnn
    torch.backends.cudnn.benchmark = True

    # 加载数据集
    train_dataset, test_dataset = load_cifar_data(dataset_name="cifar-10", dataset_root=args.dataset_root)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 构建模型
    model = build_resnet18().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)  # 调小初始LR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)  # 余弦退火

    # 训练循环
    best_acc = 0.0
    last_acc = 0.0
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trained_model')
    scaler = torch.cuda.amp.GradScaler()  # 混合精度训练

    # 训练循环前记录开始时间
    start_time = time.time()  # 使用time模块记录时间戳

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
        'scheduler': 'CosineAnnealingLR'
    }


    # ===== 新增：恢复训练支持和早停机制 =====
    no_improve = 0  # 初始化早停计数器
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"恢复训练：从epoch {start_epoch}开始，历史最佳准确率 {best_acc:.2f}%")

    for epoch in range(start_epoch, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch, scaler)
        current_acc = test(model, device, test_loader, criterion)
        scheduler.step()

        # 更新训练日志
        training_log['current_epoch'] = epoch
        training_log['current_lr'] = optimizer.param_groups[0]['lr']
        training_log['final_accuracy'] = current_acc

        if current_acc > best_acc:
            best_acc = current_acc
            no_improve = 0  # 重置计数器
            training_log['best_accuracy'] = best_acc
            training_log['best_epoch'] = epoch

            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }, os.path.join(save_dir, 'clean_best1.pth'))
            save_training_log(save_dir, 'clean_best1_model_log.txt', training_log)

        elif current_acc > last_acc:  # 新增关键判断
            no_improve = 0
        else:
            no_improve += 1

        last_acc = current_acc  # 记录上一轮准确率

        if no_improve >= args.patience:  # 默认patience=10
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
            torch.save(model.state_dict(), os.path.join(save_dir, 'cifar10_clean_final.pth'))
            save_training_log(save_dir, 'final_model_log.txt', training_log)

        # 最佳模型日志也记录训练时间
        training_log['best_accuracy'] = best_acc
        save_training_log(save_dir, 'clean_best1_model_log.txt', training_log)

if __name__ == '__main__':
    main()