#cw_test.py
import os
import time
import torch
import numpy as np
from utils import load_model_and_data,get_model_name
import torchattacks
from torchattacks import CW


def run_cw_test():
    # 获取device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 50}\n开始CW攻击测试\n{'=' * 50}")
    print(f"当前设备: {device}")

    # 加载模型和数据
    model, test_loader = load_model_and_data(mode='test')
    model.eval().to(device)  # 确保模型在正确设备上

    # 获取模型名称
    model_name = get_model_name()
    total_batches = len(test_loader)

    # 1. 先测试无扰动准确率（c=0）
    print("\n=== 无扰动测试 ===")
    clean_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            clean_correct += (model(x).argmax(1) == y).sum().item()
    clean_acc = 100 * clean_correct / len(test_loader.dataset)
    print(f"无扰动准确率: {clean_acc:.2f}%")

    # 2. 梯度攻击参数配置（从弱到强）
    cw_params = [
        {'c': 0.001, 'steps': 30, 'lr': 0.0005},
        {'c': 0.001, 'steps': 50, 'lr': 0.001},
        {'c': 0.01, 'steps': 50, 'lr': 0.001},  # 极弱攻击
        {'c': 0.1, 'steps': 100, 'lr': 0.005},  # 弱攻击
        {'c': 1, 'steps': 100, 'lr': 0.01},  # 中等攻击
        {'c': 10, 'steps': 200, 'lr': 0.01},  # 强攻击
        {'c': 100, 'steps': 300, 'lr': 0.01}  # 极强攻击
    ]

    results = {
        'no_attack': {
            'accuracy': clean_acc,
            'attack_success_rate': 0,
            'avg_confidence': 0,
            'time_sec': 0  # 添加time_sec字段
        }
    }

    for param_idx, params in enumerate(cw_params, 1):
        start_time = time.time()
        print(f"\n{'=' * 30} 测试进度: {param_idx}/{len(cw_params)} {'=' * 30}")
        print(f"当前参数组合: c={params['c']}, steps={params['steps']}, lr={params['lr']}")

        # 初始化攻击
        attack = CW(model, c=params['c'], steps=params['steps'], lr=params['lr'])
        correct = 0
        confidences = []

        for batch_idx, (x, y) in enumerate(test_loader, 1):
            x, y = x.to(device), y.to(device)

            # 显示进度
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"处理批次: {batch_idx}/{total_batches}", end='\r')

            # 生成对抗样本
            x_adv = attack(x, y)

            # 计算置信度
            with torch.no_grad():
                logits = model(x_adv)
                y_onehot = torch.nn.functional.one_hot(y, num_classes=logits.shape[-1])
                correct_logit = (logits * y_onehot).sum(1)
                wrong_logit = (logits - 1e6 * y_onehot).max(1)[0]
                batch_conf = (wrong_logit - correct_logit).mean().item()
                confidences.append(batch_conf)

                correct += (logits.argmax(1) == y).sum().item()

        # 计算结果指标
        acc = 100 * correct / len(test_loader.dataset)
        asr = 100 - acc  # 攻击成功率
        avg_conf = np.mean(confidences)
        elapsed = time.time() - start_time

        # 存储结果
        results[str(params)] = {
            'accuracy': acc,
            'attack_success_rate': asr,
            'avg_confidence': avg_conf,
            'time_sec': elapsed,
            'params': params.copy()  # 保存参数副本
        }

        # 实时打印结果
        print(f"\n测试完成: c={params['c']}")
        print(f"准确率: {acc:.2f}% | 攻击成功率: {asr:.2f}%")
        print(f"平均置信度: {avg_conf:.4f} | 耗时: {elapsed:.2f}s")

        # 保存结果（使用模型名称命名文件）
        os.makedirs('../results', exist_ok=True)
        report_filename = f"{model_name}_robustness_cw_report.txt"  # 使用模型名称
        report_path = os.path.join('../results', report_filename)

        with open(report_path, 'w') as f:
            f.write(f"模型名称: {model_name}\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"设备: {device}\n\n")
            f.write(f"\n基准准确率(无扰动): {clean_acc:.2f}%\n")
            f.write("=" * 50 + "\n")

            for k, v in results.items():
                if k == 'no_attack':
                    f.write("\n无扰动测试结果:\n")
                    f.write(f"准确率: {v['accuracy']:.2f}%\n")
                else:
                    f.write(f"\n参数组合: {k}\n")
                    f.write(f"准确率: {v['accuracy']:.2f}% (下降:{clean_acc - v['accuracy']:.2f}%)\n")
                    f.write(f"攻击成功率: {v['attack_success_rate']:.2f}%\n")
                    f.write(f"平均置信度: {v['avg_confidence']:.4f}\n")
                    f.write(f"耗时: {v['time_sec']:.2f}秒\n")
                f.write("-" * 50 + "\n")

        print(f"\n{'=' * 30} 测试完成 {'=' * 30}")
        print(f"完整结果已保存到: {report_path}")


if __name__ == "__main__":
    run_cw_test()