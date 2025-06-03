"""
evaluator.py
"""
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from attack_methods import AttackMethods


class RobustnessEvaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 添加device定义
        self.model.to(self.device)  # 确保模型在正确设备上
        self.results = {'clean': {'clean_baseline': {'accuracy': None, 'time': None}}}

    def evaluate(self, attack_name, params_list):
        """评估FGSM/PGD攻击"""
        print(f"\n{'=' * 50}")
        print(f"开始评估 {attack_name.upper()} 攻击")
        print(f"参数配置数量: {len(params_list)}")
        print(f"测试集批次: {len(self.test_loader)}")
        print(f"{'=' * 50}")

        # 参数预处理
        processed_params = []
        for params in params_list:
            params = params.copy()
            if attack_name == 'fgsm':
                if params['epsilon'] > 0.1:
                    print(f"[警告] FGSM的epsilon={params['epsilon']} 可能过大")
            elif attack_name == 'pgd':
                    pass
            processed_params.append(params)

        # 测试流程
        self.results[attack_name] = {}
        print(f"\n▶ 开始{attack_name.upper()}测试（共{len(processed_params)}组参数）")

        for i, params in enumerate(processed_params, 1):
            print(f"\n 正在测试参数组 {i}/{len(processed_params)}")
            print(f" 参数: {params}")

            start_time = time.time()
            correct = total = 0

            for batch_idx, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)  # 使用self.device
                if attack_name == 'clean':
                    x_adv = x
                elif attack_name == 'fgsm':
                    x_adv = AttackMethods.fgsm(self.model, x, y, **params)
                elif attack_name == 'pgd':
                    x_adv = AttackMethods.pgd(self.model, x, y, **params)

                with torch.no_grad():
                    correct += (self.model(x_adv).argmax(1) == y).sum().item()
                    total += y.size(0)

                if batch_idx % 10 == 0:
                    print(f"  已处理 {batch_idx}/{len(self.test_loader)} batches", end='\r')

            accuracy = 100 * correct / total
            print(f"\n✓ 完成 | 准确率: {accuracy:.2f}% | 耗时: {time.time() - start_time:.1f}s")

            if attack_name == 'clean':
                self.results['clean'] = {
                    'clean_baseline': {
                        'accuracy': accuracy,
                        'time': time.time() - start_time,
                        'params': {}
                    }
                }
            else:
                self.results[attack_name][str(params)] = {
                    'accuracy': accuracy,
                    'time': time.time() - start_time,
                    'params': params
                }

    # def plot_robustness_curve(self, attack_name):
    #     """绘制鲁棒性曲线"""
    #     data = self.results[attack_name]
    #     x = [p['params'].get('eps', p['params'].get('epsilon', 0)) for p in data.values()]
    #     y = [v['accuracy'] for v in data.values()]
    #
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(x, y, 'o-', label=attack_name.upper())
    #     plt.xlabel('epsilon' if attack_name != 'cw' else 'c')
    #     plt.ylabel('Accuracy (%)')
    #     plt.grid(True)
    #     plt.savefig(f"../results/{attack_name}_curve.png")
    #     plt.close()