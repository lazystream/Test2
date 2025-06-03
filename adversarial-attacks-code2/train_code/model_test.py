import os
import time
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 模型加载路径 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, 'trained_model', 'clean_best1.pth')

# 自动加载完整模型（包含结构和参数）
checkpoint = torch.load(model_path, map_location=device)
clean_model = models.resnet18(num_classes=10)  # 基础结构

# 自动应用训练时的结构修改
clean_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
clean_model.maxpool = nn.Identity()
clean_model.fc = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(clean_model.fc.in_features, 10)
)

# 加载参数
clean_model.load_state_dict(checkpoint['model_state_dict'])
clean_model = clean_model.to(device)
clean_model.eval()
# 1. 加载CIFAR-10测试集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
custom_data_path = "../dataset/cifar-10-batches-py"
test_dataset = datasets.CIFAR10(root='../dataset', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


# 2. 完整实现对抗攻击函数
class AttackMethods:
    @staticmethod
    def fgsm(model, x, y, epsilon=8 / 255):
        """FGSM攻击"""
        #print(f"\n[FGSM] 正在生成对抗样本(ε={epsilon:.4f})...")
        x.requires_grad = True
        loss = nn.CrossEntropyLoss()(model(x), y)
        model.zero_grad()
        loss.backward()
        x_adv = x + epsilon * x.grad.sign()
        x_adv = torch.clamp(x_adv, -1, 1)
        #print(f"[FGSM] 对抗样本生成完成")
        return x_adv.detach()

    @staticmethod
    def pgd(model, x, y, epsilon=8 / 255, alpha=2 / 255, iterations=10, restarts=1, random_start=True):
        #print(f"\n[PGD] 开始攻击(ε={epsilon:.4f}, α={alpha:.4f}, {iterations}次迭代)...")
        worst_x_adv = x.clone()
        worst_loss = float('-inf')

        for _ in range(restarts):
            # 严格根据random_start参数决定初始化方式
            if random_start:
                x_adv = x.clone() + torch.empty_like(x).uniform_(-epsilon, epsilon)
            else:
                x_adv = x.clone()
            x_adv = torch.clamp(x_adv, -1, 1)

            # 保持原有迭代逻辑不变
            for _ in range(iterations):
                x_adv.requires_grad = True
                loss = nn.CrossEntropyLoss()(model(x_adv), y)
                model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    if loss > worst_loss:
                        worst_loss = loss
                        worst_x_adv = x_adv.clone()

                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
                x_adv = torch.clamp(x_adv, -1, 1).detach()
        #print(f"[PGD] 攻击完成，最佳损失值: {worst_loss:.4f}")
        return worst_x_adv

    @staticmethod
    def cw(model, x, y, c=1e-4, max_iter=1000, learning_rate=0.01,
           binary_search_steps=5, abort_early=True, targeted=False, kappa=0):
        #print(f"\n[CW] 开始C&W攻击(c={c:.1e}, {max_iter}次迭代)...")
        best_x_adv = x.clone()
        best_confidence = -float('inf')  # 改为-inf，寻找最大loss

        for _ in range(binary_search_steps):
            x_adv = x.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([x_adv], lr=learning_rate)

            for step in range(max_iter):
                optimizer.zero_grad()
                output = model(x_adv)
                loss = nn.CrossEntropyLoss()(output, y)
                loss.backward()
                optimizer.step()

                # 修正为保留更难分类的样本
                if abort_early and step % 50 == 0:
                    with torch.no_grad():
                        current_conf = nn.CrossEntropyLoss()(model(x_adv), y)
                        if current_conf > best_confidence:  # 改为>比较
                            best_confidence = current_conf
                            best_x_adv = x_adv.clone()

                with torch.no_grad():
                    x_adv = torch.min(torch.max(x_adv, x - 8 / 255), x + 8 / 255)
                    x_adv = torch.clamp(x_adv, -1, 1)
        #print(f"[CW] 攻击完成，最终置信度: {best_confidence:.4f}")
        return best_x_adv


# 3. 多参数测试引擎
class RobustnessEvaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.results = {'clean': {'accuracy': None, 'time': None}}

    def evaluate(self, attack_name, params_list):
        """评估不同参数下的模型鲁棒性"""
        print(f"\n{'=' * 50}")
        print(f"开始评估 {attack_name.upper()} 攻击")
        print(f"参数配置数量: {len(params_list)}")
        print(f"测试集批次: {len(self.test_loader)}")
        print(f"{'=' * 50}")

        # 1. 参数预处理和验证
        processed_params = []
        for params in params_list:
            params = params.copy()  # 避免修改原始参数

            # FGSM参数检查
            if attack_name == 'fgsm':
                if params['epsilon'] > 0.1:
                    print(f"[警告] FGSM的epsilon={params['epsilon']} 可能过大")

            # PGD参数检查
            elif attack_name == 'pgd':
                if params['alpha'] > params['epsilon']:
                    raise ValueError("PGD的alpha不能大于epsilon")
                if params['iterations'] < 1:
                    raise ValueError("PGD的iterations必须≥1")
                if params['epsilon'] <= 0 or params['alpha'] <= 0:
                    raise ValueError("PGD的epsilon和alpha必须>0")

            # CW参数调整
            elif attack_name == 'cw':
                if params['c'] < 1e-6:
                    original_c = params['c']
                    params['c'] = 1e-6
                    print(f"[调整] CW参数c从{original_c:.1e}调整为1e-6")

            processed_params.append(params)

        # 2. 正式评估流程
        self.results[attack_name] = {}
        # 添加总进度打印
        print(f"\n▶ 开始{attack_name.upper()}测试（共{len(processed_params)}组参数）")

        for i, params in enumerate(processed_params):
            # 打印当前参数组信息
            print(f"\n 正在测试参数组 {i + 1}/{len(processed_params)}")
            print(f" 参数: {params}")

            if attack_name == 'clean':
                key = "clean_baseline"  # 使用固定的明确键名
            else:
                key = ",".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in sorted(params.items()))

            start_time = time.time()
            correct = 0
            total = 0

            # 添加batch进度显示
            print(f" 开始处理{len(self.test_loader)}个batch...")

            for batch_idx, (x, y) in enumerate(self.test_loader):
                x, y = x.to(device), y.to(device)

                # 每10个batch打印一次进度
                if batch_idx % 10 == 0:
                    print(f"  已处理 {batch_idx}/{len(self.test_loader)} batches", end='\r')

                if attack_name == 'clean':
                    x_adv = x  # 干净样本直接使用原始输入
                elif attack_name == 'fgsm':
                    x_adv = AttackMethods.fgsm(self.model, x, y, **params)
                elif attack_name == 'pgd':
                    x_adv = AttackMethods.pgd(self.model, x, y, **params)
                elif attack_name == 'cw':
                    x_adv = AttackMethods.cw(self.model, x, y, **params)
                else:
                    raise ValueError(f"未知攻击方法: {attack_name}")

                with torch.no_grad():
                    outputs = self.model(x_adv)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)

            elapsed_time = time.time() - start_time
            accuracy = 100 * correct / total
            print(f"\n✓ 参数组 {i + 1} 测试完成")
            print(f" 准确率: {accuracy:.2f}% | 耗时: {elapsed_time:.1f}s")


            self.results[attack_name][key] = {
                'accuracy': accuracy,
                'time': elapsed_time,
                'params': params
            }

        # 3. FGSM零扰动验证
        if attack_name == 'fgsm':
            # 查找最接近0的epsilon结果
            epsilon_values = [p['epsilon'] for p in processed_params]
            min_epsilon = min(epsilon_values)

            if min_epsilon < 1e-6:  # 视为0
                # 找到对应的key
                zero_key = next(
                    k for k in self.results['fgsm']
                    if abs(float(k.split('=')[1]) - 0) < 1e-6
                )
                zero_eps_acc = self.results['fgsm'][zero_key]['accuracy']
                clean_acc = self.results['clean']["clean_baseline"]['accuracy']

                if not math.isclose(zero_eps_acc, clean_acc, abs_tol=0.01):
                    print(f"[验证] FGSM(ε≈0)准确率: {zero_eps_acc:.4f}% (干净样本: {clean_acc:.4f}%)")
            else:
                print(f"[提示] 最小ε值为{min_epsilon:.2e}，未执行零扰动验证")
        print(f"\n{attack_name.upper()}评估完成，总耗时: {time.time() - start_time:.1f}s")
        return self.results

    def plot_robustness_curve(self, attack_name):
        """绘制鲁棒性曲线"""
        if attack_name not in self.results:
            raise ValueError(f"未找到 {attack_name} 的测试结果")

        data = self.results[attack_name]

        # 根据攻击类型选择参数键
        if attack_name == 'cw':
            param_key = 'c'  # CW攻击使用c参数
        else:
            param_key = 'epsilon'  # FGSM/PGD使用epsilon

        try:
            x = [float(p['params'][param_key]) for p in data.values()]
            y = [v['accuracy'] for v in data.values()]
        except KeyError:
            raise ValueError(f"无法找到参数键 '{param_key}' 用于 {attack_name} 攻击")

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o-', label=attack_name.upper())
        plt.xlabel(param_key)
        plt.ylabel('Accuracy (%)')
        plt.title(f'Robustness Curve ({attack_name.upper()})')
        plt.grid(True)
        plt.legend()

        # 保存图像
        os.makedirs(os.path.join('..', 'results', 'plots'), exist_ok=True)
        plt.savefig(os.path.join('..', 'results', 'plots', f'{attack_name}_robustness.png'))
        plt.close()


# 4. 参数配置
attack_configs = {
    'fgsm': [
        {'epsilon': e}
        for e in np.linspace(0, 16/255, 5)  # 0到16/255均匀采样
    ],
    'pgd': [
        {
            'epsilon': 8/255,
            'alpha': 2/255,
            'iterations': i,
            'random_start': True,  # 添加随机初始化
            'restarts': 1  # 添加多次重启
        }
        for i in [1, 5, 10, 20]
    ],
    'cw': [
        {
            'c': c,
            'max_iter': 500,
            'learning_rate': 0.01,
            'binary_search_steps': 5,  # 参数搜索增强
            'abort_early': True  # 添加提前终止
        }
        for c in np.logspace(-4, -1, 4)
    ]
}

# 5. 执行测试
print("开始模型鲁棒性评估")
print(f"设备: {device}")
print(f"测试集大小: {len(test_dataset)}")
print(f"攻击方法: {list(attack_configs.keys())}")
print("="*50)
evaluator = RobustnessEvaluator(clean_model, test_loader)


# 测试干净样本
print("\n" + "="*30)
print("测试干净样本准确率...")
evaluator.evaluate('clean', [{}])
clean_acc = evaluator.results['clean']["clean_baseline"]['accuracy']
print(f"干净样本测试完成，准确率: {clean_acc:.2f}%")

# 测试对抗样本
total_attacks = sum(len(v) for v in attack_configs.values())
current_attack = 0
for attack_name, params_list in attack_configs.items():
    current_attack += 1
    print(f"\n★ 开始执行 {attack_name.upper()} 攻击 ({current_attack}/{len(attack_configs)})")
    evaluator.evaluate(attack_name, params_list)
    evaluator.plot_robustness_curve(attack_name)
    print(f"✓ {attack_name.upper()} 攻击测试完成")


# 6. 生成综合报告
def generate_report(results):
    """生成详细测试报告"""
    report = ["对抗训练验证报告", "=" * 50, "\n"]

    # 基础测试结果
    report.append("基础测试结果:")
    report.append("-" * 50)
    report.append(f"干净样本准确率: {results['clean']['clean_baseline']['accuracy']:.2f}%")

    # 各攻击方法结果
    for attack_name in ['fgsm', 'pgd', 'cw']:
        if attack_name in results:
            report.append(f"\n{attack_name.upper()}攻击结果:")
            report.append("-" * 50)
            for param_str, data in results[attack_name].items():
                params = data['params']
                # 对PGD特殊处理
                if attack_name == 'pgd':
                    param_str = f"eps={params['epsilon']:.3f}, alpha={params['alpha']:.3f}, iter={params['iterations']}"
                elif attack_name == 'cw':
                    param_str = f"c={params['c']:.1e}, iter={params['max_iter']}"
                report.append(
                    f"参数: {param_str} | 准确率: {data['accuracy']:.2f}% | "
                    f"耗时: {data['time']:.2f}s"
                )
    # 添加总结段落
    report.append("\n总结:")
    report.append("-" * 50)
    min_acc = min(
        v['accuracy']
        for k in results.keys()
        if k != 'clean'
        for v in results[k].values()
    )
    report.append(f"最低对抗准确率: {min_acc:.2f}% (鲁棒性指标)")

    # 保存报告
    os.makedirs(os.path.join('..', 'results'), exist_ok=True)
    with open(os.path.join('..', 'results', 'clean_best1_robustness_report.txt'), 'w') as f:
        f.write("\n".join(report))

    return report


# 生成并打印报告
report = generate_report(evaluator.results)
print("\n".join(report[:20]))  # 打印前20行预览
print("\n完整报告已保存到: results/clean_best1_robustness_report.txt")