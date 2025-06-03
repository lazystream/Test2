#maint_test.py
import os
import numpy as np
from evaluator import RobustnessEvaluator
from attack_methods import AttackMethods
from utils import load_model_and_data, generate_report, get_model_name

if __name__ == "__main__":
    # 加载模型和数据
    model, test_loader = model, test_loader = load_model_and_data(mode='test')

    # 获取模型名称
    model_name = get_model_name()

    # 初始化评估器
    evaluator = RobustnessEvaluator(model, test_loader)

    # 测试干净样本
    print("\n" + "=" * 30)
    print("测试干净样本准确率...")
    evaluator.evaluate('clean', [{}])

    # FGSM和PGD参数配置
    attack_configs = {
        'fgsm': [{'epsilon': e} for e in np.linspace(0, 16 / 255, 5)],
        'pgd': [
                # 标准Linf攻击（对抗训练黄金标准）
                {
                    'eps': 8/255,        # 与网页7的CIFAR-10标准对齐[7](@ref)
                    'eps_iter': 2/255,   # 单步扰动=总扰动/4（经验公式）
                    'iterations': 7      # Madry对抗训练论文验证的最优次数[7](@ref)
                },
                # 增强版Linf攻击（压力测试场景）
                {
                'eps': 16/255,       # 双倍扰动测试模型鲁棒性极限
                'eps_iter': 4/255,   # 保持步长比例一致性（16/4=4）
                'iterations': 20,    # 网页1建议的强度测试迭代次数[1](@ref)
                'restarts': 3        # 网页4推荐的多起点搜索策略[4](@ref)
                }
                ]
    }

    # 执行测试
    for attack_name, params_list in attack_configs.items():
        print(f"\n★ 开始执行 {attack_name.upper()} 攻击")
        evaluator.evaluate(attack_name, params_list)
        # evaluator.plot_robustness_curve(attack_name)

    # 生成报告（不包含CW部分）
    # 生成报告并保存到以模型名称命名的文件
    report = generate_report(evaluator.results)

    report_filename = f"{model_name}_robustness_main_report.txt"
    os.makedirs(os.path.join('..', 'results'), exist_ok=True)
    report_path = os.path.join('..', 'results', report_filename)

    with open(report_path, 'w') as f:
        f.write("\n".join(report))

    print("\n".join(report[:20]))
    print(f"\n完整报告已保存到: {report_path}")