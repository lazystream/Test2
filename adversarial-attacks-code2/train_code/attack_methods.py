"""
attack_methods.py
"""
import torch
import torch.nn as nn

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
    def pgd(model, x, y, norm='Linf', eps=8 / 255, eps_iter=2 / 255,  # 修改点1：参数标准化
            iterations=10, restarts=1, random_start=True):
        """支持多范数的PGD攻击[2,4](@ref)
        Args:
            norm: 扰动范数类型 (Linf/L2)
            eps: 总扰动幅度(Linf时为像素值范围，L2时为L2半径)
            eps_iter: 单步扰动幅度
        """
        worst_x_adv = x.clone()
        worst_loss = float('-inf')

        for _ in range(restarts):
            # 初始化扰动（根据范数类型调整初始化逻辑）[4](@ref)
            delta = torch.zeros_like(x)
            if random_start:
                if norm == "Linf":
                    delta.uniform_(-eps, eps)
                elif norm == "L2":
                    delta.normal_()
                    delta *= eps * torch.rand(x.shape[0], 1, 1, 1, device=x.device)
                delta = torch.clamp(x + delta, -1, 1) - x  # 修改点2：L2的球面初始化
            x_adv = x + delta

            # 关键修改点：重构梯度计算逻辑
            for _ in range(iterations):
                x_adv = x_adv.clone().detach().requires_grad_(True)  # 保持叶节点属性

                # 使用上下文管理器保证梯度计算
                with torch.enable_grad():
                    loss = nn.CrossEntropyLoss()(model(x_adv), y)
                    grad = torch.autograd.grad(loss, x_adv)[0]

                # 停止梯度追踪后更新对抗样本
                with torch.no_grad():
                    if norm == "Linf":
                        grad = grad.sign()
                        step = eps_iter * grad
                    elif norm == "L2":
                        grad_norm = torch.norm(grad.view(x.shape[0], -1), p=2, dim=1)
                        grad = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-10)
                        step = eps_iter * grad

                    x_adv = x_adv + step
                    delta = x_adv - x

                # 投影操作（新增L2投影）[1,4](@ref)
                if norm == "Linf":
                    delta = torch.clamp(delta, -eps, eps)
                elif norm == "L2":
                    # 修改现有投影操作（当前代码第2-3行）
                    delta_norm = torch.norm(delta.view(x.shape[0], -1), p=2, dim=1)
                    delta = delta * (delta_norm <= eps).float().view(-1, 1, 1, 1) + \
                            delta * (eps / (delta_norm + 1e-10)).view(-1, 1, 1, 1) * (delta_norm > eps).float().view(-1,1,1,1) # 增加数值稳定性[3,8](@ref
                x_adv = torch.clamp(x + delta, -1, 1)
                x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)  # 双重约束确保扰动不超界[7](@ref)

                # 保存最佳攻击样本[4](@ref)
                with torch.no_grad():
                    current_loss = loss.item()
                    if current_loss > worst_loss:
                        worst_loss = current_loss
                        worst_x_adv = x_adv.clone()

        return worst_x_adv.detach()


