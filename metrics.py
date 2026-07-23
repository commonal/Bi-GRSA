import torch

def InfoNCE(view1, view2, temperature):
    view1, view2 = torch.nn.functional.normalize(
        view1, dim=1), torch.nn.functional.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)

def InfoNCE_i(view1, view2, view3,temperature,gama):
    view1, view2,view3 = torch.nn.functional.normalize(
        view1, dim=1), torch.nn.functional.normalize(view2, dim=1), torch.nn.functional.normalize(view3, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score_1 = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score_1 = torch.exp(ttl_score_1 / temperature).sum(dim=1)
    ttl_score_2 = torch.matmul(view1, view3.transpose(0, 1))
    ttl_score_2 = torch.exp(ttl_score_2 / temperature).sum(dim=1)

    cl_loss = -torch.log(pos_score / (gama*ttl_score_2 + ttl_score_1 + pos_score))
    return torch.mean(cl_loss)


def InfoNCE_with_weights(view1, view2, temperature, weights=None):
    """
    修正版：权重仅应用于负样本，保护正样本不被降权
    """
    # 1. 归一化
    view1 = torch.nn.functional.normalize(view1, dim=1)
    view2 = torch.nn.functional.normalize(view2, dim=1)

    # 2. 计算所有 Logits (N x N)
    logits = torch.matmul(view1, view2.transpose(0, 1)) / temperature

    # 3. 计算正样本 Logits (N)
    # 也可以直接取对角线: pos_logits = torch.diag(logits)
    pos_logits = (view1 * view2).sum(dim=-1) / temperature

    # 4. 指数化
    exp_logits = torch.exp(logits)

    if weights is not None:
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)  # [1, N]

        # [修正点核心] 创建一个 Mask，只对负样本应用权重
        batch_size = view1.shape[0]
        # 生成对角线 Mask (正样本位置为 1，其余为 0)
        pos_mask = torch.eye(batch_size, device=view1.device).bool()

        # 应用权重：我们希望 Negatives = Negatives * weights
        # 但是我们不想改变 Positives
        # 技巧：先全部乘权重，然后把对角线恢复回去
        # 或者：(exp_logits * weights) * (~pos_mask) + exp_logits * pos_mask

        weighted_exp_logits = exp_logits * weights

        # 关键：将正样本位置的值恢复为原始的 exp(pos_logits)，不受权重影响
        # 也就是：分母 = exp(pos) + sum(w_neg * exp(neg))
        # 下面这行代码把对角线上的值 替换回 原始未加权的值
        weighted_exp_logits = torch.where(pos_mask, torch.exp(pos_logits).unsqueeze(1), weighted_exp_logits)

        # 求和得到分母
        denominator = weighted_exp_logits.sum(dim=1)
    else:
        denominator = exp_logits.sum(dim=1)

    # 5. 计算损失
    # Loss = -log( exp(pos) / denominator )
    #      = - ( pos_logits - log(denominator) )
    log_prob = pos_logits - torch.log(denominator + 1e-8)

    return -torch.mean(log_prob)
# def InfoNCE_with_weights(view1, view2, temperature, weights=None):
#     """
#     带IPS权重的InfoNCE损失函数
#     :param view1: 第一个视图的嵌入
#     :param view2: 第二个视图的嵌入
#     :param temperature: 温度参数
#     :param weights: 用于调整负样本重要性的权重 (IPS权重)，形状应与ttl_score矩阵匹配
#     :return: 对比损失
#     """
#     view1, view2 = torch.nn.functional.normalize(
#         view1, dim=1), torch.nn.functional.normalize(view2, dim=1)
#     pos_score = (view1 * view2).sum(dim=-1)
#     pos_score = torch.exp(pos_score / temperature)
#
#     # 计算所有样本间的相似度
#     ttl_score = torch.matmul(view1, view2.transpose(0, 1))
#     ttl_score = torch.exp(ttl_score / temperature)
#
#     # 应用权重来平衡负样本的影响
#     if weights is not None:
#         # 确保权重维度正确
#         if weights.dim() == 1:
#             weights = weights.unsqueeze(0)
#         ttl_score = ttl_score * weights
#
#     # 计算加权后的总分（分母）
#     ttl_score = ttl_score.sum(dim=1)
#
#     cl_loss = -torch.log(pos_score / ttl_score)
#     return torch.mean(cl_loss)