# 定义了多个损失函数类和一个自动加权多任务损失类，主要用于时间序列预测任务中的模型训练和评估

import torch as t
import torch
import torch.nn as nn
import numpy as np

# 执行安全的除法操作，确保结果中不会出现 NaN 或 Inf 值
def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    # 计算 a/b 后将结果中的 NaN 和 Inf 替换为 0
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

# 实现均绝对百分比误差（MAPE）损失，适用于时间序列预测任务，衡量预测值与目标值的相对误差
class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor):
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target) # 使用 divide_no_nan 计算权重 weights = mask / target，避免除零问题
        return t.mean(t.abs((forecast - target) * weights)) # 计算预测值和目标值的绝对误差，并乘以权重

# 实现对称均绝对百分比误差（sMAPE）损失
class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor):
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        # 使用 divide_no_nan 计算分母，避免除零问题
        # 计算预测值和目标值的绝对误差，并归一化
        # 返回误差的均值，乘以 200 以符合 sMAPE 的定义
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target), t.abs(forecast.data) + t.abs(target.data)) * mask)

# 实现均绝对比例误差（MASE）损失
class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor):
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        # 计算 insample 数据的平均绝对误差（分母）
        # 计算目标值和预测值的绝对误差，并乘以分母的倒数
        # 返回误差的均值
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

# 实现自动加权的多任务损失函数
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    # 初始化时为每个任务的损失分配一个可学习的权重参数
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)
    # 在前向传播中，根据权重参数动态调整每个任务的损失贡献，权重参数通过反向传播进行优化
    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum