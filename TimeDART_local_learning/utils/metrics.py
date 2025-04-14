import numpy as np

# 计算相对平方误差，用于衡量预测值与真实值之间的误差相对于真实值的波动程度
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

# 计算预测值与真实值之间的相关系数，反映两者的线性相关性
def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

# 计算预测值与真实值之间的平均绝对误差
def MAE(pred, true):
    return np.mean(np.abs(pred - true))

# 计算预测值与真实值之间的平均平方误差
def MSE(pred, true):
    return np.mean((pred - true) ** 2)

# 计算预测值与真实值之间的均方根误差
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

# 计算预测值与真实值之间的平均绝对百分比误差
def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

# 计算预测值与真实值之间的平均平方百分比误差
def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

# 汇总多个误差指标，返回 MAE、MSE、RMSE、MAPE 和 MSPE
def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
