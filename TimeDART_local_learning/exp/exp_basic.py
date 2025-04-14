import os
import torch
from models import TimeDART, SimMTM, PatchTST, TimeDART_v2

# 相当于提供了一个实验框架的骨架，重点在于设备管理和模型选择的灵活性
# 用于管理深度学习实验的核心流程，包括模型的构建、设备的选择、数据的获取、训练、验证和测试等，当然还有一部分在此还未实现
class Exp_Basic(object):
    # 还是先进行初始化
    def __init__(self, args):
        # 将传入的参数对象 args 赋值给类的实例变量 self.args，以便在类的其他方法中访问这些参数。
        self.args = args
        self.model_dict = { # 定义一个字典，映射模型名称到相应的模型类
            'TimeDART': TimeDART,
            'SimMTM': SimMTM,
            'PatchTST': PatchTST,
            'TimeDART_v2': TimeDART_v2
        }
        self.device = self._acquire_device() # 负根据 args 中的配置决定使用 CPU 还是 GPU 进行计算
        self.model = self._build_model().to(self.device) # 构建模型并将其移动到指定的设备上

    # 该方法用于构建模型，具体实现由子类提供（raise NotImplementedError）（由于不同实验可能需要不同的模型结构，具体的实现逻辑需要由继承该类的子类提供）
    def _build_model(self):
        raise NotImplementedError
        return None

    # 用于选择和初始化设备（CPU 或 GPU），并设置环境变量以便于多 GPU 使用
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        self.args.device = device
        return device
    # 不用，因为在data_factory.py中已经实现了
    def _get_data(self):
        pass
    # 以下均不用，也已经实现
    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
