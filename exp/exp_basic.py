import os
import torch
from models import TimeDART, SimMTM, PatchTST, TimeDART_v2, TimeDART_v3 # 新加一个TimeDART_v3


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimeDART': TimeDART,
            'SimMTM': SimMTM,
            'PatchTST': PatchTST,
            'TimeDART_v2': TimeDART_v2,
            'TimeDART_v3': TimeDART_v3
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

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

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    # 迁移预训练权重，linear-tuning 需要
    def map_pretrained_weights(self, pretrained_model, target_model):
        """
        手动迁移预训练权重，仅迁移 embedding、sos token 和 encoder。
        """
        # 迁移 embedding 权重
        target_model.enc_embedding.load_state_dict(pretrained_model.enc_embedding.state_dict())

        # 迁移 sos token
        target_model.sos_token.data.copy_(pretrained_model.sos_token.data)

        # 迁移 encoder 权重
        target_model.encoder.load_state_dict(pretrained_model.encoder.state_dict())