from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    transfer_weights,
    show_series,
    show_matrix,
)
from utils.augmentations import masked_data
from utils.metrics import metric
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# 定义一个实验类Exp_TimeDART，继承自Exp_Basic，作为其子类
class Exp_TimeDART(Exp_Basic):
    # 初始化的时候直接调用父类的初始化方法，并创建了一个 SummaryWriter 实例，用于记录实验日志和可视化数据（如损失曲线）
    def __init__(self, args):
        super(Exp_TimeDART, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs") # 日志文件保存在outputs/logs目录下

    # 根据任务类型动态选择模型，不过这里学长删去了模型并行化并行化进行多GPU训练的设计
    def _build_model(self):
        # 如果任务是时间序列预测（forecast），则实例化模型的 Model 类（Model 类是一个时间序列预测模型）
        if self.args.downstream_task == "forecast":
            model = self.model_dict[self.args.model].Model(self.args).float()
        # 如果任务是分类（classification），则实例化模型的 ClsModel 类（lsModel 类是一个分类模型）
        elif self.args.downstream_task == "classification":
            model = self.model_dict[self.args.model].ClsModel(self.args).float()

        # if self.args.load_checkpoints:
        #     print("Loading ckpt: {}".format(self.args.load_checkpoints))

        #     transfer_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #     model = transfer_weights(
        #         self.args.load_checkpoints, model, device=transfer_device
        #     )

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size，打印模型参数的数量，便于调试和分析
        print(
            "number of model params",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        return model

    # 获取数据集和数据加载器，这样用到了 data_provider 函数，用于加载指定模式（如训练、验证或测试）的数据集和数据加载器
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # 选择优化器，这里使用 Adam 优化器，并设置学习率
    # Adam算法的核心思想是根据历史梯度的一阶矩估计（均值）和二阶矩估计（方差）自适应地调整学习率。具体来说，Adam算法会计算每个模型参数的梯度的指数移动平均值和指数移动平均值的平方根，然后使用这些平均值来调整每个参数的学习率。这样可以让学习率在训练过程中自适应地适应不同参数的梯度变化情况，从而提高训练的效果。
    # Adam算法还引入了动量的概念，在更新参数时加入动量项，可以在梯度变化较小时平稳更新，在梯度变化较大时加速更新，从而加快收敛速度
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # 这里跟Simmtm不同的是，其会根据任务类型选择损失函数
    def _select_criterion(self):
        # 当为分类任务时，使用交叉熵损失函数（CrossEntropyLoss），适用于多分类问题
        if self.args.task_name == "finetune" and self.args.downstream_task == "classification":
            criterion = nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss")
        # 当为回归任务时，使用均方误差损失函数（MSELoss），适用于回归问题
        else:
            criterion = nn.MSELoss()
            print("Using MSELoss")
        return criterion

    # 预训练部分
    def pretrain(self):

        # data preparation，首先加载训练和验证数据集
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        # 创建保存模型的目录，如果不存在则创建
        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer() # 选择优化器（Adam）
        # model_optim.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        # model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,T_max=self.args.train_epochs)
        
        # 选择学习率调度器，这里使用指数衰减学习率调度器（ExponentialLR），具体来说其会在每个epoch结束时将学习率乘以一个小于1的衰减因子（gamma），从而使学习率逐渐减小
        # 相比于余弦退火调度器，指数衰减调度器的学习率衰减速度更快，适用于需要快速收敛的场景
        model_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=model_optim, gamma=self.args.lr_decay
        )

        # pre-training
        min_vali_loss = None # 初始化最小验证损失为 None，最小验证损失是一个用于监控模型训练过程中的指标，用于判断模型是否收敛
        # 在每个 epoch 中调用 pretrain_one_epoch 和 valid_one_epoch 方法分别进行训练和验证。记录损失并保存最优模型的检查点
        for epoch in range(self.args.train_epochs):
            start_time = time.time() # 记录开始时间，用于后续计算训练耗时

            # current learning rate，获取当前学习率并打印到控制台
            # 这里使用了一个 OneCycleLR 学习率调度器，它会在训练过程中动态调整学习率，先增加到最大值，然后再逐渐减小到最小值
            print("Current learning rate: {:.7f}".format(model_scheduler.get_last_lr()[0]))

            # 调用 self.pretrain_one_epoch 方法执行单个 epoch 的训练。该方法会遍历训练数据加载器（train_loader），计算损失并更新模型参数。返回值 train_loss 是当前 epoch 的平均训练损失
            train_loss = self.pretrain_one_epoch(
                train_loader, model_optim, model_scheduler
            )
            # 调用 self.valid_one_epoch 方法执行验证。该方法会遍历验证数据加载器（vali_loader），计算验证损失但不更新模型参数。返回值 vali_loss 是当前 epoch 的平均验证损失
            vali_loss = self.valid_one_epoch(vali_loader)

            # log and Loss
            end_time = time.time() # 记录结束时间，和开始时间相减得到训练耗时
            # 输出当前 epoch 的训练损失和验证损失
            print(
                "Epoch: {}/{}, Time: {:.2f}, Train Loss: {:.4f}, Vali Loss: {:.4f}".format(
                    epoch + 1,
                    self.args.train_epochs,
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                )
            )

            # 使用 SummaryWriter 记录训练损失和验证损失到 TensorBoard 中，方便后续可视化
            # 这里的 self.writer 是一个 SummaryWriter 实例，用于记录训练过程中的各种指标
            loss_scalar_dict = {
                "train_loss": train_loss,
                "vali_loss": vali_loss,
            }
            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch) # 这里的 self.writer 是一个 SummaryWriter 实例，用于记录训练过程中的各种指标

            # checkpoint saving，保存最优模型检查点
            # 检查当前验证损失是否小于之前的最小验证损失（min_vali_loss）。如果是，则更新 min_vali_loss 并保存当前模型的检查点
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss
                # 输出当前验证损失和最小验证损失的变化情况
                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model epoch{}...".format(
                        min_vali_loss, vali_loss, epoch
                    )
                )
                min_vali_loss = vali_loss

                # 保存当前模型的状态字典（state_dict），包括模型参数和优化器状态
                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    # 保存的模型权重仅包括与编码器相关的参数（如 encoder 或 enc_embedding）。如果模型使用了多 GPU 训练（DataParallel），则移除参数名中的 module. 前缀
                    if "encoder" in k or "enc_embedding" in k:
                        if "module." in k:
                            k = k.replace("module.", "")  # multi-gpu
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.encoder_state_dict,
                }
                # 同过 torch.save 将模型的状态字典保存到指定路径下，文件名为 "ckpt_best.pth"
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))

            # 每隔 10 个 epoch 保存一次模型的检查点，文件名为 "ckpt{epoch + 1}.pth"，便于在训练中断时恢复或进行调试（过程同上）
            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if "encoder" in k or "enc_embedding" in k:
                        if "module." in k:
                            k = k.replace("module.", "")
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.encoder_state_dict,
                }
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

    # 实现了单个训练 epoch 的逻辑，包括前向传播、反向传播和优化器更新
    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):
        train_loss = [] # 用于记录当前 epoch 中每个批次的训练损失
        # 调用 _select_criterion 方法选择损失函数（均方误差、交叉熵损失二选一），用于计算模型预测值与真实值之间的误差
        model_criterion = self._select_criterion()

        self.model.train() # 将模型设置为训练模式，启用 Dropout 和 BatchNorm 等训练时特有的操作
        # Dropout ：正则化技术，用于防止模型过拟合。它通过随机地将一部分神经元的输出置为零，从而减少模型对特定神经元的依赖
        # BatchNorm ：归一化技术，用于加速模型训练和提高模型的稳定性。它通过对每个批次的输入进行标准化，使其均值为 0，方差为 1，从而加速收敛
        
        # 通过遍历训练数据加载器（train_loader），获取每个批次的数据（batch_x、batch_y、batch_x_mark、batch_y_mark）
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            train_loader
        ):
            model_optim.zero_grad() # 清空优化器的梯度缓存，准备进行反向传播

            # 将输入数据 batch_x 和目标数据 batch_y 转换为浮点类型，并移动到指定设备（如 GPU，这个在exp_basic.py中指定）
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            pred_x = self.model(batch_x) # 前向传播，调用模型的前向方法（self.model(batch_x)）生成预测值 pred_x
            diff_loss = model_criterion(pred_x, batch_x) # 使用损失函数计算预测值 pred_x 与输入数据 batch_x 的误差，结果存储在 diff_loss 中
            diff_loss.backward() # 反向传播，计算模型参数的梯度

            model_optim.step() # 更新模型参数，使用优化器（model_optim）根据计算出的梯度更新模型参数
            train_loss.append(diff_loss.item()) # 将当前批次的训练损失添加到 train_loss 列表中

        model_scheduler.step() # 更新学习率，使用学习率调度器（model_scheduler）更新当前学习率
        train_loss = np.mean(train_loss) # 计算当前 epoch 的平均训练损失

        return train_loss

    # 实现了单个验证 epoch 的逻辑，与训练类似，但不进行反向传播
    def valid_one_epoch(self, vali_loader):
        vali_loss = []
        model_criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)
                diff_loss = model_criterion(pred_x, batch_x)
                vali_loss.append(diff_loss.item())

        vali_loss = np.mean(vali_loss)

        return vali_loss

    # 正式训练函数，主要用于训练模型并进行验证和测试，包括数据加载、训练、验证、测试、学习率调整、日志记录以及模型保存
    def train(self, setting):
        # 和预训练一样，首先加载训练、验证数据集，不过这里还需要一个测试数据集
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        # 创建保存模型的目录，如果不存在则创建
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer() # 选择优化器（Adam）
        # 引入早停机制，用于监控验证损失并在验证损失不再下降时提前停止训练，防止过拟合，这里的patience和verbose两个超参数分别为耐心值和启用详细日志输出
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 选择学习率调度器，这里使用 OneCycleLR 学习率调度器，它会在训练过程中动态调整学习率，先增加到最大值，然后再逐渐减小到最小值
        # --------------------------------------------------------------------------
        model_optim = self._select_optimizer() # 奇怪，为什么又选择了一次优化器
        # --------------------------------------------------------------------------
        model_criteria = self._select_criterion() # 选择损失函数（均方误差、交叉熵损失二选一），用于计算模型预测值与真实值之间的误差
        model_scheduler = lr_scheduler.OneCycleLR( # 使用了 OneCycleLR 学习率调度器，适合短期训练任务（具体设计已经解释过了）
            optimizer=model_optim, # 优化器
            steps_per_epoch=len(train_loader), # 每个 epoch 的步数，即训练数据集的批次数
            pct_start=self.args.pct_start, # 学习率调度器的起始百分比，表示在训练的前 pct_start 个 epoch 中逐渐增加学习率
            epochs=self.args.train_epochs, # 总的训练 epoch 数
            max_lr=self.args.learning_rate, # 最大学习率
        )

        # 训练过程
        for epoch in range(self.args.train_epochs):
            iter_count = 0 # 迭代次数计数器
            train_loss = [] # 用于记录当前 epoch 中每个批次的训练损失
            train_loader = tqdm(train_loader, desc="Training") # 显示训练进度条

            print("Current learning rate: {:.7f}".format(model_optim.param_groups[0]['lr'])) # 打印当前学习率，便于监控

            self.model.train() # 和预训练一样，将模型设置为训练模式，启用 Dropout 和 BatchNorm 等训练时特有的操作
            start_time = time.time()

            # 与预训练一致，不在赘述
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1 # 迭代次数加1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)

                # 与预训练方法不同的是，这里的 pred_x 和 batch_y 都是经过了 mask 处理的（不是必须，但为可选项）
                # 这里的 mask 处理是为了在训练过程中随机遮挡一部分输入数据，以增强模型的鲁棒性和泛化能力
                # 具体来说，masking 是一种数据增强技术，通过随机遮挡输入数据的一部分来迫使模型学习到更鲁棒的特征表示
                f_dim = -1 if self.args.features == "MS" else 0
                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                loss = model_criteria(pred_x, batch_y)
                loss.backward()
                model_optim.step()

                # 若 self.args.lradj 设置为 "step"，则调用 adjust_learning_rate 手动调整学习率
                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

                # 将当前批次的损失值（标量）添加到 train_loss 列表中，用于后续计算当前 epoch 的平均训练损失
                train_loss.append(loss.item())

            # 计算当前 epoch 的平均训练损失，这里三个损失函数的计算方式是一样的
            train_loss = np.mean(train_loss)
            vali_loss = self.valid(vali_loader, model_criteria)
            test_loss = self.valid(test_loader, model_criteria)

            end_time = time.time()

            # 将每个 epoch 的训练、验证和测试损失记录到日志文件中，路径为 log.txt，便于后续分析训练过程
            print(
                "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                    epoch + 1,
                    len(train_loader),
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                    test_loss,
                )
            )
            log_path = path + "/" + "log.txt"
            with open(log_path, "a") as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}\n".format(
                        epoch + 1,
                        len(train_loader),
                        end_time - start_time,
                        train_loss,
                        vali_loss,
                        test_loss,
                    )
                )
            
            # 调用早停机制，监控验证损失并在验证损失不再下降时提前停止训练，防止过拟合
            if early_stopping.early_stop:
            early_stopping(vali_loss, self.model, path=path)
                print("Early stopping")
                break
            
            # 如果学习率调度器不是 step，则使用 OneCycleLR 学习率调度器更新学习率
            if self.args.lradj != "step": 
                adjust_learning_rate(model_optim, model_scheduler, epoch + 1, self.args)

        # 在训练结束后，加载最优模型的检查点，路径为 "checkpoint.pth"，并将模型参数加载到当前模型中
        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0")) # 加载模型参数

        self.lr = model_scheduler.get_last_lr()[0] # 获取当前学习率

        return self.model

    # 这块和 train() 方法类似，不过这里是验证模型的性能，不需要进行反向传播和参数更新，不再赘述
    def valid(self, vali_loader, model_criteria):
        vali_loss = []
        self.model.eval()
        vali_loader = tqdm(vali_loader, desc="Validation")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = pred_x.detach().cpu()
                true = batch_y.detach().cpu()

                loss = model_criteria(pred_x, batch_y)
                vali_loss.append(loss.item())

        vali_loss = np.mean(vali_loss)
        self.model.train()

        return vali_loss

    # 测试函数，主要包括数据加载、前向传播、结果收集与处理，以及性能指标的计算和保存
    def test(self):
        # 从数据集中加载测试数据
        test_data, test_loader = self._get_data(flag="test")

        preds = [] # 用于存储模型预测结果
        trues = [] # 用于存储真实值

        # 创建保存测试结果的目录，如果不存在则创建
        folder_path = "./outputs/test_results/{}".format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval() # 将模型设置为评估模式，禁用 Dropout 和 BatchNorm 等训练时特有的操作
        with torch.no_grad(): # 禁用梯度计算，节省内存和加速计算

            # 和训练过程类似，不过和验证部分一样不需要进行反向传播和参数更新
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = pred_x.detach().cpu() # 将预测结果从 GPU 移动到 CPU
                true = batch_y.detach().cpu() # 将真实值从 GPU 移动到 CPU
                # 将两者存储到 preds 和 trues 列表中
                preds.append(pred)
                trues.append(true)

        # 将 preds 和 trues 列表转换为 NumPy 数组，并调整形状（重塑为三维数组），确保预测值和真实值的时间步和特征维度对齐
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # 调用 metric 函数计算模型的性能指标（均方误差和平均绝对误差），并打印结果（这里仅仅是打印了均方误差和平均绝对误差，还有其它三个指标没有展示）
        mae, mse, _, _, _ = metric(preds, trues)
        print(
            "{0}->{1}, mse:{2:.3f}, mae:{3:.3f}".format(
                self.args.input_len, self.args.pred_len, mse, mae
            )
        )
        # 将测试结果追加写入到 score.txt 文件中，便于后续分析
        f = open(folder_path + "/score.txt", "a")
        f.write(
            "{0}->{1}, {2:.3f}, {3:.3f} \n".format(
                self.args.input_len, self.args.pred_len, mse, mae
            )
        )
        f.close()

    # --------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------
    
    # 以下是分类任务的训练、验证、测试函数，和上面的 train() 方法类似，不过这里是分类任务，所以使用了交叉熵损失函数等（CrossEntropyLoss）适用于多分类问题的处理方式
    # 损失函数的选择——回归任务：使用 均方误差（MSELoss） 作为损失函数，适用于连续值预测。损失函数计算模型预测值和真实值之间的平方误差，目标是最小化误差/分类任务：使用 交叉熵损失（CrossEntropyLoss） 作为损失函数，适用于多分类问题。损失函数计算预测类别分布与真实类别分布之间的交叉熵，目标是最大化正确类别的概率
    # 模型输出的处理——回归任务：模型输出为连续值（浮点数），直接用于计算损失。不需要进一步处理预测值，直接与目标值进行比较。/分类任务：模型输出为类别概率分布（通过 softmax 激活函数）。需要通过 torch.argmax 提取预测的类别索引。
    # 目标值的处理——回归任务：目标值为连续值（浮点数），直接用于损失计算。/分类任务：目标值为离散类别索引（整数），需要转换为 long 类型以适配交叉熵损失函数。
    # 性能指标的计算——回归任务：使用 均方误差（MSE） 和 平均绝对误差（MAE） 等指标评估模型性能。/分类任务：使用 准确率（Accuracy） 和 F1 分数（F1 Score） 等指标评估模型性能。
    # 训练流程——回归任务：直接计算损失并更新模型参数。/分类任务：计算损失后，还需要计算预测类别的准确率和 F1 分数。
    # 验证与测试流程——回归任务：验证和测试时，直接计算预测值与目标值的误差。/分类任务：验证和测试时，除了计算损失，还需要计算准确率和 F1 分数。
    # 日志记录与输出——回归任务：记录损失（如 MSE 和 MAE）到日志文件中。/分类任务：记录损失、准确率和 F1 分数到日志文件中。
    # 早停机制的监控指标——回归任务：早停机制通常监控验证损失（如 MSE 或 MAE）。/分类任务：早停机制通常监控验证准确率（-vali_acc，取负值以实现最大化）
    def cls_train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_criteria = self._select_criterion()
        model_scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            train_loss = []
            train_acc = []
            train_f1 = []

            print("Current learning rate: {:.7f}".format(model_optim.param_groups[0]['lr']))

            self.model.train()
            train_loader = tqdm(train_loader)
            start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                loss.backward()
                model_optim.step()
                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                acc = accuracy_score(trues, preds)
                f1 = f1_score(trues, preds, average='macro')

                train_loss.append(loss.item())
                train_acc.append(acc)
                train_f1.append(f1)

            train_loss = np.mean(train_loss)
            train_acc = np.mean(train_acc)
            train_f1 = np.mean(train_f1)

            vali_loss, vali_acc, vali_f1 = self.cls_valid(vali_loader, model_criteria)
            test_loss, test_acc, test_f1 = self.cls_valid(test_loader, model_criteria)
            self.cls_test(write_log=False)

            end_time = time.time()
            print(
                "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | ".format(
                    epoch + 1, len(train_loader), end_time - start_time
                ) +
                "Train Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f} | ".format(
                    train_loss, train_acc, train_f1
                ) +
                "Vali Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f} | ".format(
                    vali_loss, vali_acc, vali_f1
                ) +
                "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}".format(
                    test_loss, test_acc, test_f1
                )
            )
            log_path = path + "/" + "log.txt"
            with open(log_path, "a") as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f}, Acc: {4:.4f}, F1: {5:.4f} | Vali Loss: {6:.7f}, Acc: {7:.4f}, F1: {8:.4f} | Test Loss: {9:.7f}, Acc: {10:.4f}, F1: {11:.4f}\n".format(
                        epoch + 1, len(train_loader), end_time - start_time, train_loss, train_acc, train_f1, vali_loss, vali_acc, vali_f1, test_loss, test_acc, test_f1
                    )
                )

            early_stopping(-vali_acc, self.model, path=path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "step":
                adjust_learning_rate(model_optim, model_scheduler, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def cls_valid(self, vali_loader, model_criteria):
        vali_acc = []
        vali_f1 = []
        vali_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                vali_loss.append(loss.item())
                acc = accuracy_score(trues, preds)
                f1 = f1_score(trues, preds, average='macro')
                vali_acc.append(acc)
                vali_f1.append(f1)
        
        vali_loss = np.mean(vali_loss)
        vali_acc = np.mean(vali_acc)
        vali_f1 = np.mean(vali_f1)

        return vali_loss, vali_acc, vali_f1

    def cls_test(self, write_log=True):
        test_data, test_loader = self._get_data(flag="test")
        model_criteria = self._select_criterion()

        preds_all = []
        trues_all = []
        test_loss = []

        folder_path = "./outputs/test_results/{}".format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                test_loss.append(loss.item())
                preds_all.extend(preds)
                trues_all.extend(trues)

        test_loss = np.mean(test_loss)
        test_acc = accuracy_score(trues_all, preds_all)
        test_f1 = f1_score(trues_all, preds_all, average='macro')

        print(
            "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}".format(
                test_loss, test_acc, test_f1
            )
        )
        if write_log:
            f = open(folder_path + "/score.txt", "a")
            f.write(
                "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}\n".format(test_loss, test_acc, test_f1)
            )
            f.close()