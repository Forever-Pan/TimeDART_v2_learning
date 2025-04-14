# 实现了一个名为 TimeDART 的深度学习模型，主要用于时间序列数据的建模和预测。模型的设计灵活，支持两种主要任务：预训练（pretrain） 和 微调（finetune）
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.TimeDART_EncDec import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    CausalTransformer,
    Diffusion,
    DenoisingPatchDecoder,
    DilatedConvEncoder,
    ClsEmbedding,
    ClsHead,
    OldClsHead,
    ClsFlattenHead,
    ARFlattenHead,
)
from layers.Embed import Patch, PatchEmbedding, PositionalEncoding

# 用于预测的头部模块，负责将编码器的输出映射到目标预测值，通常用于时间序列预测任务的最后阶段，将编码器的高维特征映射到目标预测值
# 将输入张量展平（flatten），然后通过全连接层（Linear）映射到预测长度
# 调整输出张量的维度以匹配 [batch_size, pred_len, num_features]
class FlattenHead(nn.Module):
    def __init__(
        self,
        seq_len: int, # 输入序列的长度
        d_model: int, # 模型的维度
        pred_len: int, # 预测的长度
        dropout: float, # dropout 的比率，用于防止过拟合的丢弃率
    ):
        super(FlattenHead, self).__init__()
        self.pred_len = pred_len
        self.flatten = nn.Flatten(start_dim=-2) # 将输入张量展平，从指定维度开始（这里是 -2），以便后续通过全连接层处理
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len) # 全连接层，用于将展平后的张量映射到预测长度（pred_len）
        self.dropout = nn.Dropout(dropout) # 应用丢弃率，随机丢弃部分神经元以增强模型的泛化能力

    # 定义了模块的前向传播逻辑，接收输入张量 x，并返回预测结果
    # x 的形状为 [batch_size, num_features, seq_len, d_model]，批量大小、特征数量、序列长度和模型维度
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        x = self.flatten(x)  # (batch_size, num_features, seq_len * d_model)，将输入张量展平，将 seq_len 和 d_model 合并为一个维度
        x = self.forecast_head(x)  # (batch_size, num_features, pred_len)，使用全连接层将展平后的张量映射到预测长度
        x = self.dropout(x)  # (batch_size, num_features, pred_len)，随机丢弃部分神经元，增强模型的泛化能力
        x = x.permute(0, 2, 1)  # (batch_size, pred_len, num_features)，调整张量的维度，使其符合时间序列预测的输出格式
        return x

# 回归任务（包含预训练和微调）模型的实现
class Model(nn.Module):
    """
    TimeDART
    """
    # 初始化模型参数（超参数）
    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len # 输入序列的长度

        # For Model Hyperparameters
        self.d_model = args.d_model # 模型的维度
        self.num_heads = args.n_heads # 注意力头的数量
        self.feedforward_dim = args.d_ff # 前馈网络的维度
        self.dropout = args.dropout # dropout 的比率
        self.device = args.device # 设备类型（CPU 或 GPU）
        self.task_name = args.task_name # 任务名称（预训练或微调）
        self.pred_len = args.pred_len # 预测的长度
        self.use_norm = args.use_norm # 是否使用归一化
        # 通道独立性处理，ChannelIndependence 模块用于对输入数据进行通道级别的独立处理
        self.channel_independence = ChannelIndependence(
            input_len=self.input_len,
        )

        # Patch，将时间序列划分为小块，便于后续处理
        self.patch_len = args.patch_len # 每个补丁的长度
        self.stride = args.stride # 步幅
        self.patch = Patch(
            patch_len=self.patch_len,
            stride=self.stride,
        )
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 1 # 计算划分后的序列长度

        # Embedding
        # 嵌入层，将每个 patch 转换为高维特征表示
        self.enc_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            d_model=self.d_model,
        )
        # 位置编码，用于为每个 patch 添加位置信息
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
        )
        # 添加 SOS（Start of Sequence）标记，用于序列的开始
        sos_token = torch.randn(1, 1, self.d_model, device=self.device) # 随机初始化 SOS 标记
        self.sos_token = nn.Parameter(sos_token, requires_grad=True) # 将 SOS 标记设置为可学习参数

        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast( # 添加 SOS 标记并丢弃最后一个补丁
            sos_token=self.sos_token,
        )

        # Encoder (Casual Trasnformer)
        # 编码器，使用因果 Transformer 进行时间序列建模，可选的 Diffusion 模块用于数据增强
        self.diffusion = Diffusion(
            time_steps=args.time_steps, # 扩散时间步数
            device=self.device, # 设备类型
            scheduler=args.scheduler, # 调度器类型
        )
        self.encoder = CausalTransformer( # 因果 Transformer 编码器
            d_model=args.d_model, # 模型的维度
            num_heads=args.n_heads, # 注意力头的数量
            feedforward_dim=args.d_ff, # 前馈网络的维度
            dropout=args.dropout, # dropout 的比率
            num_layers=args.e_layers, # 编码器层的数量
        )
        # self.encoder = DilatedConvEncoder(
        #     in_channels=self.d_model,
        #     channels=[self.d_model] * args.e_layers,
        #     kernel_size=3,
        # )

        # Decoder
        # 若为预训练任务，则使用 DenoisingPatchDecoder 进行去噪处理
        if self.task_name == "pretrain":
            self.denoising_patch_decoder = DenoisingPatchDecoder(
                d_model=args.d_model,
                num_layers=args.d_layers,
                num_heads=args.n_heads,
                feedforward_dim=args.d_ff,
                dropout=args.dropout,
                mask_ratio=args.mask_ratio,
            )

            # 使用 FlattenHead 将高维特征映射到目标预测值，并进行降噪和预测
            self.projection = FlattenHead(
                seq_len=self.seq_len,
                d_model=self.d_model,
                pred_len=args.input_len,
                dropout=args.head_dropout,
            )
            # self.projection = ARFlattenHead(
            #     d_model=self.d_model,
            #     pred_len=args.input_len,
            #     dropout=args.head_dropout,
            # )
            self.projection = ARFlattenHead(
                d_model=self.d_model,
                patch_len=self.patch_len,
                dropout=args.head_dropout,
            )

        # 在微调模式下，直接使用 FlattenHead 生成预测。
        elif self.task_name == "finetune":
            self.head = FlattenHead(
                seq_len=self.seq_len,
                d_model=args.d_model,
                pred_len=args.pred_len,
                dropout=args.head_dropout,
            )

    # pretrain 方法通过归一化、特征提取、降噪解码等步骤，学习时间序列的通用表示。它的设计结合了扩散模型和因果 Transformer，能够有效捕获时间序列的复杂模式，同时通过降噪机制提升模型的鲁棒性
    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        batch_size, input_len, num_features = x.size()
        if self.use_norm: # 若使用归一化，则对输入数据进行归一化处理（归一化的目的是消除不同特征之间的量纲差异，提升模型的训练稳定性）
            # Instance Normalization
            # 首先计算每个特征的均值 (means)和标准差 (stdevs)，然后对输入数据进行归一化处理，使其均值为 0，标准差为 1
            means = torch.mean(
                x, dim=1, keepdim=True
            ).detach()  # [batch_size, 1, num_features], detach from gradient
            x = x - means  # [batch_size, input_len, num_features]
            stdevs = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()  # [batch_size, 1, num_features] ，使用 .detach() 确保均值和标准差不参与梯度计算，避免影响模型的反向传播
            x = x / stdevs  # [batch_size, input_len, num_features]

        # Channel Independence
        # 使用 ChannelIndependence 模块对输入数据进行通道级别的独立处理
        # 这一步将数据从形状 [batch_size, input_len, num_features] 转换为 [batch_size * num_features, input_len, 1]，为后续的 Patch 划分做准备
        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        # Patch
        # 使用 Patch 模块将时间序列划分为小块 (patches)，每个 patch 的长度由模型参数 patch_len 决定
        # 输出形状为 [batch_size * num_features, seq_len, patch_len]，其中 seq_len 是划分后的序列长度
        x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]


    # 编码器处理部分

        # For Casual Transformer
        # 使用 PatchEmbedding 模块将每个 patch 转换为高维特征表示，输出形状为 [batch_size * num_features, seq_len, d_model]
        x_embedding = self.enc_embedding(
            x_patch
        )  # [batch_size * num_features, seq_len, d_model]

        # 使用 AddSosTokenAndDropLast 模块添加 SOS 标记并丢弃最后一个补丁，输出形状为 [batch_size * num_features, seq_len, d_model]
        x_embedding_bias = self.add_sos_token_and_drop_last(
            x_embedding
        )  # [batch_size * num_features, seq_len, d_model]

        # 使用 PositionalEncoding 为每个 patch 添加位置信息，帮助模型捕获序列的时间依赖性
        x_embedding_bias = self.positional_encoding(x_embedding_bias)

        # 使用 CausalTransformer 编码器对输入数据进行编码，输出形状为 [batch_size * num_features, seq_len, d_model]
        x_out = self.encoder(
            x_embedding_bias,
            is_mask=True,
        )  # [batch_size * num_features, seq_len, d_model]


    # 扩散降噪处理部分
    # 扩散
        # Noising Diffusion
        # 使用 Diffusion 模块对输入数据添加噪声，生成噪声版本的 patch (noise_x_patch)
        noise_x_patch, noise, t = self.diffusion(
            x_patch
        )  # [batch_size * num_features, seq_len, patch_len]

        # 使用 PatchEmbedding 模块将噪声 patch 转换为高维特征表示，输出形状为 [batch_size * num_features, seq_len, d_model]
        noise_x_embedding = self.enc_embedding(
            noise_x_patch
        )  # [batch_size * num_features, seq_len, d_model]

        # 使用 PositionalEncoding 为噪声 patch 添加位置信息，输出形状为 [batch_size * num_features, seq_len, d_model]
        noise_x_embedding = self.positional_encoding(noise_x_embedding)

    # 降噪
        # For Denoising Patch Decoder
        # 使用 DenoisingPatchDecoder 对噪声 patch 进行去噪处理，输出形状为 [batch_size * num_features, seq_len, d_model]
        predict_x = self.denoising_patch_decoder(
            query=noise_x_embedding, 
            key=x_out,
            value=x_out,
            is_tgt_mask=True,
            is_src_mask=True,
        )  # [batch_size * num_features, seq_len, d_model]

    # 解码器处理部分

        # For Decoder
        # 将预测结果 reshape 为 [batch_size, num_features, seq_len, d_model]，以便后续处理
        predict_x = predict_x.reshape(
            batch_size, num_features, -1, self.d_model
        )  # [batch_size, num_features, seq_len, d_model]
        # 使用 FlattenHead 将高维特征映射到目标预测值，输出形状为 [batch_size, input_len, num_features]（这里对应的关系看了好久）
        predict_x = self.projection(predict_x)  # [batch_size, input_len, num_features]

        # Instance Denormalization
        # 如果使用了归一化，则还需要对预测结果进行反归一化处理，将其还原到原始数据的尺度
        if self.use_norm:
            predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
                1, input_len, 1
            )  # [batch_size, input_len, num_features]
            predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
                1, input_len, 1
            )  # [batch_size, input_len, num_features]

        return predict_x




    # 微调部分，主要目标是根据输入时间序列生成预测结果
    def forecast(self, x):
        batch_size, _, num_features = x.size()
        if self.use_norm: # 同理，若使用归一化，则对输入数据进行归一化处理（归一化的目的是消除不同特征之间的量纲差异，提升模型的训练稳定性）
            # 首先计算每个特征的均值 (means)和标准差 (stdevs)，然后对输入数据进行归一化处理，使其均值为 0，标准差为 1
            means = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means
            stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach() # 使用 .detach() 确保均值和标准差不参与梯度计算，避免影响模型的反向传播
            x = x / stdevs

        # 使用 ChannelIndependence 模块对输入数据进行通道级别的独立处理，将数据从形状 [batch_size, input_len, num_features] 转换为 [batch_size * num_features, input_len, 1]，为后续的 Patch 划分做准备
        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        # 使用 Patch 模块将时间序列划分为小块 (patches)，每个 patch 的长度由模型参数 patch_len 决定。输出形状为 [batch_size * num_features, seq_len, patch_len]，其中 seq_len 是划分后的序列长度
        x = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

    # 编码器处理部分
        # 嵌入层：使用 PatchEmbedding 模块将每个 patch 转换为高维特征表示，输出形状为 [batch_size * num_features, seq_len, d_model]
        x = self.enc_embedding(x)  # [batch_size * num_features, seq_len, d_model]
        # 位置编码：使用 PositionalEncoding 为每个 patch 添加位置信息，帮助模型捕获序列的时间依赖性
        x = self.positional_encoding(x)  # [batch_size * num_features, seq_len, d_model]

        # 使用 CausalTransformer 编码器对输入数据进行编码，输出形状为 [batch_size * num_features, seq_len, d_model]
        x = self.encoder(
            x,
            is_mask=False,
        )  # [batch_size * num_features, seq_len, d_model]

        # 将预测结果 reshape 为 [batch_size, num_features, seq_len, d_model]
        x = x.reshape(
            batch_size, num_features, -1, self.d_model
        )  # [batch_size, num_features, seq_len, d_model]

        # forecast
        # 使用 FlattenHead （通过 self.head）将高维特征映射到目标预测值，输出形状为 [batch_size, pred_len, num_features]
        x = self.head(x)  # [bs, pred_len, n_vars]

        # denormalization
        # 同理，若使用了归一化，则还需要对预测结果进行反归一化处理，将其还原到原始数据的尺度
        if self.use_norm:
            x = x * (stdevs[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
            x = x + (means[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)

        return x
    
    # 前向传播函数，根据任务名称选择预训练或微调的前向传播逻辑
    def forward(self, batch_x):

        if self.task_name == "pretrain": # 若为预训练任务，则调用 pretrain 函数
            return self.pretrain(batch_x)
        elif self.task_name == "finetune": # 若为微调任务，则调用 forecast 函数并返回预测结果
            dec_out = self.forecast(batch_x)
            return dec_out[:, -self.pred_len: , :]
        else: # 若任务名称不合法，则抛出异常
            raise ValueError("task_name should be 'pretrain' or 'finetune'")




# 分类任务（包含预训练和微调）模型的实现
# 两者区别总结（前回归，后分类）
# 任务目标——目标是根据输入的时间序列生成连续的预测值。输出是一个形状为 [batch_size, pred_len, num_features] 的张量，其中 pred_len 是预测的时间步数，num_features 是特征数量/目标是根据输入的时间序列生成分类结果。输出是一个形状为 [batch_size, num_classes] 的张量，其中 num_classes 是分类的类别数量
# 输入数据处理——使用 ChannelIndependence 模块对输入数据进行通道级别的独立处理，将数据从 [batch_size, input_len, num_features] 转换为 [batch_size * num_features, input_len, 1]。使用 Patch 模块将时间序列划分为小块（patches），每个 patch 的长度由 patch_len 决定，输出形状为 [batch_size * num_features, seq_len, patch_len]/使用 ClsEmbedding 模块直接对输入数据进行嵌入处理，将其转换为高维特征表示，输出形状为 [batch_size, seq_len, d_model]。不需要 ChannelIndependence 和 Patch 模块，因为分类任务不需要对输入数据进行通道独立性处理或划分为小块
# 嵌入与位置编码——使用 PatchEmbedding 模块将每个 patch 转换为高维特征表示。使用 PositionalEncoding 为每个 patch 添加位置信息，帮助模型捕获时间序列的时间依赖性。/使用 ClsEmbedding 模块直接对整个时间序列进行嵌入。同样使用 PositionalEncoding 添加位置信息，但嵌入的输入是整个时间序列，而不是划分后的 patches。
# 编码器——都使用 CausalTransformer 作为编码器，提取时间序列的上下文特征。编码器的输出形状为 [batch_size * num_features, seq_len, d_model]（Model）或 [batch_size, seq_len, d_model]（ClsModel）。
# 解码器——在预训练任务中，使用 DenoisingPatchDecoder 进行降噪处理。在微调任务中，使用 FlattenHead 将编码器的高维特征映射到目标预测值，输出形状为 [batch_size, pred_len, num_features]/在预训练任务中，使用 DenoisingPatchDecoder 进行降噪处理。在微调任务中，使用 OldClsHead 将编码器的高维特征映射到分类结果，输出形状为 [batch_size, num_classes]。
# 归一化与反归一化——在回归任务中，使用归一化（Instance Normalization）对输入数据进行标准化处理（均值为 0，标准差为 1）。在预测结果生成后，使用反归一化将预测值还原到原始数据的尺度。/分类任务中不需要归一化和反归一化，因为分类结果是离散的类别标签，而不是连续值。
# 输出——输出是时间序列的预测值，形状为 [batch_size, pred_len, num_features]。适用于时间序列预测任务，如未来值的回归预测。/输出是分类结果，形状为 [batch_size, num_classes]。适用于时间序列分类任务，如根据时间序列判断类别。
    def __init__(self, args):
        super(ClsModel, self).__init__()
        self.input_len = args.input_len

        # For Model Hyperparameters
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.feedforward_dim = args.d_ff
        self.dropout = args.dropout
        self.device = args.device
        self.task_name = args.task_name
        self.num_classes = args.num_classes

        # Patch
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 2
        padding = self.seq_len * self.stride - self.input_len

        # Embedding
        self.enc_embedding = ClsEmbedding(
            num_features=args.enc_in,
            d_model=args.d_model,
            kernel_size=args.patch_len,
            stride=args.stride,
            padding=padding,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
        )

        sos_token = torch.randn(1, 1, self.d_model, device=self.device)
        self.sos_token = nn.Parameter(sos_token, requires_grad=True)

        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(
            sos_token=self.sos_token,
        )

        # Encoder (Casual Trasnformer)
        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            device=self.device,
            scheduler=args.scheduler,
        )
        self.encoder = CausalTransformer(
            d_model=args.d_model,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            dropout=args.dropout,
            num_layers=args.e_layers,
        )
        # self.encoder = DilatedConvEncoder(
        #     in_channels=self.d_model,
        #     channels=[self.d_model] * args.e_layers,
        #     kernel_size=3,
        # )

        # Decoder
        if self.task_name == "pretrain":
            self.denoising_patch_decoder = DenoisingPatchDecoder(
                d_model=args.d_model,
                num_layers=args.d_layers,
                num_heads=args.n_heads,
                feedforward_dim=args.d_ff,
                dropout=args.dropout,
                mask_ratio=args.mask_ratio,
            )

            self.projection = ClsFlattenHead(
                seq_len=self.seq_len,
                d_model=self.d_model,
                pred_len=args.input_len,
                num_features=args.c_out,
                dropout=args.head_dropout,
            )

        elif self.task_name == "finetune":
            self.head = OldClsHead(
                seq_len=self.seq_len,
                d_model=args.d_model,
                num_classes=args.num_classes,
                dropout=args.head_dropout,
            )

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        # Instance Normalization
        # batch_size, input_len, num_features = x.size()
        # means = torch.mean(
        #     x, dim=1, keepdim=True
        # ).detach()  # [batch_size, 1, num_features], detach from gradient
        # x = x - means  # [batch_size, input_len, num_features]
        # stdevs = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        # ).detach()  # [batch_size, 1, num_features]
        # x = x / stdevs  # [batch_size, input_len, num_features]

        # For Casual Transformer
        x_embedding = self.enc_embedding(
            x
        )  # [batch_size, seq_len, d_model]
        x_embedding_bias = self.add_sos_token_and_drop_last(
            x_embedding
        )  # [batch_size, seq_len, d_model]
        x_embedding_bias = self.positional_encoding(x_embedding_bias)
        x_out = self.encoder(
            x_embedding_bias,
            is_mask=True,
        )  # [batch_size, seq_len, d_model]

        # Noising Diffusion
        noise_x_patch, noise, t = self.diffusion(
            x
        )  # [batch_size, seq_len, patch_len]
        noise_x_embedding = self.enc_embedding(
            noise_x_patch
        )  # [batch_size, seq_len, d_model]
        noise_x_embedding = self.positional_encoding(noise_x_embedding)

        # For Denoising Patch Decoder
        predict_x = self.denoising_patch_decoder(
            query=noise_x_embedding,
            key=x_out,
            value=x_out,
            is_tgt_mask=True,
            is_src_mask=True,
        )  # [batch_size, seq_len, d_model]

        # For Decoder
        predict_x = self.projection(predict_x)  # [batch_size, input_len, num_features]

        # Instance Denormalization
        # predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
        #     1, input_len, 1
        # )  # [batch_size, input_len, num_features]
        # predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
        #     1, input_len, 1
        # )  # [batch_size, input_len, num_features]

        return predict_x

    def forecast(self, x):
        # batch_size, _, num_features = x.size()
        # means = torch.mean(x, dim=1, keepdim=True).detach()
        # x = x - means
        # stdevs = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        # ).detach()
        # x = x / stdevs

        x = self.enc_embedding(x)  # [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)  # [batch_size, seq_len, d_model]
        x = self.encoder(
            x,
            is_mask=False,
        )  # [batch_size, seq_len, d_model]
        # forecast
        x = self.head(x)  # [bs, num_classes]

        return x
    
    def forward(self, batch_x):

        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            return self.forecast(batch_x)
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")