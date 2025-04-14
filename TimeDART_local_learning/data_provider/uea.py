import os # 文件和目录操作
import numpy as np # 数值计算
import pandas as pd # 数据处理
import torch # Pytorch 核心库


# 和PatchTST一样，用于时间序列数据处理为适合模型输入的格式，如对变长序列进行填充（padding），以确保批次中的所有样本具有相同的长度，同时生成相应的掩码（mask）以标记填充位置
# data 包含多个元组，每个元组包含一个特征张量(seq_length, feat_dim)和一个标签张量(num_labels,)
def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """
    # 获取批次大小（batch_size）并将输入的样本列表解包为特征列表 features 和标签列表 labels
    batch_size = len(data)
    features, labels = zip(*data)

    # 计算每个样本的序列长度，并根据最大长度 max_len 对特征进行填充
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None: # max_len is None时，使用当前批次中最长的序列长度
        max_len = max(lengths)
    
    # 创建一个全零的张量 X，形状为 (batch_size, max_len, feat_dim)，用于存储填充后的特征
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    # 将每个样本的特征填充到 X 中，填充长度为 max_len，即序列长度超过 max_len 的部分将被截断，不足的部分用 0 填充
    for i in range(batch_size): 
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)，将标签列表转换为张量
    # 创建一个全零的张量 padding_masks，形状为 (batch_size, padded_length)，用于存储填充掩码
    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks

# 创建一个填充掩码，形状为 (batch_size, max_len)，用于标记填充位置
def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

# 数据归一化器 Normalizer 类，用于对数据进行标准化或最小-最大缩放
# 知识点提及：归一化是数据预处理中的重要步骤，目的是将数据缩放到某个范围或标准化为零均值和单位方差，以便更好地适配机器学习模型
class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """
    # 初始化方法，接受归一化类型和可选的均值、标准差、最小值和最大值
    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    # 归一化方法，根据指定的归一化类型对输入数据进行处理，其中self已经在init方法中定义，df为输入数据框（dataframe）
    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        # 标准化，先计算dataframe的均值和标准差，然后进行归一化处理
        if self.norm_type == "standardization":
            if self.mean is None: # 如果均值和标准差未提供，则计算它们
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)
        # 最小-最大缩放，先计算dataframe的最大值和最小值，然后进行归一化处理
        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)
        # 每个样本的标准差归一化，按样本分组计算均值和标准差，然后进行归一化处理
        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')
        # 每个样本的最小-最大缩放，按样本分组计算最大值和最小值，然后进行归一化处理
        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

# 通过线性插值法填充 pandas.Series 中的缺失值（NaN）
def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    # 如果y中存在缺失值，则使用线性插值法进行填充（并且可以向前或向后扩展）
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y

# 对 pandas.Series 进行下采样
# 知识点提及：下采样是指将数据的采样频率降低，例如从每秒采样到每分钟采样
def subsample(y, limit=256, factor=2): # 默认限制为256，因子为2
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    # 如果y的长度超过限制（256），则使用指定的因子（2）进行下采样
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y
