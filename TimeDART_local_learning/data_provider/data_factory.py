# 从 data_provider.data_loader 导入多个数据加载类
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Physio, Dataset_PEMS, Dataset_Epilepsy
# 导入 collate_fn 函数
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

# 将数据集名称映射到相应的数据加载类
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Electricity': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Weather': Dataset_Custom,
    'ECL': Dataset_Custom,
    'ILI': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'HAR': Dataset_Physio,
    'EEG': Dataset_Physio,
    'PEMS03': Dataset_PEMS,
    'PEMS04': Dataset_PEMS,
    'PEMS07': Dataset_PEMS,
    'PEMS08': Dataset_PEMS,
    'Epilepsy': Dataset_Epilepsy,
}

# 作为通用的数据加载器函数，接收参数 args（包含数据加载和任务相关的参数，例如数据集名称、批量大小、输入长度等）和 flag（指定当前数据加载的阶段），并根据不同的任务类型和数据集名称返回相应的数据集和数据加载器
def data_provider(args, flag):
    Data = data_dict[args.data] # 获取数据集类

    timeenc = 0 if args.embed != 'timeF' else 1 # 根据 args.embed 设置时间编码

    # 如果flag为'test'或'val'，则不打乱数据（shuffle_flag=False），默认丢弃最后一个批次（drop_last=True）
    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = True
        # 如果任务类型为异常检测或分类，批量大小为 args.batch_size；否则，批量大小为 1
        if args.downstream_task == 'anomaly_detection' or args.downstream_task == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation，一般是用于测试
        freq = args.freq
    else: # 如果flag为'train'，则打乱数据（shuffle_flag=True），默认丢弃最后一个批次（drop_last=True）
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    # 根据args.downstream_task 的值，对不同任务类型进行处理，最后返回相应的数据集和数据加载器
    if args.downstream_task == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path, # 数据集根目录
            win_size=args.input_len, # 窗口大小
            flag=flag, # 当前数据加载的阶段（训练、验证或测试）
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
    elif args.downstream_task == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.input_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        print(flag, len(data_set), len(data_loader))
        return data_set, data_loader
