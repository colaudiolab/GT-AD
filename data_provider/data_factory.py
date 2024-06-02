from data_provider.data_loader import UCRSegLoader,NIPS_TS_WaterSegLoader,NIPS_TS_SwanSegLoader,PSMSegLoader,MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader
from torch.utils.data import DataLoader

data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'NIPS_TS_Swan':NIPS_TS_SwanSegLoader,
    'NIPS_TS_GECCO':NIPS_TS_WaterSegLoader,
    'UCR':UCRSegLoader
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'train':
        shuffle_flag = True
        drop_last = False
    else:
        shuffle_flag = False
        drop_last = False

    data_set = Data(
        root_path=args.root_path,
        win_size=args.win_size,
        picture=args.picture,
        flag=flag,)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

