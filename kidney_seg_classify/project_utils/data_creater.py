from torch.utils.data import DataLoader
import sys
sys.path.append('.')  # 将当前目录添加到路径中
from data_loader.ADC import ADC
from data_loader.CP import CP
from data_loader.DWI import DWI
from data_loader.EP import EP   
from data_loader.NP import NP
from data_loader.T1WI import T1WI
from data_loader.T2WI import T2WI

data_list = ['ADC','CP','DWI','EP','NP','T1WI','T2WI']   

    
def create_data(args, config):
    global train_loader, val_loader, test_loader
    params = {'batch_size': config.DATA.BATCH_SIZE,
              'shuffle': True,
              'num_workers': config.DATA.NUM_WORKERS}
    # split_percent = config.DATA.SPLIT
    # mode = args.mode
    # seed = args.seed
    dataset_name = config.DATA.DATASET
    assert dataset_name in data_list, "Can't find dataset in the list!"

    if dataset_name == 'ADC':
        train_loader = ADC(mode='train')
        val_loader = ADC(mode='val')
        test_loader = ADC(mode='test')
    elif dataset_name == 'CP':
        train_loader = CP(mode='train')
        val_loader = CP(mode='val')
        test_loader = CP(mode='test')
    elif dataset_name == 'DWI':
        train_loader = DWI(mode='train')
        val_loader = DWI(mode='val')
        test_loader = DWI(mode='test')
    elif dataset_name == 'EP':
        train_loader = EP(mode='train')
        val_loader = EP(mode='val')
        test_loader = EP(mode='test')
    elif dataset_name == 'NP':
        train_loader = NP(mode='train')
        val_loader = NP(mode='val')
        test_loader = NP(mode='test')
    elif dataset_name == 'T1WI':
        train_loader = T1WI(mode='train')
        val_loader = T1WI(mode='val')
        test_loader = T1WI(mode='test')
    elif dataset_name == 'T2WI':
        train_loader = T2WI(mode='train')
        val_loader = T2WI(mode='val')
        test_loader = T2WI(mode='test')

        
    train_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)
    test_generator = DataLoader(test_loader, batch_size=params['batch_size'], shuffle=config.TEST.SHUFFLE, num_workers=params['num_workers'])
    # test_generator = DataLoader(test_loader, **params)

    return train_generator, val_generator, test_generator
