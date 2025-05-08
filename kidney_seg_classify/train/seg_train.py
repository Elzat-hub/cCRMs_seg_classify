import sys
sys.path.append('..')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   #显卡编号 nvidia-smi



from torch import nn
import torch
import numpy as np
import argparse
from monai.losses import FocalLoss, DiceCELoss
from monai.transforms import AsDiscrete
from trainer.seg_trainer import train_epoch, val_epoch, save_checkpoint
from project_utils.model_creater import create_model
from project_utils.data_creater import create_data
from project_utils.Lr_scheduler import build_scheduler
from config.setting import get_config
from project_utils.general import datestr,reproducibility, make_dirs
import matplotlib.pyplot as plt


seed = 3407
train_losses = []
val_losses = []


def get_arguments():
    parser = argparse.ArgumentParser(description='Segmentation with VNet')
    parser.add_argument('--cfg', type=str, 
                        # default='../config/ADC_seg_classify.yaml'
                        # default='../config/CP_seg_classify.yaml'
                        # default='../config/DWI_seg_classify.yaml'
                        # default='../config/EP_seg_classify.yaml'
                        # default='../config/NP_seg_classify.yaml'
                        # default='../config/T1WI_seg_classify.yaml'
                        default='../config/T2WI_seg_classify.yaml'
                        )  #分类分割的参数。 
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume',
                        default=None,
                        type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--classes', type=int, default=2)
    #早停相关参数
    parser.add_argument('--patience', type=int, default=15, help='Number of epochs to wait before early stopping')
    parser.add_argument('--delta', type=float, default=0.0005, help='Minimum change in monitored quantity to qualify as an improvement')

    # dataset specific configs:
    # parser.add_argument('--split', type=float, default=1.0)
    # parser.add_argument('--mode', type=str, default='dino')
    # parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    config = get_config(args)
    model_name = config.MODEL.TYPE
    dataset = config.DATA.DATASET

    args.save = './saved_models/' + model_name + '_checkpoints/' + config.TAG + '_{}_{}'.format(dataset, datestr())
    return args, config

def plot_losses(train_losses, val_losses, tag):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    filename = f'{tag}_loss_curve.png'
    plt.savefig(filename)
    plt.close()

#定义早停
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        # 确保保存路径的目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def main():
    global start_epoch, epoch
    args, config = get_arguments()
    reproducibility(args, seed)
    model, optimizer = create_model(args, config)
    

    # Initialize / load checkpoint
    if args.resume is None:
        make_dirs(args.save)
        start_epoch = 1
        best_acc = 0.

    else:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['acc']
        model.load_state_dict(checkpoint['model_state_dict'])

    # Move to GPU, if available
    if args.cuda:
        model = model.cuda()

    # Custom data loaders
    train_generator, val_generator, test_generator = create_data(args, config)

    # Build learning rate scheduler
    lr_scheduler = build_scheduler(config=config, optimizer=optimizer, n_iter_per_epoch=len(train_generator))
    # scaler = amp.GradScaler()

    # Loss function
    # criterion_1 = nn.BCEWithLogitsLoss()     #分割的loss function
    seg_criterion = DiceCELoss(to_onehot_y=True,       #此处参考Vnet文件，引用了monai的diceloss
                           softmax=True,
                           squared_pred=True,
                           smooth_nr=0.0,
                           smooth_dr=1e-6)
    class_criterion = FocalLoss(gamma=3.0, to_onehot_y=True)   #这是分类的loss function

    post_label = AsDiscrete(to_onehot=args.classes)
    post_pred = AsDiscrete(argmax=True,
                           to_onehot=args.classes)
    # 初始化EarlyStopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, delta=args.delta)

    # Epochs
    for epoch in range(start_epoch, config.TRAIN.EPOCHS + 1):
        train_loss = train_epoch(model=model, data_loader=train_generator, seg_loss_func=seg_criterion, 
                             optimizer=optimizer, epoch=epoch, config=config,
                             lr_scheduler=lr_scheduler, post_label=post_label, post_pred=post_pred)

        val_loss, recent_acc = val_epoch(model=model, data_loader=val_generator,
                                     seg_loss_func=seg_criterion, epoch=epoch, config=config,
                                     post_label=post_label, post_pred=post_pred)
        
        # record loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Compare loss
        is_best = recent_acc > best_acc
        if is_best:
            best_acc = recent_acc

        # Save checkpoint
        save_checkpoint(directory=args.save, model=model, epoch=epoch, config=config, is_best=is_best,
                        optimizer=optimizer, lr_scheduler=lr_scheduler, acc=best_acc)        

        # 检查是否应该早停
        early_stopping(val_loss, model, f'{args.save}/checkpoint_best.pth')  # 使用 args.save 作为目录
        if early_stopping.early_stop:
            print("Early stopping")
            break

        plot_losses(train_losses, val_losses, config.TAG)
if __name__ == '__main__':
    main()
    
