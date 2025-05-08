import datetime
import os
import time
import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from timm.utils import AverageMeter
import numpy as np
import config.setting
from project_utils.general import prepare_input
from project_utils.general import get_grad_norm
import torch.nn.functional as F

class AverageMeterEval(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)

dice_acc = DiceMetric(include_background=True,
                      reduction=MetricReduction.MEAN,
                      get_not_nans=True)   

def train_epoch(config, model, data_loader, seg_loss_func, optimizer, epoch, lr_scheduler, post_label=None, post_pred=None):
    model.train()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    dice_acc.reset()

    start = time.time()
    end = time.time()
    for idx, (image, seg_lab, image_name, seg_lab_name) in enumerate(data_loader):
        data, seg_target = image.cuda(0), seg_lab.cuda(0)
        
        seg_logits = model(data)
        seg_loss = seg_loss_func(seg_logits, seg_target)

        optimizer.zero_grad()
        seg_loss.backward()
        
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())

        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        loss_meter.update(seg_loss.item(), data.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            print(f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                  f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                  f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                  f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                  f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                  f'mem {memory_used:.0f}MB')

    return loss_meter.avg


def val_epoch(model, data_loader, epoch, config, seg_loss_func, post_label=None, model_infer=None, post_pred=None):
    model.eval()
    dice = 0.0
    loss_meter = AverageMeter()
    dice_acc.reset()
          
    print("==== Validation ====", config.MODEL.TYPE, "====", config.DATA.DATASET)
    with torch.no_grad():
        for idx, (image, seg_lab, image_name, seg_lab_name) in enumerate(data_loader):
            data, seg_target = image.cuda(0), seg_lab.cuda(0)
            if model_infer is not None:
               seg_logits = model_infer(data)
            else:   
                seg_logits = model(data)

            val_labels_list = decollate_batch(seg_target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(seg_logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            seg_loss = seg_loss_func(seg_logits, seg_target)
            loss_meter.update(seg_loss.item(), data.size(0))

            # Dice
            dice_acc(y_pred=val_output_convert, y=val_labels_convert)
            seg_acc = dice_acc.aggregate()
            seg_avg_acc = seg_acc[0].detach().cpu().numpy()
            dice += float(seg_avg_acc)

        seg_avg_acc = dice / len(data_loader)  # 验证过程中的分割平均准确率
        print(f'VAL EPOCH {epoch}:\t Segment_ACC {seg_avg_acc:.4f}\t Loss {loss_meter.avg:.4f}')  
 
    return loss_meter.avg, seg_avg_acc

# def val_epoch(model, data_loader, epoch, config, seg_loss_func, post_label=None, model_infer=None, post_pred=None):
#     model.eval()
#     loss_meter = AverageMeter()
#     dice_acc.reset()
          
#     print("==== Validation ====", config.MODEL.TYPE, "====", config.DATA.DATASET)
#     with torch.no_grad():
#         for idx, (image, seg_lab, image_name, seg_lab_name) in enumerate(data_loader):
#             data, seg_target = image.cuda(0), seg_lab.cuda(0)
#             if model_infer is not None:
#                seg_logits = model_infer(data)
#             else:   
#                 seg_logits = model(data)

#             val_labels_list = decollate_batch(seg_target)
#             val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
#             val_outputs_list = decollate_batch(seg_logits)
#             val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

#             seg_loss = seg_loss_func(seg_logits, seg_target)
#             loss_meter.update(seg_loss.item(), data.size(0))

#             # 累积Dice分数
#             dice_acc(y_pred=val_output_convert, y=val_labels_convert)

#         # 计算整个epoch的平均Dice分数
#         epoch_dice = dice_acc.aggregate().item()
        
#         print(f'VAL EPOCH {epoch}:\t Segment_ACC (Dice) {epoch_dice:.4f}\t Loss {loss_meter.avg:.4f}')  
 
#     return loss_meter.avg, epoch_dice

def save_checkpoint(directory, model, epoch, config, acc, is_best=False,
                    lr_scheduler=None, optimizer=None, name=None):
    """
    Saves checkpoint at a certain global step during training.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    ckpt_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'acc': acc
    }

    if name is None:
        name = "{}_segment_last.pth.tar".format(config.DATA.DATASET)

    torch.save(ckpt_dict, os.path.join(directory, name))
    if is_best:
        name = "{}_segment_best.pth.tar".format(config.DATA.DATASET)
        torch.save(ckpt_dict, os.path.join(directory, name))