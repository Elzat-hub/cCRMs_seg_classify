import math
import os
import shutil
import pickle
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def save_list(name, list):
    with open(name, "wb") as fp:
        pickle.dump(list, fp)


def load_list(name):
    with open(name, "rb") as fp:
        list_file = pickle.load(fp)
    return list_file


def prepare_input(input_tuple, inModalities=-1, inChannels=-1, cuda=True, args=None):
    global input_tensor, target
    if args is not None:
        modalities = args.inModalities
        channels = args.inChannels
        in_cuda = args.cuda
    else:
        modalities = inModalities
        channels = inChannels
        in_cuda = cuda
    if modalities == 4:
        if channels == 4:
            img_1, img_2, img_3, img_4, target = input_tuple
            input_tensor = torch.cat((img_1, img_2, img_3, img_4), dim=1)
        elif channels == 3:
            # t1 post constast is ommited
            img_1, _, img_3, img_4, target = input_tuple
            input_tensor = torch.cat((img_1, img_3, img_4), dim=1)
        elif channels == 2:
            # t1 and t2 only
            img_1, _, img_3, _, target = input_tuple
            input_tensor = torch.cat((img_1, img_3), dim=1)
        elif channels == 1:
            # t1 only
            input_tensor, _, _, target = input_tuple
    if modalities == 3:
        if channels == 3:
            img_1, img_2, img_3, target = input_tuple
            input_tensor = torch.cat((img_1, img_2, img_3), dim=1)
        elif channels == 2:
            img_1, img_2, _, target = input_tuple
            input_tensor = torch.cat((img_1, img_2), dim=1)
        elif channels == 1:
            input_tensor, _, _, target = input_tuple
    elif modalities == 2:
        if channels == 2:
            img_t1, img_t2, target = input_tuple

            input_tensor = torch.cat((img_t1, img_t2), dim=1)

        elif channels == 1:
            input_tensor, _, target = input_tuple
    elif modalities == 1:
        input_tensor, target = input_tuple

    if in_cuda:
        input_tensor, target = input_tensor.cuda(), target.cuda()

    return input_tensor, target


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDecaying learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def datestr():
    now = time.gmtime()
    return '{:02}_{:02}_{:02}_{:02}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def reproducibility(args, seed):
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    # FOR FASTER GPU TRAINING WHEN INPUT SIZE DOESN'T VARY
    # LET'S TEST IT
    cudnn.benchmark = True


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    print(input.size(), target.size())
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class Visualizer:
    def __init__(self, class_names, class_colors):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        self.class_names = class_names
        self.n_classes = class_names
        self.class_colors = class_colors

    def draw_roc_auc(self, y_true, y_pred, title, x_label="False Positive Rate", y_label="True Positive Rate"):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i, :, :, :].view(-1).detach().cpu().numpy(), y_pred[:, i+1, :, :, :].view(-1).detach().cpu().numpy())
            roc_auc[i] = auc(fpr[i], tpr[i])

        for i, color in zip(range(self.n_classes), self.class_colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})".format(self.class_names, roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc="lower right")
        # plt.show()
        return roc_auc

def adjust_learning_rate_cosine(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.nEpochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm