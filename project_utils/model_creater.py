import torch.optim as optim

from models.Unet3d import UNet3D
from models.Vnet import VNet, VNetLight

model_list = ['UNET3D', 'VNET', 'VNET2']


def create_model(args,config):
    global model, optimizer
    model_name = config.MODEL.TYPE
    assert model_name in model_list
    optimizer_name = config.TRAIN.OPTIMIZER.NAME
    lr = config.TRAIN.BASE_LR
    in_channels = config.IN_CHANNELS
    num_classes = config.MODEL.NUM_CLASSES
    dropout_rate = config.TRAIN.DROPOUT_RATE
    image_size = config.DATA.IMG_SIZE
    # dropout_rate = args.TRAIN.DROPOUT_RATE
    weight_decay = 1e-5
    print("===== Building Model ===== " + model_name + " =====")

    if model_name == 'UNET3D':
        model = UNet3D(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    elif model_name == 'VNET':
        model = VNet(in_channels=in_channels, elu=False, classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'VNET2':
        model = VNetLight(in_channels=in_channels, elu=False, classes=num_classes)

    print("===== " + model_name + ' ===== number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
