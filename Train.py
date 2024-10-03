# -*- coding: utf-8 -*-
import os
import torch
import monai
import logging
import numpy as np
from skimage import io
from datetime import datetime
# from SE_Progress import setup_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Test_data.Model.Network import DSCNet_pro
from Loss import cross_loss
from MyDataloader import Dataloader, Dataloader_test

# setup_seed(1000)

import warnings

warnings.filterwarnings("ignore")


# Use <AverageMeter> to calculate the mean in the process
class AverageMeter(object):

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
        self.avg = self.sum / self.count


# One epoch in training process
def train_epoch(model, loader, optimizer, criterion, dice_calculation, epoch, n_epochs, args, iter_task_progress):
    losses = AverageMeter()
    dice = 0.
    model.train()
    task_train_epoch = iter_task_progress.add_task("", total=len(loader))
    for batch_idx, (image, label) in enumerate(loader):
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        optimizer.zero_grad()
        model.zero_grad()
        output = model(image)

        loss = criterion(label, output)
        dice = dice_calculation(batch_idx, args.batch_size, dice, label, output)

        losses.update(loss.data, label.size(0))

        loss.backward()
        optimizer.step()

        res = "\t".join([
            "Epoch: [%d/%d]" % (epoch + 1, n_epochs),
            "Iter: [%d/%d]" % (batch_idx + 1, len(loader)),
            "Lr: [%f]" % (optimizer.param_groups[0]["lr"]),
            "Loss %f" % (losses.avg),
            "Dice %f" % (dice),
        ])
        print(res)

        description = "[bold yellow]loss %f dice %f" % (losses.avg, dice)
        iter_task_progress.update(task_train_epoch, description=description, advance=1)

    iter_task_progress.stop_task(task_train_epoch)
    iter_task_progress.update(task_train_epoch, visible=False)
    return losses.avg, dice


# Generate the log
def Get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# Train process
def Train_net(net, args, detail_task_progress, iter_task_progress, pred_task_progress):
    dice_mean, dice_m = 0, 0

    # Determine if trained parameters exist
    if not args.if_retrain and os.path.exists(
            os.path.join(args.Dir_Weights, args.model_name)):
        net.load_state_dict(
            torch.load(os.path.join(args.Dir_Weights, args.model_name)))
        # print(os.path.join(args.Dir_Weights, args.model_name))
    if torch.cuda.is_available():
        net = net.cuda()

    # Load dataset
    train_dataset = Dataloader(args)
    train_dataloader = DataLoader(train_dataset,
                                  # num_workers=1,
                                  # pin_memory=True,
                                  batch_size=args.batch_size,
                                  shuffle=True)
    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=args.lr,
                                  betas=(0.9, 0.95))

    scheduler = ReduceLROnPlateau(optimizer,
                                  mode="min",
                                  factor=0.8,
                                  patience=10)
    criterion = cross_loss()
    dice_calculation = dice_coef()

    dt = datetime.today()
    log_name = (str(dt.date()) + "_" + str(dt.time().hour) + "_" +
                str(dt.time().minute) + "_" + str(dt.time().second) + "_" +
                args.log_name)

    writer = SummaryWriter(args.Dir_Log + log_name)   # todo

    # Main train process
    task_train = detail_task_progress.add_task("", total=args.n_epochs)   # todo
    for epoch in range(args.start_train_epoch, args.n_epochs):
        loss, dice_tran = train_epoch(net, train_dataloader, optimizer, criterion, dice_calculation, epoch,
                           args.n_epochs, args, iter_task_progress)
        torch.save(net.state_dict(), os.path.join(args.Dir_Weights, args.model_name))

        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('Dice', dice_tran, epoch)

        if epoch >= args.start_verify_epoch and epoch % args.verify_step == 0:   # args.start_verify_epoch
            net.load_state_dict(
                torch.load(os.path.join(args.Dir_Weights, args.model_name)))
            # The validation set is selected according to the task
            dice_mean = predict_monai(net, args.Image_Te_txt, args.save_path, args, pred_task_progress)
            if dice_mean > dice_m:
                dice_m = dice_mean
                torch.save(
                    net.state_dict(),
                    os.path.join(args.Dir_Weights, args.model_name_max),
                )
            writer.add_scalar('Val_Dice', dice_mean, epoch)
        description = "[bold white]loss:{:.6f}, dice:{:.4f}, dice_max:{:.4f}".format(loss, dice_mean, dice_m)
        detail_task_progress.update(task_train, description=description, advance=1)
    detail_task_progress.stop_task(task_train)
    detail_task_progress.update(task_train, visible=False)


def read_file_from_txt(txt_path):  # 从txt里读取数据
    files = []
    for line in open(txt_path, "r"):
        files.append(line.strip())
    return files


def reshape_img(image, y, x):
    out = np.zeros([y, x], dtype=np.float32)
    out[0:image.shape[0], 0:image.shape[1]] = image[0:image.shape[0],
                                                    0:image.shape[1]]
    return out


# Predict process

def predict_monai(model, image_dir, save_path, args, pred_task_progress, if_save=False):
    # print("Predict test data")
    model.eval()

    sw_batch_size, overlap = args.batch_size, 0.5
    patch_size = (args.ROI_shape, args.ROI_shape)
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
        cache_roi_weight_map=True,
    )

    dice_calculation = dice_coef()
    dice = 0.

    test_dataset = Dataloader_test(args)

    batch_size = 1 if if_save else args.batch_size
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
    task = pred_task_progress.add_task("", total=len(test_dataloader))

    for batch_idx, (image, label, name) in enumerate(test_dataloader):
        # print(name)
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        with torch.no_grad():
            pred = inferer(image, model)
            dice = dice_calculation(batch_idx, batch_size, dice, label, pred)

            if if_save:
                output = pred.data.cpu().numpy().astype(dtype=np.float32)
                output = np.where(output > 0.5, 255., 0).astype(np.uint8)
                print('save_path',save_path)
                print('name',name)
                io.imsave(save_path +str(name) + '.png', output[0][0])

        description = ("Now Predicting Dice is %.5f" % (dice))
        pred_task_progress.update(task, description=description, advance=1)

    pred_task_progress.stop_task(task)
    pred_task_progress.update(task, visible=False)
    return dice


def Create_files(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.save_path_max):
        os.mkdir(args.save_path_max)


def Predict_Network(net, args, pred_task_progress):
    if torch.cuda.is_available():
        net = net.cuda()
    try:
        net.load_state_dict(
            torch.load(os.path.join(args.Dir_Weights, args.model_name_max)))
        # print(os.path.join(args.Dir_Weights, args.model_name_max))
    except:
        # print(
        #     "Warning 100: No parameters in weights_max, here use parameters in weights"
        # )
        net.load_state_dict(
            torch.load(os.path.join(args.Dir_Weights, args.model_name)))
        # print(os.path.join(args.Dir_Weights, args.model_name))
    # predict(net, args.Image_Te_txt, args.save_path_max, args, pred_task_progress)
    dice_mean = predict_monai(net, args.Image_Te_txt, args.save_path_max, args, pred_task_progress, if_save=True)
    print(dice_mean)


def Train(args, detail_task_progress, iter_task_progress, pred_task_progress):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DSCNet_pro(
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        kernel_size=args.kernel_size,
        extend_scope=args.extend_scope,
        if_offset=args.if_offset,
        device=device,
        number=args.n_basic_layer,
        dim=args.dim,
    )
    Create_files(args)
    if not args.if_onlytest:
        Train_net(net, args, detail_task_progress, iter_task_progress, pred_task_progress)
        Predict_Network(net, args, pred_task_progress)
    else:
        Predict_Network(net, args, pred_task_progress)