import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import sys
sys.path.append("..")
import argparse

import torch.nn.functional as F

from timm.utils import accuracy, AverageMeter
import time
import argparse
import datetime
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils_st import load_checkpoint, save_checkpoint, save_max_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from performance import performances_val

from apex import amp

def parse_option():
    parser = argparse.ArgumentParser('Conv-MLP ImageNet training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--frame-len', type=int, help="frame length for single video")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    loss = F.cross_entropy(logit, truth, reduce=is_average)
    return loss

def main(config):
    val_name = 'msu'
    data_loader_train, data_loader_val = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    model = build_model(config)

    model.cuda()
    # logger.info(str(model))

    optimizer = build_optimizer(config, model)

    model = torch.nn.DataParallel(model)

    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = softmax_cross_entropy_criterion

    global queue
    queue = torch.randn(config.DATA.BATCH_SIZE*config.DATA.FRAME_LEN*3, 512)
    queue = F.normalize(queue, p=1, dim=1)
    queue = queue.cuda()
    global fea_center

    max_accuracy = 0.5
    max_epoch = 0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        hter, auc, acc = validate(config, data_loader_val, model, val_name)
        logger.info(f"HTER of the network on the oulu test images: {hter:.4f} with AUC: {auc:.4f} and ACC: {acc:.4f}.")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler)

        hter, auc, acc = validate(config, data_loader_val, model, val_name)
        logger.info(f"HTER of the network on the test images: {hter:.4f} with AUC: {auc:.4f} and ACC: {acc:.4f}.")
        
        if hter < max_accuracy:
            max_accuracy = hter
            max_epoch = epoch
            save_max_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        max_accuracy = min(max_accuracy, hter)
        logger.info(f"Minimal HTER of the network on the test images: {max_accuracy:.4f} with epoch #{max_epoch}.")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        samples = samples.view(config.DATA.BATCH_SIZE*config.DATA.FRAME_LEN*3, 9, 3, 48, 48)
        targets = targets.cuda(non_blocking=True)
        # targets = targets.view(config.DATA.BATCH_SIZE*3)
        # print("target shape :",targets.shape)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        # outputs, feas = model(samples)
        outputs, feas = model(samples)
        # feas = feas.clone()
        # feas = feas.detach()
        # outputs = outputs.detach()
        targets = targets.view(feas.shape[0])
        # print("outputs shape: ",outputs.shape)
        mask_pos = targets > 0
        mask_neg = targets == 0
        fea_pos = feas[mask_pos]
        fea_neg = feas[mask_neg]
        global queue
        queue = queue.detach()
        fea_center = torch.mean(queue, dim=0)
        pos_center = torch.mean(fea_pos, dim=0)
        # neg_center = torch.mean(fea_neg, dim=0)
        # cur_dist = torch.pow(pos_center - neg_center, 2).sum()/feas.shape[1]
        count = int(config.DATA.BATCH_SIZE*config.DATA.FRAME_LEN)
        # count = torch.FloatTensor(count)
        loss_center = torch.pow(fea_pos - fea_center, 2).sum(dim=1).sum()/count/feas.shape[1]*2 - \
                        torch.pow(fea_neg - fea_center, 2).sum(dim=1).sum()/count/2/feas.shape[1] - \
                        torch.pow(fea_neg - pos_center, 2).sum(dim=1).sum()/count/2/feas.shape[1]

        color_p = outputs[mask_pos, 1].cpu().data.numpy()
        color_n = outputs[mask_neg, 0].cpu().data.numpy()
        color_dis_p = 1 - color_p  # protocol 4 : 1;
        color_dis_p[color_dis_p < 0] = 0
        color_dis_n = 1 - color_n  # protocol 4 : 1;
        color_dis_n[color_dis_n < 0] = 0
        color_dis_n = color_dis_n.mean()
        color_dis_p = color_dis_p.mean()

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            # loss = criterion(outputs, targets) + (color_dis_p + color_dis_n)*0.1 + loss_center*0.1
            loss = criterion(outputs, targets)
            # loss = loss_center
            # loss = loss_center - cur_dist
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            # loss = criterion(outputs, targets) + (color_dis_p + color_dis_n)*0.1 + loss_center*0.1
            loss = criterion(outputs, targets)
            # loss = loss_center
            # loss = loss_center - cur_dist
 
            optimizer.zero_grad()
            
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if int(config.TRAIN.EPOCHS) % 3 == 0:
            queue[:config.DATA.BATCH_SIZE*config.DATA.FRAME_LEN,:] = fea_pos
        elif int(config.TRAIN.EPOCHS) % 3 == 1:
            queue[config.DATA.BATCH_SIZE*config.DATA.FRAME_LEN:config.DATA.BATCH_SIZE*config.DATA.FRAME_LEN*2,:] = fea_pos
        else:
            queue[config.DATA.BATCH_SIZE*config.DATA.FRAME_LEN*2:,:] = fea_pos

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, name):

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = softmax_cross_entropy_criterion

    model.eval()
    with torch.no_grad():
        score_list = []
        pred_list = []
        label_list = []
        for idx, (images, target) in enumerate(data_loader):
            # print("image shape : ",images.shape)
            # images = images.view(96,9,3,48,48)
            images = images.view(24, 9, 3, 48, 48)
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # src
            # res, dep_res = net(input, input_lr)
            res, feas = model(images)
            # logit_sum = res + dep_res * 0.5

            # logit_mean = res.mean(dim=0)
            target = target.view(res.shape[0])[0].data.cpu().numpy()
            # print("target = ",target)
            prob = torch.softmax(res, dim=1)[:, 1]
            pred = prob.mean(dim=0)#sum() / 48
            pred = pred.data.cpu().numpy()
            pred_list.append(pred)
            label_list.append(target)
            score_list.append("{} {}\n".format(pred,target))

        map_score_filename = "%s_score.txt"%name
        with open(map_score_filename, 'w') as file:
            file.writelines(score_list)

        # auc_score = roc_auc_score(label_list, pred_list)
        # print("auc score : ",auc_score)

        test_ACC, fpr, FRR, HTER, auc_test, test_err = performances_val(map_score_filename)

    logger.info(f' * HTER {HTER:.4f}  AUC {auc_test:.4f}  ACC {test_ACC:.4f}')
    return HTER, auc_test, test_ACC


if __name__ == '__main__':
    _, config = parse_option()
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    main(config)