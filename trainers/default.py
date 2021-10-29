import time
import torch
import torch.nn as nn
import tqdm

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.conv_type import BinarizeWeight



__all__ = ["train", "validate", "modifier"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer, codebook=None):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        # if codebook learning,     output is binary labels (1000 dim)
        # else                      output is scores (k dims) representing confidence in kth label
        output = model(images)

        # compute loss
        if not args.instance_code:
            loss = criterion(output, target)

        else:
            sig = nn.Sigmoid()
            target_code = (codebook(target).cuda(args.gpu, non_blocking=True)+1)/2
            loss = criterion(sig(output), target_code.detach())
        
        losses.update(loss.item(), images.size(0))
        print(model.state_dict()['module.fc.weight'][0])
        # measure accuracy
        acc1, acc5 = accuracy(output, target, args, codebook=codebook, topk=(1, 5))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch, codebook=None):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)

            # measure accuracy
            acc1, acc5 = accuracy(output, target, args, codebook=codebook, topk=(1, 5))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))

            # compute loss
            if not args.instance_code:
                loss = criterion(output, target)

            else:
                sig = nn.Sigmoid()
                target_code = (codebook(target).cuda(args.gpu, non_blocking=True)+1)/2
                loss = criterion(sig(output), target_code.detach())
            
            losses.update(loss.item(), images.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)
    
    return top1.avg, top5.avg

def modifier(args, epoch, model):
    return
