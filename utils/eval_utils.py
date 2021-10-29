from os import system
import torch
import torch.nn as nn
from utils.conv_type import BinarizeWeight

def accuracy(output, target, args, codebook=None, topk=(1, 5)):
    """
    Computes the accuracy of the output relative to a decoding scheme. Currently supports exact
    exact decoding and minimum hamming distance decoding schemes.
    """

    scheme = 'mhd'
    if args.decode:
        scheme = args.decode
    
    with torch.no_grad():
        
        if not args.instance_code:
            # Output is num_classes-dim vector with scores representing the confidence
            # that the model has in that class
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append((correct_k.mul_(100.0 / batch_size)).item())
            return res

        # Returns the accuracy based on the Hamming distance between the output and target
        elif scheme == 'mhd':
            # Output starts as k-bit multilabel vectors. Compare with codebook to find closest
            # class codes. Resulting output is 'closest' num_classes-dim vector from the codebook
            output = BinarizeWeight.apply(output)@torch.transpose(codebook.weight, 0, 1)
            maxk = max(topk)
            batch_size = target.size(0)
            # Returns indices of classes with lowest hamming distance (ie. max dot product)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append((correct_k.mul_(100.0 / batch_size)).item())
            return res

        # Returns the accuracy measured in terms of how many output codes
        # exactly match the target codes
        elif scheme == 'ed':            
            # Make the target label a code rather than a class
            target = (codebook(target).cuda(args.gpu, non_blocking=True) + 1) / 2
            m = nn.Sigmoid()
            output = m(output)

            batch_size = target.size(0)
            pred = output.clone().detach()
            pred[pred>=0.5] = 1
            pred[pred<0.5] = 0
            diff = torch.abs(pred-target).sum(1)
            correct = diff.numel() - diff.nonzero().size(0)
            res = []
            for k in topk:
                res.append(correct*100.0/batch_size)
            return res

