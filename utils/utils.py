import torch
import os
import shutil

N_CLASSES = 3474


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    # assert(len(lr)==1) #we support only one param_group
    # lr = lr[0]
    return lr


def save_checkpoint(state, is_best, output_dir):
    filename = os.path.join(output_dir, 'checkpoint', 'checkpoint_fold_%1d_epoch_%03d_score_%0.4f.pth'%(state['fold'], state['epoch'], state['score']))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(output_dir, 'model_best.pth'))


def metric_average(val, name):
    import horovod.torch as hvd
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor.detach().cpu(), name=name)
    return avg_tensor.item()
