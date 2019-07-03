import torch
import numpy as np
from utils import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import random


# initializations
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run(params):
    # code to set up report
    writer = SummaryWriter(comment=params.name_test)
    import itertools
    globalitzer = itertools.count()

    best_loss = 100.
    for epoch in range(params.epochs):
        # update lr
        if epoch > 0 and epoch % 10 == 0:
            for g in params.optimizer.param_groups:
                g['lr'] *= 0.1
                print("update lr")

        av_loss, acc = params.model.run_train(params)
        val_av_loss, val_acc = params.model.run_val(params)

        val_globalitzer = next(globalitzer)
        print(f'TRAIN Epoch:{epoch} - Loss:{av_loss} valid Loss:{val_av_loss}, Accur:{acc}, valid accur{val_acc}')
        writer.add_scalar('train/av_loss', av_loss, global_step=val_globalitzer)
        writer.add_scalar('train/val_av_loss', val_av_loss, global_step=val_globalitzer)
        if val_av_loss < best_loss:
            print("Saving checkpoint in epoch " + str(epoch) + " and loss " + str(av_loss))
            best_loss = val_av_loss
            save_checkpoint(epoch, params.model, av_loss, acc, val_acc, params.name_test + ".pth.tar")
    test_accur = params.model.run_test(params)
    print("CORRECT ACCURACY: " + str(test_accur))
