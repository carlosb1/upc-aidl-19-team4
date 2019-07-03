import torch
import numpy as np
import torch.nn as nn
from cfp_dataset import CFPDataset
from transforms import simple_transforms
from models import VGGSiameseNet, SiameseCosine
from params import PERC_DATA, BATCH_SIZE, WORKERS
import sys
from bunch import Bunch
from models import test_cosine

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def calculate_threshold_validation(val_dataloader):
    total_same_min = None
    total_diff_max = None
    # Calculate threshold
    for i, X in enumerate(val_dataloader):
        x1, x2, label = X
        x1 = x1.to(device)
        x2 = x2.to(device)

        label = label.float().to(device)
        out1, out2 = model.forward((x1, x2))
        results = cos(out1, out2)
        np_results = results.cpu().data.numpy()
        np_labels = label.cpu().data.numpy()

        index_same = np.where(np_labels == 1.0)
        index_diff = np.where(np_labels == 0.0)
        same_min = np_results[index_same].min()
        diff_max = np_results[index_diff].max()
        if not total_diff_max or total_diff_max < diff_max:
            total_diff_max = diff_max
        if not total_same_min or total_same_min > same_min:
            total_same_min = same_min

    mean = np.mean([total_same_min, total_diff_max])
    return total_same_min, total_diff_max, mean

if __name__ == '__main__':
    PATH_DATASET = '/home/carlosb/data/cfp-dataset'
    info = torch.load(sys.argv[1])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = info['model']
    # Init dataset
    validset = CFPDataset(PATH_DATASET, img_transforms=simple_transforms, split='val', perc_data=PERC_DATA)
    testset = CFPDataset(PATH_DATASET, img_transforms=simple_transforms, split='test', perc_data=PERC_DATA)

    dataloader = {
        'val':
        torch.utils.data.DataLoader(dataset=validset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=WORKERS),
        'test':
        torch.utils.data.DataLoader(dataset=testset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=WORKERS)
    }

    params = Bunch()
    params.model = model
    params.dataloader = dataloader
    
    if len(sys.argv) >= 3:
        thresh = float(sys.argv[2]) 
    else: 
        total_same_min, total_diff_max, mean = calculate_threshold_validation(dataloader['val'])
        print("same min: " + str(total_same_min))
        print("diff max: " + str(total_diff_max))
        print("mean: " + str(mean))
    
        best_acc = 0.
        best_tresh = 0.
        for pos_threshold in np.arange(mean, total_diff_max, 0.01):
            print("try with treshold: " + str(pos_threshold))
            pos_acc = test_cosine(params, threshold=pos_threshold, typedataloader='test')
            if best_acc < pos_acc:
                best_acc = pos_acc
                best_tresh = pos_threshold
                print("new best acc: " + str(best_acc))
                print("new best trehs: " + str(best_tresh))
    
        thresh = best_tresh

    test_best_acc = test_cosine(params, threshold=thresh, typedataloader='test')
    valid_best_acc = test_cosine(params, threshold=thresh, typedataloader='val')
    print("###" + str(sys.argv[1]))
    print("######test best acc: " + str(test_best_acc))
    print("######valid best acc: " + str(valid_best_acc))
    print("######best trehs: " + str(thresh))
