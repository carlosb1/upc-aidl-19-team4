import torch
import numpy as np
from tqdm import tqdm


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:

        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def get_annotations(ds):
    list_idx = []
    for i in tqdm(range(len(ds))):
        list_idx.append(ds[i][1])

    list_idx = np.array(list_idx)

    return list_idx


def evaluate_mAP(feats, ds, N=-1, random_seed=1234):
    # all similarities
    all_sim = torch.mm(feats, feats.t())

    # all ranked lists
    all_ranked = torch.sort(all_sim, descending=True).indices
    all_ranked = all_ranked.cpu().numpy()

    # get annotations
    annotations = get_annotations(ds)

    all_ap = []

    # set random seed
    idx = np.arange(len(ds))
    np.random.seed(random_seed)
    np.random.shuffle(idx)

    if N > -1:
        idx = idx[:N]

    for k, i in enumerate(idx):
        # get ID query
        id_query = ds[i][1]

        # build list of relevant ids
        gt = np.zeros_like(annotations)
        gt[annotations[all_ranked[i, ...]] == id_query] = 1  # set1 positions of rel.class

        # compute ap
        ap = average_precision(gt)

        all_ap.append(ap)
        if k % 100 == 0:
            print(f'{k}, Query {i}\t{ds.classes[id_query]}\t AP={ap}')

    return all_ap


def extract_features_retrieval(model, dataloader, device):

    FEATS = None  # We collect all ds representations
    LABELS = None  # And labels (for evaluation)
    with tqdm(total=len(dataloader)) as pbar:
        for X in dataloader:
            ima = X[0].to(device)
            label = X[1]
            feat = model.extract_feature(ima)
            if FEATS is None:
                FEATS = feat
                LABELS = label
            else:
                FEATS = torch.cat((FEATS, feat), dim=0)
                LABELS += label
            pbar.update(1)

    return FEATS, LABELS


def accuracy(output, target, topk=(1,)):
    """ accuracy using pytorch functionalities
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # transpose
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count


def save_checkpoint(epoch, model, loss, train_acc, val_acc, filename='checkpoint_vggsiamese.pth.tar'):
    state = {'epoch': epoch,
             'loss': loss,
             'train_acc': train_acc,
             'val_acc': val_acc,
             'model': model}
    torch.save(state, 'BEST_' + filename)
