import torch
import torch.nn as nn
from torch.nn.functional import triplet_margin_loss
from torchvision.models import vgg16_bn
from utils import AverageMeter, accuracy
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_triplet(params):
    assert(type(params.loss) == type(triplet_margin_loss))
    all_loss = AverageMeter()
    params.optimizer.zero_grad()
    # batch
    for i, X in enumerate(params.dataloader['train']):
        params.model = params.model.train()
        positive, anchor, negative = X
        positive = positive.to(device)
        anchor = anchor.to(device)
        negative = negative.to(device)

        out_positive = params.model.extract_feature(positive)
        out_anchor = params.model.extract_feature(anchor)
        out_negative = params.model.extract_feature(negative)

        loss = params.loss(out_positive, out_anchor, out_negative)
        params.optimizer.zero_grad()
        loss.backward()
        params.optimizer.step()
        all_loss.update(loss.item(), anchor.size(0))
    return all_loss.avg, -1.0


def val_triplet(params):
    assert(type(params.loss) == type(triplet_margin_loss))
    all_loss = AverageMeter()
    # batch
    for i, X in enumerate(params.dataloader['val']):
        params.model = params.model.eval()
        positive, anchor, negative = X
        positive = positive.to(device)
        anchor = anchor.to(device)
        negative = negative.to(device)
        out_positive = params.model.extract_feature(positive)
        out_anchor = params.model.extract_feature(anchor)
        out_negative = params.model.extract_feature(negative)
        loss = params.loss(out_positive, out_anchor, out_negative)
        all_loss.update(loss.item(), anchor.size(0))
    return all_loss.avg, -1.0


def train(params):
    all_loss = AverageMeter()
    params.optimizer.zero_grad()
    # batch
    for i, X in enumerate(params.dataloader['train']):
        params.model = params.model.train()
        x1, x2, label = X
        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.float().to(device)
        out1, out2 = params.model.forward((x1, x2))
        loss = params.loss(out1, out2, label)
        params.optimizer.zero_grad()
        loss.backward()
        params.optimizer.step()
        all_loss.update(loss.item(), x1.size(0))
    return all_loss.avg, -1.0


def val(params):
    all_loss = AverageMeter()
    # batch
    for i, X in enumerate(params.dataloader['val']):
        params.model = params.model.eval()
        x1, x2, label = X
        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.float().to(device)
        out1, out2 = params.model.forward((x1, x2))
        loss = params.loss(out1, out2, label)
        all_loss.update(loss.item(), x1.size(0))
    return all_loss.avg, -1.0


def test_cosine(params, threshold=0.99, typedataloader='test'):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    total_accuracy_values = {0.0: 0, 1.0: 0, -1.0: 0}
# Calculate accuracy
    for i, X in enumerate(params.dataloader[typedataloader]):
        x1, x2, label = X
        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.float().to(device)
        out1, out2 = params.model.forward((x1, x2))
        results = cos(out1, out2)
        np_results = results.cpu().data.numpy()
        np_labels = label.cpu().data.numpy()
        threshold_results = (np_results >= threshold).astype(float)
        cmp_results = threshold_results - np_labels
        unique, counts = np.unique(cmp_results, return_counts=True)
        result_accuracy = dict(zip(unique, counts))
        if 0.0 in result_accuracy:
            total_accuracy_values[0.0] += result_accuracy[0.0]
        if 1.0 in result_accuracy:
            total_accuracy_values[1.0] += result_accuracy[1.0]
        if -1.0 in result_accuracy:
            total_accuracy_values[-1.0] += result_accuracy[-1.0]

    print(str(total_accuracy_values))
    total = float(total_accuracy_values[0.0] + total_accuracy_values[1.0] + total_accuracy_values[-1.0])
    total_accur = float(total_accuracy_values[0.0]) / total
    return total_accur


def train_dec(params):
    params.model.train()
    acc = AverageMeter()
    losses = AverageMeter()
    params.optimizer.zero_grad()
    for i, X in enumerate(params.dataloader['train']):
        x1, x2, label = X
        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.long().to(device)
        outputs = params.model(x1, x2)
        loss = params.loss(outputs, label)
        loss.backward()

        losses.update(loss.item(), x1.size(0))
        acc.update(accuracy(outputs.data, label)[0], x1.size(0))

        params.optimizer.step()
        params.optimizer.zero_grad()
    return losses.avg, acc.avg


def val_dec(params):
    params.model.eval()
    losses = AverageMeter()
    acc = AverageMeter()
    # batch
    for i, X in enumerate(params.dataloader['val']):
        x1, x2, label = X
        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.long().to(device)

        outputs = params.model(x1, x2)
        loss = params.loss(outputs, label)
        losses.update(loss.item(), x1.size(0))
        acc.update(accuracy(outputs.data, label)[0], x1.size(0))
    return losses.avg, acc.avg.item()


def test(params):
    params.model.eval()
    acc = AverageMeter()
    with torch.no_grad():
        for i, (img1, img2, gt) in enumerate(params.dataloader['test']):
            img1 = img1.to('cuda:0')
            img2 = img2.to('cuda:0')
            gt = gt.long().to('cuda:0')
            batch_size = img1.size(0)
            outputs = params.model(img1, img2)
            acc.update(accuracy(outputs.data, gt)[0], batch_size)
    return acc.avg


def l2norm(x):
    x = x / torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))
    return x


class VGGSiameseNet(nn.Module):
    def __init__(self):
        super(VGGSiameseNet, self).__init__()
        vgg16_model = vgg16_bn(pretrained=True)
        self.feat = vgg16_model.features
        self.linear_classifier = vgg16_model.classifier[0]

        self.avgpool1 = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool2 = nn.AdaptiveAvgPool2d(
            (7, 7))  # TODO It is not necessary two different avgpools
        self.run_train = train
        self.run_val = val
        self.run_test = test_cosine

    def forward(self, X):
        x1 = X[0]
        x2 = X[1]

        out1 = self.feat(x1)
        out1 = self.avgpool1(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.linear_classifier(out1)
        out1 = l2norm(out1)

        out2 = self.feat(x2)
        out2 = self.avgpool2(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.linear_classifier(out2)
        out2 = l2norm(out2)

        return out1, out2

    def extract_feature(self, x):
        x = self.feat(x)
        x = self.avgpool1(x)
        x = x.view(x.size(0), -1)
        x = self.linear_classifier(x)
        x = l2norm(x)
        return x


class TripletVGGSiameseNet(nn.Module):
    def __init__(self):
        super(TripletVGGSiameseNet, self).__init__()
        vgg16_model = vgg16_bn(pretrained=True)
        self.feat = vgg16_model.features
        self.linear_classifier = vgg16_model.classifier[0]

        self.avgpool1 = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool2 = nn.AdaptiveAvgPool2d(
            (7, 7))  # TODO It is not necessary two different avgpools
        self.run_train = train_triplet
        self.run_val = val_triplet
        self.run_test = test_cosine

    def forward(self, X):
        x1 = X[0]
        x2 = X[1]

        out1 = self.feat(x1)
        out1 = self.avgpool1(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.linear_classifier(out1)
        out1 = l2norm(out1)

        out2 = self.feat(x2)
        out2 = self.avgpool2(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.linear_classifier(out2)
        out2 = l2norm(out2)

        return out1, out2

    def extract_feature(self, x):
        x = self.feat(x)
        x = self.avgpool1(x)
        x = x.view(x.size(0), -1)
        x = self.linear_classifier(x)
        x = l2norm(x)
        return x

class SiameseCosine(nn.Module):
    def __init__(self, pretrained=False):
        super(SiameseCosine, self).__init__()

        self.feat_conv = vgg16_bn(pretrained=pretrained).features
        self.feat_linear = nn.Linear(in_features=512 * 7 * 7,
                                     out_features=4096)
        self.run_train = train
        self.run_val = val
        self.run_test = test_cosine

    def forward(self, X):
        img1 = X[0]
        img2 = X[1]
        feat_1 = self.feat_conv(img1).view(img1.size(0), -1)
        feat_1 = self.feat_linear(feat_1)

        feat_1 = l2norm(feat_1)

        feat_2 = self.feat_conv(img2).view(img2.size(0), -1)
        feat_2 = self.feat_linear(feat_2)
        feat_2 = l2norm(feat_2)
        return feat_1, feat_2

    def extract_feature(self, x):
        feat = self.feat_conv(x).view(x.size(0), -1)
        feat = self.feat_linear(feat)
        feat = l2norm(feat)
        return feat


class SiameseDecision(nn.Module):
    def __init__(self, pretrained=False):
        super(SiameseDecision, self).__init__()

        self.feat = vgg16_bn(pretrained=pretrained).features
        self.decision_network = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7 * 2, out_features=4096),
            nn.ReLU(True), nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(True),
            nn.Dropout(), nn.Linear(in_features=4096, out_features=2))

        self.run_train = train_dec
        self.run_val = val_dec
        self.run_test = test

    def forward(self, img1, img2):
        feat_1 = self.feat(img1).view(img1.size(0), -1)

        feat_2 = self.feat(img2).view(img2.size(0), -1)

        feat = torch.cat((feat_1, feat_2), 1)

        return self.decision_network(feat)


class SiameseLinearDecision(nn.Module):
    def __init__(self, pretrained=False):
        super(SiameseLinearDecision, self).__init__()

        self.feat_conv = vgg16_bn(pretrained=pretrained).features
        self.feat_linear = nn.Linear(in_features=512 * 7 * 7,
                                     out_features=4096)
        self.decision_network = nn.Sequential(
            nn.Linear(in_features=4096 * 2, out_features=4096), nn.ReLU(True),
            nn.Dropout(), nn.Linear(in_features=4096, out_features=2))

        self.run_train = train_dec
        self.run_val = val_dec
        self.run_test = test

    def forward(self, img1, img2):
        feat_1 = self.feat_conv(img1).view(img1.size(0), -1)

        feat_1 = self.feat_linear(feat_1)

        feat_2 = self.feat_conv(img2).view(img2.size(0), -1)
        feat_2 = self.feat_linear(feat_2)

        feat = torch.cat((feat_1, feat_2), 1)

        return self.decision_network(feat)
