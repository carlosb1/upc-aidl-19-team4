import torch
import torch.nn as nn
from torch.nn.functional import triplet_margin_loss
from cfp_dataset import CFPDataset, TripletCFPDataset
from transforms import simple_transforms, normal_transforms, small_transforms
from models import VGGSiameseNet, SiameseCosine, SiameseLinearDecision, SiameseDecision, TripletVGGSiameseNet
from bunch import Bunch

PERCENTAGES = [0.5, 0.3, 0.2]
BATCH_SIZE = 20
CH_SIZE = 128
WORKERS = 6
SIMPLE_TRANSFORM = 'simple'
SMALL_TRANSFORM = 'small'
NORMAL_TRANSFORM = 'normal'
LR = 0.01
NUM_EPOCHS = 1
WEIGHT_DECAY = 0.0001
PERC_DATA = 1.0
MODE_DATA_TOTAL = 'total'
MODE_DATA_SARA = 'sara'
NAME_TEST = 'name_test'
MODEL_SIAMESE1 = 'siamese1'
TRIPLET_MODEL_SIAMESE1 = 'siamese1_triplet'
MODEL_SIAMESE2 = 'siamese2'
MODEL_DECISION_LINEAR = 'decision_linear'
MODEL_DECISION = 'decision'
OPTIMIZER_SGD = 'SGD'
OPTIMIZER_ADAM = 'ADAM'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BuilderTrain():
    def __init__(self, path_dataset):
        self.path_dataset = path_dataset
        self.type_transform = SIMPLE_TRANSFORM
        self.mode_data = MODE_DATA_TOTAL
        self.perc_data = PERC_DATA
        self.type_model = MODEL_SIAMESE1
        self.name_test = NAME_TEST
        self.type_optimizer = OPTIMIZER_SGD
        self.lr = LR
        self.weight_decay = WEIGHT_DECAY
        self.epochs = NUM_EPOCHS
        self.batch_size = BATCH_SIZE
        self.workers = WORKERS
        self.filepath_checkpoint = None

    def transform(self, type_transform):
        self.type_transform = type_transform
        return self

    def dataset(self, mode_data, perc_data, batch_size):
        self.mode_data = mode_data
        self.perc_data = perc_data
        self.batch_size = batch_size
        return self

    def model(self, type_model, filepath_checkpoint=None):
        self.type_model = type_model
        self.filepath_checkpoint = filepath_checkpoint
        return self

    def name_run(self, name_test):
        self.name_test = name_test
        return self

    def optimizer(self, type_optimizer, lr, weight_decay):
        self.type_optimizer = type_optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        return self

    def criterion(self, type_loss):
        self.type_loss = type_loss
        return self

    def num_epochs(self, epochs):
        self.epochs = epochs
        return self

    def build(self):
        params = Bunch()

        if self.type_transform == SIMPLE_TRANSFORM:
            params.transforms = simple_transforms
        elif self.type_transform == SMALL_TRANSFORM:
            params.transforms = small_transforms
        else:
            params.transforms = normal_transforms

        if self.type_model == MODEL_SIAMESE1:
            params.model = VGGSiameseNet().to(device)
            params.loss = nn.CosineEmbeddingLoss(margin=0.5).to(device)
            params.retrieval = True
            same_diff = [1.0, -1.0]
        elif self.type_model == TRIPLET_MODEL_SIAMESE1:
            params.model = TripletVGGSiameseNet().to(device)
            params.loss = triplet_margin_loss
            params.retrieval = True
            params.triplet = True
        elif self.type_model == MODEL_SIAMESE2:
            params.model = SiameseCosine().to(device)
            params.loss = nn.CosineEmbeddingLoss(margin=0.5).to(device)
            params.retrieval = True
            same_diff = [1.0, -1.0]
        elif self.type_model == MODEL_DECISION_LINEAR:
            params.model = SiameseLinearDecision(pretrained=True).to(device)
            params.loss = nn.CrossEntropyLoss()
            params.retrieval = False
            same_diff = [1.0, 0.0]
        else:
            params.model = SiameseDecision(pretrained=True).to(device)
            params.loss = nn.CrossEntropyLoss()
            params.retrieval = False
            same_diff = [1.0, 0.0]
        if params.triplet:
                params.trainset = TripletCFPDataset(self.path_dataset, img_transforms=params.transforms, split='train', perc_data=self.perc_data)
                params.validset = TripletCFPDataset(self.path_dataset, img_transforms=simple_transforms, split='val', perc_data=self.perc_data)
                params.testset = CFPDataset(self.path_dataset, img_transforms=simple_transforms, split='test', perc_data=self.perc_data, same_dif=[1.0, -1.0])
        else:
            # Init dataset
            if self.mode_data == 'total':
                assert(self.type_transform == SIMPLE_TRANSFORM)
                ds_pairs = CFPDataset(self.path_dataset, img_transforms=params.transforms, split=self.mode_data, perc_data=self.perc_data, same_dif=same_diff)
                splitted_data_percs = [int(round(x * len(ds_pairs), 0)) for x in PERCENTAGES]
                params.trainset, params.validset, params.testset = torch.utils.data.random_split(ds_pairs, splitted_data_percs)
            else:
                params.trainset = CFPDataset(self.path_dataset, img_transforms=params.transforms, split='train', perc_data=self.perc_data, same_dif=same_diff)
                params.validset = CFPDataset(self.path_dataset, img_transforms=simple_transforms, split='val', perc_data=self.perc_data, same_dif=same_diff)
                params.testset = CFPDataset(self.path_dataset, img_transforms=simple_transforms, split='test', perc_data=self.perc_data, same_dif=same_diff)

        params.dataloader = {
            'train': torch.utils.data.DataLoader(dataset=params.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers),
            'val': torch.utils.data.DataLoader(dataset=params.validset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers),
            'test': torch.utils.data.DataLoader(dataset=params.testset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
        }
        if self.type_optimizer == OPTIMIZER_ADAM:
            params.optimizer = torch.optim.Adam(params.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            params.optimizer = torch.optim.SGD(params.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        params.name_test = self.name_test
        params.epochs = self.epochs

        return params
