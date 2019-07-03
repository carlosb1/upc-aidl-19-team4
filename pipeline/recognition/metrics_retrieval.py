import torch
import torch.nn as nn
from cfp_dataset import RetrievalCFPDataset
from torchvision import transforms
from models import VGGSiameseNet 
from config import PERC_DATA, WORKERS, BATCH_SIZE
from utils import extract_features_retrieval


PATH_IMAGE = './examples/1.jpg'
SMALL_SIZE = (24, 24)
trans = transforms.Compose([transforms.Resize(SMALL_SIZE), transforms.ToTensor()])
trans2 = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

import sys
if len(sys.argv) > 1 and sys.argv[1] == "aws":
    PATH_DATASET = '/root/data/cfp-dataset'
    info = torch.load('/root/project/backups/BEST_checkpoint_vggsiamese_1.pth.tar')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    PATH_DATASET = '/home/carlosb/python-workspace/upc-aidl-19-team4/datasets/cfp-dataset'
    info = torch.load('/home/carlosb/Desktop/BEST_checkpoint_vggsiamese_1.pth.tar')
    device = 'cpu'

model = info['model']
testset = RetrievalCFPDataset(PATH_DATASET, img_transforms=trans, perc_data=PERC_DATA)

dataloader = {
    'test':
    torch.utils.data.DataLoader(dataset=testset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=WORKERS)
}
feats, labels = extract_features_retrieval(model, dataloader['test'], device)
from PIL import Image
image = Image.open(PATH_IMAGE).convert('RGB')
image = trans2(image)
query = model.extract_feature(image[None, ...].to(device))

dist = cos(query, feats)
index_sorted = torch.argsort(dist)
top_10 = index_sorted[:10]

top = 0
for i in top_10:
    print(str(top) + " - " + str(i.item()) + " - " + str(labels[i]))
    top += 1
