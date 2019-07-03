from torch.utils.data import dataset
from PIL import Image
import random

NAME_FILE_F = 'Pair_list_F.txt.new'
NAME_FILE_P = 'Pair_list_P.txt.new'

TRAIN_FOLDERS = ["01", "02", "03", "04", "05", "06"]
VAL_FOLDERS = ["07", "08"]
TEST_FOLDERS = ["09", "10"]
ALL_FOLDERS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

SIZE_IDS = 500


class RetrievalCFPDataset(dataset.Dataset):
    def __init__(self, dataset_root='./cfg_dataset', img_transforms=None, perc_data=1., folders=TEST_FOLDERS):
        self.path_root = dataset_root
        self.data = []
        self.load_image_data()
        self.data = self.data[:int(len(self.data) * perc_data)]
        print("Dataset loaded")
        print("{0} samples".format(len(self.data)))
        self.transforms = img_transforms
        self.load_image_data()

    def load_image_data(self):
        self.frontal_images_directories = self.path_root + "/Protocol/" + NAME_FILE_F
        self.profile_images_directories = self.path_root + "/Protocol/" + NAME_FILE_P
        lines = open(self.profile_images_directories).readlines()
        self.data = []
        for line in lines:
            name_file = line.strip().split(' ')[1]
            if name_file not in self.data:
                self.data.append(name_file)

        lines = open(self.frontal_images_directories).readlines()
        for line in lines:
            name_file = line.strip().split(' ')[1]
            if name_file not in self.data:
                self.data.append(name_file)

    def __getitem__(self, index):
        image_path = self.data[index]
        image = Image.open(image_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        return image, image_path

    def __len__(self):
        return len(self.data)


class CFPDataset(dataset.Dataset):
    def __init__(self, dataset_root="./cfg_dataset", img_transforms=None,
                 split=None, perc_data=1., same_dif=[1.0, 0.0]):
        # super().__init__()
        self.path_root = dataset_root
        self.data = []
        self.same = same_dif[0]
        self.dif = same_dif[1]
        self.load_image_data()

        if not split:
            split = 'total'
        split = split.lower()
        if split == 'train':
            self.load_subset(TRAIN_FOLDERS)
        elif split == 'val':
            self.load_subset(VAL_FOLDERS)
        elif split == 'test':
            self.load_subset(TEST_FOLDERS)
        else:
            self.load_subset(ALL_FOLDERS)

        self.data = self.data[:int(len(self.data) * perc_data)]
        print("Dataset loaded")
        print("{0} samples in the {1} dataset".format(len(self.data), split))
        self.transforms = img_transforms

    def load_image_data(self):
        self.frontal_images_directories = self.path_root + "/Protocol/" + NAME_FILE_F
        self.profile_images_directories = self.path_root + "/Protocol/" + NAME_FILE_P
        lines = open(self.profile_images_directories).readlines()
        self.directory_profile_images = {}
        for line in lines:
            splitted_line = line.strip().split(' ')
            self.directory_profile_images[splitted_line[0]] = splitted_line

        lines = open(self.frontal_images_directories).readlines()
        self.directory_frontal_images = {}
        for line in lines:
            splitted_line = line.strip().split(' ')
            self.directory_frontal_images[splitted_line[0]] = splitted_line

    def load_annotations(self, annotations, dir1_images, dir2_images, is_same):
        data = []
        for fil in annotations:
            lines = open(fil).readlines()
            for line in lines:
                img_pair = line.strip().split(',')
                img1_dir = dir1_images[img_pair[0]][1]
                img2_dir = dir2_images[img_pair[1]][1]
                d = {
                    "img1_path": img1_dir,
                    "img2_path": img2_dir,
                    "pair_tag": is_same
                }
                data.append(d)
        return data

    def load_subset(self, folders):
        self.diff_ff_annotations = []
        self.same_ff_annotations = []
        self.diff_fp_annotations = []
        self.same_fp_annotations = []

        for train_folder in folders:
            self.diff_ff_annotations.append(self.path_root + "/Protocol/Split/FF/" + train_folder + "/diff.txt")
            self.same_ff_annotations.append(self.path_root + "/Protocol/Split/FF/" + train_folder + "/same.txt")
            self.diff_fp_annotations.append(self.path_root + "/Protocol/Split/FP/" + train_folder + "/diff.txt")
            self.same_fp_annotations.append(self.path_root + "/Protocol/Split/FP/" + train_folder + "/same.txt")

        self.data = []
        self.data.extend(self.load_annotations(self.diff_ff_annotations, self.directory_frontal_images, self.directory_frontal_images, self.dif))
        self.data.extend(self.load_annotations(self.same_ff_annotations, self.directory_frontal_images, self.directory_frontal_images, self.same))
        self.data.extend(self.load_annotations(self.diff_fp_annotations, self.directory_frontal_images, self.directory_profile_images, self.dif))
        self.data.extend(self.load_annotations(self.same_fp_annotations, self.directory_frontal_images, self.directory_profile_images, self.same))
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        image1_path = d['img1_path']
        image2_path = d['img2_path']
        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')
        tag = d['pair_tag']
        # print(tag)
        if self.transforms is not None:
            # this converts from (HxWxC) to (CxHxW) as wel
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)

        return image1, image2, tag

    def __len__(self):
        return len(self.data)


class TripletCFPDataset(dataset.Dataset):
    def __init__(self, dataset_root="./cfg_dataset", img_transforms=None,
                 split=None, perc_data=1.):
        # super().__init__()
        self.dictionaries_ids = {}
        self.path_root = dataset_root
        self.data = []
        self.load_image_data()

        if not split:
            split = 'total'
        split = split.lower()
        if split == 'train':
            self.load_subset(TRAIN_FOLDERS)
        elif split == 'val':
            self.load_subset(VAL_FOLDERS)
        elif split == 'test':
            self.load_subset(TEST_FOLDERS)
        else:
            self.load_subset(ALL_FOLDERS)

        self.data = self.data[:int(len(self.data) * perc_data)]
        print("Dataset loaded")
        print("{0} samples in the {1} dataset".format(len(self.data), split))
        self.transforms = img_transforms

    def load_image_data(self):
        self.frontal_images_directories = self.path_root + "/Protocol/" + NAME_FILE_F
        self.profile_images_directories = self.path_root + "/Protocol/" + NAME_FILE_P
        lines = open(self.profile_images_directories).readlines()
        self.directory_profile_images = {}
        for line in lines:
            splitted_line = line.strip().split(' ')
            self.directory_profile_images[splitted_line[0]] = splitted_line
            str_id = line.strip().split('/')[-3]
            if str_id not in self.dictionaries_ids:
                self.dictionaries_ids[str_id] = []
            self.dictionaries_ids[str_id].append(splitted_line[1])

        lines = open(self.frontal_images_directories).readlines()
        self.directory_frontal_images = {}
        for line in lines:
            splitted_line = line.strip().split(' ')
            self.directory_frontal_images[splitted_line[0]] = splitted_line
            str_id = line.strip().split('/')[-3]
            if str_id not in self.dictionaries_ids:
                self.dictionaries_ids[str_id] = []
            self.dictionaries_ids[str_id].append(splitted_line[1])

    def load_annotations(self, annotations, dir1_images, dir2_images):
        data = [] 
        for fil in annotations:
            lines = open(fil).readlines()
            for line in lines:
                img_pair = line.strip().split(',')
                img1_dir = dir1_images[img_pair[0]][1]
                img2_dir = dir2_images[img_pair[1]][1]
                id_img1 = img1_dir.split('/')[-3]
                id_img2 = img2_dir.split('/')[-3]
                
                k = id_img1
                if id_img1 == id_img2:
                    # search one random dif:
                    tries = 0
                    while k == id_img1 and tries < 10:
                        k = str(random.randint(1, SIZE_IDS)).zfill(3)
                        tries += 1
                    assert(tries < 10) 
                img3_dir = random.choice(self.dictionaries_ids[k])

                if id_img1 == id_img2:
                    # search one similar but not the same
                    d = {
                        "anchor": img1_dir,
                        "positive": img2_dir,
                        "negative": img3_dir
                    }
                else:
                    d = {
                        "anchor": img1_dir,
                        "positive": img3_dir,
                        "negative": img2_dir
                    }

                data.append(d)
        return data

    def load_subset(self, folders):
        self.diff_ff_annotations = []
        self.same_ff_annotations = []
        self.diff_fp_annotations = []
        self.same_fp_annotations = []

        for train_folder in folders:
            self.diff_ff_annotations.append(self.path_root + "/Protocol/Split/FF/" + train_folder + "/diff.txt")
            self.same_ff_annotations.append(self.path_root + "/Protocol/Split/FF/" + train_folder + "/same.txt")
            self.diff_fp_annotations.append(self.path_root + "/Protocol/Split/FP/" + train_folder + "/diff.txt")
            self.same_fp_annotations.append(self.path_root + "/Protocol/Split/FP/" + train_folder + "/same.txt")

        self.data = []
        self.data.extend(self.load_annotations(self.diff_ff_annotations, self.directory_frontal_images, self.directory_frontal_images))
        self.data.extend(self.load_annotations(self.same_ff_annotations, self.directory_frontal_images, self.directory_frontal_images))
        self.data.extend(self.load_annotations(self.diff_fp_annotations, self.directory_frontal_images, self.directory_profile_images))
        self.data.extend(self.load_annotations(self.same_fp_annotations, self.directory_frontal_images, self.directory_profile_images))
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        anchor = d['anchor']
        positive = d['positive']
        negative = d['negative']
        anchor = Image.open(anchor).convert('RGB')
        positive = Image.open(positive).convert('RGB')
        negative = Image.open(negative).convert('RGB')
        # print(tag)
        if self.transforms is not None:
            # this converts from (HxWxC) to (CxHxW) as wel
            anchor = self.transforms(anchor)
            positive = self.transforms(positive)
            negative = self.transforms(negative)

        return positive, anchor, negative

    def __len__(self):
        return len(self.data)
