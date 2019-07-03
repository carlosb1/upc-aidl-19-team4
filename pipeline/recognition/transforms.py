from torchvision import transforms
from imgaug import augmenters as iaa
import numpy as np

SIZE = (224, 224)
SMALL_SIZE = (24, 24)

simple_transforms = transforms.Compose([transforms.Resize(SIZE), transforms.ToTensor()])
small_transforms = transforms.Compose([transforms.Resize(SMALL_SIZE), transforms.ToTensor()])


normal_transforms = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
                iaa.Scale(SIZE),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-20, 20), mode='symmetric'),
                iaa.Sometimes(0.25, iaa.OneOf([iaa.Dropout(p=(0, 0.1)), iaa.CoarseDropout(0.1, size_percent=0.5)])),
                iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
            ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


iaa_transforms = ImgAugTransform()
