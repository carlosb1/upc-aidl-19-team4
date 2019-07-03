from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from FDDB_dataset import FDDBDataset


def show():
    PATH_SOURCE_ANNOTATION = '/home/carlosb/python-workspace/upc-aidl-19-team4/datasets/FDDB-folds'
    dataset = FDDBDataset()
    dataset.read(PATH_SOURCE_ANNOTATION)
    images = dataset.data()['images']
    bboxes = dataset.data()['bboxes']
    for index_number_image in range(0, len(images)):
        boxes = bboxes[index_number_image]
        img = cv2.imread(images[index_number_image])
        for box in boxes:
            box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            print(box)
            print(images[index_number_image])
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0),
                          1)
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == 27 or k == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    show()
