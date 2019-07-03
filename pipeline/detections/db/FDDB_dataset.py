# MIT License
#
# Copyright (c) 2018
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import random
import constants


class FDDBDataset(object):

    __name = 'FDDBDataset'
    __minimum_face_size = constants.minimum_face_size

    @classmethod
    def name(cls):
        return (FDDBDataset.__name)

    @classmethod
    def minimum_face_size(cls):
        return (FDDBDataset.__minimum_face_size)

    def __init__(self):
        self._clear()

    def _clear(self):
        self._is_valid = False
        self._data = dict()
        self._number_of_faces = 0

    def is_valid(self):
        return (self._is_valid)

    def data(self):
        return (self._data)

    def number_of_faces(self):
        return (self._number_of_faces)

    def _filterCoordinate(self, c, m):
        if (c < 0):
            return (0)

        elif (c > m):
            return (m)
        else:
            return (c)

    def read(self, annotation_file_dir):

        self._is_valid = True
        self._clear()

        images = []
        bounding_boxes = []
        annotation_files = glob.glob(annotation_file_dir + "/*rect.txt")
        for annotation_file_name in annotation_files:
            print("Reading annotation_file_name: " + annotation_file_name)
            try:
                annotation_file = open(annotation_file_name, 'r')
                lines = annotation_file.readlines()
                to_save_file_path = None
                image_bounding_boxes = []
                for line in lines:
                    splitted_line = line.split(",")
                    image_file_path = splitted_line[0]
                    x_min = float(splitted_line[1])
                    y_min = float(splitted_line[2])
                    x_max = float(splitted_line[3])
                    y_max = float(splitted_line[4])
                    if image_file_path != to_save_file_path:
                        # We save backup points
                        if to_save_file_path:
                            images.append(to_save_file_path)
                            bounding_boxes.append(image_bounding_boxes)
                        # Reset points
                        to_save_file_path = image_file_path
                        image_bounding_boxes = []
                    # Add bounding boxes
                    image_bounding_boxes.append([x_min, y_min, x_max, y_max])
            except IOError as e:
                print("Error: " + str(e))
                self._is_valid = False
        self._data['images'] = images
        self._data['bboxes'] = bounding_boxes
        self._data['number_of_faces'] = self._number_of_faces

        return (self.is_valid())

    def get_data(self, train=40, val=10, test=50):
        if not ("images" in self._data) or not("bboxes" in self._data):
            print("read first dataset to split it")
            return
        images = []
        bboxes = []

        if (train + test + val > 100):
            return "invalid inpuit"

        rand = list(range(len(self._data['images'])))
        random.shuffle(rand)

        for index in rand:
            i = rand[index]
            images.append(self._data['images'][i])
            bboxes.append(self._data['bboxes'][i])

        train = int(train / 100 * len(images))
        val = int(val / 100 * len(images))
        test = int(test / 100 * len(images))

        train_images = images[0:train]
        train_bboxes = bboxes[0:train]

        val_images = images[train:train + val]
        val_bboxes = bboxes[train:train + val]

        test_images = images[train + val:train + val + test]
        test_bboxes = bboxes[train + val:train + val + test]

        train_data = [train_images, train_bboxes]
        val_data = [val_images, val_bboxes]
        test_data = [test_images, test_bboxes]

        data = [train_data, val_data, test_data]

        return data
