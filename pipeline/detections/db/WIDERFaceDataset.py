# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

import constants as datasets_constants


class WIDERFaceDataset(object):

    __name = 'WIDERFaceDataset'
    __minimum_face_size = datasets_constants.minimum_dataset_face_size

    @classmethod
    def name(cls):
        return (WIDERFaceDataset.__name)

    @classmethod
    def minimum_face_size(cls):
        return (WIDERFaceDataset.__minimum_face_size)

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

    def read(self, annotation_image_dir, annotation_file_name):

        self._clear()

        if (not os.path.isfile(annotation_file_name)):
            return (False)

        images = []
        bounding_boxes = []
        annotation_file = open(annotation_file_name, 'r')
        while (True):
            image_path = annotation_file.readline().strip('\n')
            if (not image_path):
                break

            image_path = os.path.join(annotation_image_dir, image_path)
            image = cv2.imread(image_path)
            if (image is None):
                continue

            nums = annotation_file.readline().strip('\n')
            one_image_boxes = []
            for face_index in range(int(nums)):
                bounding_box_info = annotation_file.readline().strip(
                    '\n').split(' ')

                xmin = float(bounding_box_info[0])
                ymin = float(bounding_box_info[1])
                width = float(bounding_box_info[2])
                height = float(bounding_box_info[3])

                xmax = xmin + width
                ymax = ymin + height

                if ((max(width, height) >=
                     WIDERFaceDataset.minimum_face_size()) and (width > 0)
                        and (height > 0)):
                    one_image_boxes.append([xmin, ymin, xmax, ymax])

            if (len(one_image_boxes)):
                images.append(image_path)
                bounding_boxes.append(one_image_boxes)
                self._number_of_faces += len(one_image_boxes)

        if (len(images)):
            self._data['images'] = images
            self._data['bboxes'] = bounding_boxes
            self._data['number_of_faces'] = self._number_of_faces

            self._is_valid = True
            print(self._number_of_faces, 'faces in ', len(images),
                  'number of images for WIDER Face dataset.')
        else:
            self._clear()

        return (self.is_valid())
