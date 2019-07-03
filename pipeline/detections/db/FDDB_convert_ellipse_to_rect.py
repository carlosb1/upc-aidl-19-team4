import sys, os
import numpy as np
from math import *
from PIL import Image

import glob
from pathlib import Path, PurePath

folder_images = '/home/carlosb/python-workspace/upc-aidl-19-team4/datasets/originalPics/'
ellipse_folder_files = '/home/carlosb/python-workspace/upc-aidl-19-team4/datasets/FDDB-folds/'


def filterCoordinate(c, m):
    if c < 0:
        return 0
    elif c > m:
        return m
    else:
        return c


def build_values_from_ellipse_filename(_folder_images, _ellipse_filename,
                                       _rect_filename):
    with open(_ellipse_filename) as f:
        lines = [line.rstrip('\n') for line in f]

    f = open(_rect_filename, 'w')
    i = 0
    while i < len(lines):
        img_file = _folder_images + lines[i] + '.jpg'
        img = Image.open(img_file)
        w = img.size[0]
        h = img.size[1]
        num_faces = int(lines[i + 1])
        for j in range(num_faces):
            ellipse = lines[i + 2 + j].split()[0:5]
            a = float(ellipse[0])
            b = float(ellipse[1])
            angle = float(ellipse[2])
            centre_x = float(ellipse[3])
            centre_y = float(ellipse[4])
            tan_t = -(b / a) * tan(angle)
            t = atan(tan_t)
            x1 = centre_x + (a * cos(t) * cos(angle) - b * sin(t) * sin(angle))
            x2 = centre_x + (a * cos(t + pi) * cos(angle) -
                             b * sin(t + pi) * sin(angle))
            x_max = filterCoordinate(max(x1, x2), w)
            x_min = filterCoordinate(min(x1, x2), w)

            if tan(angle) != 0:
                tan_t = (b / a) * (1 / tan(angle))
            else:
                tan_t = (b / a) * (1 / (tan(angle) + 0.0001))
            t = atan(tan_t)
            y1 = centre_y + (b * sin(t) * cos(angle) + a * cos(t) * sin(angle))
            y2 = centre_y + (b * sin(t + pi) * cos(angle) +
                             a * cos(t + pi) * sin(angle))
            y_max = filterCoordinate(max(y1, y2), h)
            y_min = filterCoordinate(min(y1, y2), h)

            text = img_file + ',' + str(x_min) + ',' + str(y_min) + ',' + str(
                x_max) + ',' + str(y_max) + '\n'
            f.write(text)
        i = i + num_faces + 2
    f.close()


list_ellipse_files = glob.glob(ellipse_folder_files + "*-ellipseList.txt")

for ellipse_file in list_ellipse_files:
    print("ellipse_file: " + ellipse_file)
    fil_path = Path(ellipse_file)
    splitted_name = fil_path.name.split('.')[0].split('-')[:-1]
    splitted_name.append("rect")
    rect_new_name = '-'.join(splitted_name)
    rect_new_name += fil_path.suffix
    splitted_path = [v for v in fil_path.parts[:-1]]
    splitted_path.append(rect_new_name)

    rect_filename = PurePath(*splitted_path)

    print("new rect file name: " + str(rect_filename))
    build_values_from_ellipse_filename(folder_images, ellipse_file,
                                       str(rect_filename))
