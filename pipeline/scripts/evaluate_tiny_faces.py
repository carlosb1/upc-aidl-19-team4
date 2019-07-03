# todo find some best way to fix imports
import sys
sys.path.append('../')
sys.path.append('../detections/Tiny_Faces_in_Tensorflow')
sys.path.append('../detections/db')

import cv2
import numpy as np
from detection_utils.metrics import compute_precision
import tensorflow as tf
from utils import tf_get_n_params

# from scripts folder

SHOW = True


def draw_and_show(img, gt_boxes, result_boxes):
    for box in gt_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    for box in result_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    if k == 27 or k == ord('q'):
        cv2.destroyAllWindows()
        return -1
    return 0


PATH_SOURCE_ANNOTATION = '/home/carlosb/python-workspace/upc-aidl-19-team4/datasets/FDDB-folds'
PATH_SOURCE_WIDER_TRAIN_ANNOTATION = '/home/carlosb/python-workspace/upc-aidl-19-team4/datasets/wider_face_split/wider_face_train_bbx_gt.txt'
PATH_SOURCE_WIDER_VALID_ANNOTATION = '/home/carlosb/python-workspace/upc-aidl-19-team4/datasets/wider_face_split/wider_face_val_bbx_gt.txt'
PATH_SOURCE_WIDER_TRAIN_IMAGES = '/home/carlosb/python-workspace/upc-aidl-19-team4/datasets/WIDER_train'
PATH_SOURCE_WIDER_VALID_IMAGES = '/home/carlosb/python-workspace/upc-aidl-19-team4/datasets/WIDER_val'


def get_fddb_dataset():
    from db.FDDB_dataset import FDDBDataset
    dataset = FDDBDataset()
    dataset.read(PATH_SOURCE_ANNOTATION)

    test_data = dataset.get_data()[1]
    images = test_data[0]
    bboxes = test_data[1]
    return images, bboxes


def get_wider_dataset():
    from db.WIDERFaceDataset import WIDERFaceDataset
    train_dataset = WIDERFaceDataset()
    train_dataset.read(PATH_SOURCE_WIDER_TRAIN_IMAGES, PATH_SOURCE_WIDER_TRAIN_ANNOTATION)

    valid_dataset = WIDERFaceDataset()
    valid_dataset.read(PATH_SOURCE_WIDER_VALID_IMAGES, PATH_SOURCE_WIDER_VALID_ANNOTATION)
    return valid_dataset.data()['images'], valid_dataset.data()['bboxes']


import sys

if len(sys.argv) > 1:
    images, bboxes = get_fddb_dataset()
else:
    images, bboxes = get_wider_dataset()

from tiny_face_eval import init, detect

weight_file_path = '../detections/Tiny_Faces_in_Tensorflow/model.pickle'
with tf.Session() as sess:
    model, average_image, clusters, clusters_h, clusters_w, normal_id, score_final, x = init(sess, weight_file_path)

    predictions = []
    time_exs = []
    for index_number_image in range(0, len(images)):
        boxes = bboxes[index_number_image]
        filename_img = images[index_number_image]
        img = cv2.imread(filename_img)
        # TODO only one method to load image
        detected_boxes, time_ex = detect(sess, model, filename_img, average_image, clusters, clusters_h, clusters_w, normal_id, score_final, x)
        detected_boxes = [result_box for result_box in detected_boxes]
        time_exs.append(time_ex)

        gt_boxes = []
        result_boxes = []
        for box in boxes:
            box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), 1]
            gt_boxes.append(box)
            print("gt: " + str(box))
        for detected_box in detected_boxes:
            formatted_yolo_box = [int(detected_box[0]), int(detected_box[1]), int(detected_box[2]), int(detected_box[3]), 1]
            result_boxes.append(formatted_yolo_box)
            print("detected: " + str(formatted_yolo_box))

        if SHOW:
            result = draw_and_show(img, gt_boxes, result_boxes)
            if result == -1:
                break
        # calculate precision
        gt_predic = np.array(gt_boxes)
        result_predic = np.array(result_boxes)
        predict = compute_precision(result_predic, gt_predic)
        predictions.append(predict)

    import statistics
    num_params = tf_get_n_params()
    print("num parameters: " + str(num_params))
    print("predic: " + str(statistics.mean(predictions)))
    print("time mean: " + str(statistics.mean(time_exs)))
