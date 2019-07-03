# Introduction

*student: Carlos Báez*

Project report for the Deep learning postgrade at UPC tech talent (Barcelona). This report explains all the work done, results and extracted conclusions

The main idea in the project was the implementation of an end-to-end person recognition system. For this I decided to split the project in two parts:

- **Detection**. Study of different implemented algorithms and different datasets to choose the best option for us
   
- **Face Recognition**. It is implemented and modified four different solutions with a saimese architecture.


# Structure Code
```
pipeline                                      -> Main source folder
├── detections                                -> Detection pipeline
│   ├── db                                    -> Datasets classes to configure dataset 
│   │   ├── constants.py                      -> Necessary constants for dataset classes
│   │   ├── FDDB_convert_ellipse_to_rect.py   -> Parser from ellipse to rectangle for FDDB dataset
│   │   ├── FDDB_dataset.py                   -> FDDB dataset class which loads in memory all the dataset information
│   │   ├── FDDB_show_images.py               -> Example to display FDDB dataset
│   │   └── WIDERFaceDataset.py               -> Wider dataset class example 
│   ├── Tiny_Faces_in_Tensorflow              -> Folder for Tiny Faces model
│   │   ├── tiny_face_eval.py                 -> Entrypoint for tiny_faces model (Tensorflow)
│   └── yolo                                  -> Folder for YOLO model 
│       └── yolo                              -> Folder for YOLO package
│           ├── model.py                      -> YOLO model
│           └── yolo.py                       -> YOLO class, functions to call inference and high level functions for detection
├── README.md                                 -> Main README.md
├── recognition                               -> Recognition pipeline
│   ├── cfp_dataset.py                        -> CFP Dataset class
│   ├── metrics.py                            -> Class to calculate threshold and accuracy
│   ├── metrics_retrieval.py                  -> Class to implement ranking
│   ├── models.py                             -> Class with different models
│   ├── params.py                             -> Builder params patterns to customize different tests
│   ├── parse_cfg_dataset.sh                  -> Script to fix dataset paths
│   ├── tests.py                              -> Class to execute different tests
│   ├── train.py                              -> Main class which train loop
│   ├── transforms.py                         -> Data augmentation classes
│   └── utils.py                              -> Functions for different use cases
└── scripts                                   -> General scripts
    ├── evaluate_tiny_faces.py                -> Script to execute and evaluate tiny faces
    ├── evaluate_yolo.py                      -> Script to execute and evaluate yolo
    ├── get_models.sh                         -> Download YOLO weights
    ├── README.md                             -> README with dependencies
    └── utils.py                              -> Other functions
scripts                                       -> General scripts
├── graphs_model.ipynb                        -> Draw seaborn bubble graph
├── local2remote.sh                           -> Script to upload from local to one server
├── print_graphs.ipynb                        -> Draw matplot graphs
├── remote2local.sh                           -> Script to download from remote to local 
├── train.ipynb                               -> Collab to set up environment and ssh connection
└── value_models.csv                          -> Information to print in graphs ()

```

# Documentation

## Detection
For the detection module. It is was studied and analysed two neural networks and two datasets:
- *Tiny Faces*
	- code https://github.com/cydonia999/Tiny_Faces_in_Tensorflow
	- paper https://arxiv.org/abs/1612.04402
- *YOLO v3*, trained for faces
	- code https://github.com/sthanhng/yoloface
	- paper https://pjreddie.com/media/files/papers/YOLOv3.pdf


For datasets, I did two different:

- *FDDB* Dataset http://vis-www.cs.umass.edu/fddb/
- *Wider* Dataset http://shuoyang1213.me/WIDERFACE/ 


In this graph, it is added the result of the analysis (accuracy, time and number of parameters for each network):

![alt text][bubbles]

## Recognition

It was implemented a Siamese Network with [VGG][vgg] features.
We implemented four different networks:
- Two siamese neural networks getting features from a VGG convolutional network and apply a cosine similarity
- Two siamese networks which a concatenation in order to join features and get a classification.

They are two type of experiments:
- Change optimizer SGD or ADAM (With different learning rates and weight decay) (1e-3, 5e-4)
- With and without data augmentation.


## Recognition architecture
The backend architecute is a VGG16-bn (batch normalized) and its convolutional layers. They are used as a siamese network applying them in two images and get their features. For this project, it is used pretrained networks that speed up our training process.

After this point, it is applied different techniques to check the performance and compare results:
- First one, it applies a cosine similarity loss function to search better results with the convolutional layers
	- v1 It is the simplest version, it only gets the VGG featurs
	- v2 In this version, it is added a linear version to flat the features, and it is trained.
- In the second one, it is joined the two branches to get a classificaiton. Furthemore, It is added  some improvements in order to fit better the analysed problem..
	- The neural network  named decision, it includes a minimal decision network with a few linear layers to decide. All this after to concatenate both features (from the two branches)
	- In the decision network linear, it is added a linear layer before the concatenation to improve the training and the performance. It tries to get better feature for our use case.



### VGG backend

In this image, I can preview the VGG architecture and its convolutional module and the linear layer from it is extracted the features
![alt_text][vgg_arch]

[VGG backend architecture][vgg_features]

### Siamese Cosine immplementation

In this graph, it is possible preview the all commented networks.

#### Siamese Cosine 1
![alt_text][siamese1_layers]
#### Siamese Cosine 2
![alt_text][siamese2_layers]
#### Decision
![alt_text][decision_layers]
#### Decision linear
![alt_text][decision_linear_layers]


## Conclusions

- In general, Siamese cosine v1 works better. 
- The Cosine similarity loss works better than any type cross entropy.
- **The best recipe: cosine v1 + SGD + Data augmentation + Triplet loss** [weights]
- Siamese cosine v2 doesn't reuse the linear VGG layer and it decrease its performance.
- Decision layers have problems to train with the dataset, the overfits appears very fast (7 or 6 epoch). It is very  important to tune params and add data augmentation.
- Adam optimizer has heavy problems if It must find the best tuned parameters
- In general, decision networks need more epochs to learn better due to train the decision network and new layers.



### Result table


|      Name       | SGD (val acc., test acc.)  | SGD + Data aug | Adam + Data aug |
|-----------------|----------------------------|----------------|-----------------|
| Cosine v1       | 81.14, 83.39               |   80.53, 83.71 |    73.03, 74.21 |
| Cosine v2       | 71.35, 71.57               |   73.03, 74.21 |    70.75, 71.81 |
| Decision        | 79.35, 80.46               |    80.6, 82.28 |    49.80, 50.00 |
| Decision linear | 78.28, 79.28               |   81.71, 82.28 |    76.75, 73.96 |

#### Triplet loss results (Best results)

|      Name       | SGD + Data aug | Adam + Data aug |
|-----------------|----------------|-----------------|
| Triplet v1      |   83.28, 86.32 |    81.71, 82.17 |

### Siamese cosine tests (V1 and V2)

#### Cosine networks SGD test
![siamese1_sara_sgd]
![siamese2_sara_sgd]

#### Cosine networks SGD + Data aug. test
![siamese1_sara_sgd_normtrans]
![siamese2_sara_sgd_normtrans]

#### Cosine networks ADAM + Data aug. test
![siamese1_sara_adam_normtrans]
![siamese2_sara_adam_normtrans]

### Cosine v1 with triplet loss! (SGD and ADAM) + Data augmentation
![triplet1_sara_sgd_normtrans]
![triplet1_sara_adam_normtrans]

### Decision networks (Decision and linear network)

#### Decision networks SGD 
![decision_sara_sgd]
![decision_linear_sara_sgd]

#### Decision networks SGD + Data aug
![decision_sara_sgd_normtrans]
![decision_linear_sara_sgd_normtrans]

#### Decision networks Adam + Data aug
![decision_sara_adam_normtrans]
![decision_linear_sara_adam_normtrans]

[bubbles]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/bubbles.png "Bubbles"





[weights]: https://drive.google.com/open?id=1s3Zj0PesMp2juGmS7ERd5GWvxuxk-u2D
[vgg]: https://arxiv.org/pdf/1409.1556.pdf
[decision_layers]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_layers.png "Decision layers"
[decision_linear_layers]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_linear_layers.png "Decision linear layers"
[decision_linear_sara_adam_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_linear_sara_adam_normtrans.png "Decision linear Adam  + Data Augmentation"
[decision_linear_sara_sgd]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_linear_sara_sgd.png "Decision linear SGD"
[decision_linear_sara_sgd_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_linear_sara_sgd_normtrans.png "Decision linear  SGD + Data augmentation"
[decision_sara_sgd]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_sara_sgd.png "Decision SGD"
[decision_sara_adam_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_sara_adam_normtrans_lr54.png "Decision Adam  + Data Augmentation"
[decision_sara_sgd_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_sara_sgd_normtrans_v2.png "Decision SGD  + Data Augmentation"
[siamese1_layers]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_layers.png "Siamese Cosine 1 layers"
[siamese1_sara_sgd]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_sara_sgd.png "Siamese Cosine 1 SGD"
[siamese1_sara_adam_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_sara_adam_normtrans_lr54.png "Siamese Cosine 1 Adam  + Data Augmentation"
[siamese1_sara_sgd_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_sara_sgd_normtrans.png  "Siamese Cosine 1 SGD  + Data Augmentation"
[siamese2_layers]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese2_layers.png  "Siamese 2 layers"
[siamese2_sara_adam_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese2_sara_adam_normtrans.png "Siamese Cosine 2 Adam  + Data Augmentation"
[siamese2_sara_sgd]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese2_sara_sgd.png  "Siamese Cosine 2 SGD"
[siamese2_sara_sgd_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese2_sara_sgd_normtrans.png  "Siamese Cosine 2  SGD  + Data Augmentation"
[vgg_arch]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/vgg_arch.png "VGG architecture"
[vgg_features]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/vgg_features.png "VGG features"

[triplet1_sara_adam_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_sara_triplet_adam_normtrans.png "Siamese Triplet 1 Adam  + Data Augmentation"
[triplet1_sara_sgd_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_sara_triplet_sgd_normtrans.png  "Siamese Triplet 1 SGD  + Data Augmentation"



