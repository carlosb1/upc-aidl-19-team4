import tensorflow as tf
import numpy as np


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def pytorch_get_n_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def tf_get_n_params():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
