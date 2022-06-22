import numpy as np


def feature_map_Gaussian(X, anchors, dim_rff):
    a = np.cos(X.dot(anchors))
    b = np.sin(X.dot(anchors))
    return 1 / np.sqrt(dim_rff) * np.concatenate((a, b), axis=1)

def get_anchors_gaussian_rff(dim_input, dim_rff, gamma):
    return gamma * np.random.normal(size=(dim_input, dim_rff))


class GaussianRFF(object):
    def __init__(self, dim_input, dim_rff, gamma):
        self.dim_rff = dim_rff
        self.anchors = get_anchors_gaussian_rff(
            dim_input, dim_rff, gamma)

    def feature_map(self, X):
        return feature_map_Gaussian(X, self.anchors, self.dim_rff)
