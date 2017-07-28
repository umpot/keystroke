import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from numpy.linalg import cholesky
from scipy.spatial.distance import cityblock, mahalanobis, euclidean
from scipy.linalg import sqrtm

def create_test_matrixes():
    orig=[]
    for _ in range(10000):
        orig.append(np.random.random(3).reshape(3,1))

    transformed = []


    transition = np.array([
        [1,2,3],
        [2,0,1],
        [0,1,1]
    ])

    for x in orig:
        t = np.dot(transition, x)
        transformed.append(t)



    return np.hstack(orig), np.hstack(transformed), transition


def decorrelation_matrix_old(m):
    covv = np.cov(m, rowvar=True)
    covv = np.linalg.inv(covv)
    hol = np.linalg.cholesky(covv)
    hol = np.transpose(hol)

    return hol

def decorrelation_matrix(m):
    covv = np.cov(m, rowvar=True)
    hol = sqrtm(covv)
    # hol = hol.real
    hol = np.linalg.inv(hol)
    # hol = np.transpose(hol)

    return hol

def test_if_everything_works():
    orig, transformed, transition = create_test_matrixes()
    hol = decorrelation_matrix(transformed)

    decorelated = np.dot(hol, transformed)
    covv = np.cov(decorelated)
    print covv

    actual = covv
    expected = np.eye(3)
    diff = np.abs(actual - expected)

    assert np.any(diff<1e-10)


def manhaten_distance_after_decorelation(x,y, hol):
    x = np.dot(hol, x)
    y = np.dot(hol, y)

    return cityblock(x,y)
