import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
import cv2
import math
# import reader_disturb_blank as reader
from sklearn import manifold
import sklearn
import time
# from isomap_analyze_model import *
import torch
from itertools import chain
import pickle
import json

ETH_path = "E:/Dataset/eth"
Gaze360_path = "E:/Dataset/Gaze360/"
MPII_path = "E:/Dataset/MPIIFaceGaze/"
EyeDiap_path = "E:/Dataset/EyeDiap/"
dataset_paths = {'ETH': ETH_path,
                 'Gaze360': Gaze360_path,
                 'MPII': MPII_path,
                 'EyeDiap': EyeDiap_path}


def gazeTo2d(gaze):
    try:
        yaw = np.arctan2(-gaze[0], -gaze[2])
    except:
        print(gaze)
        exit()

    pitch = np.arcsin(np.clip(-gaze[1], -1, 1))
    return np.array([yaw, pitch])


def gazeto3d(gaze):
    assert len(gaze)==2
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def angular(gaze, label):
    if len(gaze)==2:
        gaze = gazeto3d(gaze)

    if len(label)==2:
        label = gazeto3d(label)
    total = np.sum(gaze * label)

    return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

def gazeTo3d_array(gaze):
    assert len(gaze.shape) == 2
    assert gaze.shape[1] == 2
    gaze_gt = np.zeros((gaze.shape[0], 3))
    gaze_gt[:, 0] = -np.cos(gaze[:, 1]) * np.sin(gaze[:, 0])
    gaze_gt[:, 1] = -np.sin(gaze[:, 1])
    gaze_gt[:, 2] = -np.cos(gaze[:, 1]) * np.cos(gaze[:, 0])
    return gaze_gt

def gazeTo2d_array(gaze):
    assert len(gaze.shape) == 2
    assert gaze.shape[1] == 3
    yaw = np.arctan2(-gaze[:, 0], -gaze[:, 2]).reshape((gaze.shape[0], 1))


    pitch = np.arcsin(-gaze[:, 1]).reshape((gaze.shape[0], 1))
    return np.hstack((yaw, pitch))

def angular_batch(gaze, label):
    if gaze.shape[1]==2:
        gaze = gazeTo3d_array(gaze)
    if label.shape[1] == 2:
        label = gazeTo3d_array(label)
    # print(gaze.shape, label.shape, '-----------------q')
    total = np.sum(gaze * label, axis=1)
    return np.arccos(np.clip(total/(np.linalg.norm(gaze, ord=2, axis=1)* np.linalg.norm(label, ord=2, axis=1)), -0.99999999, 0.99999999))

def AngularError(gaze, label):
    total = torch.sum(gaze * label, 1)
    cos_v = total/(torch.linalg.norm(gaze, 2, dim=1)*torch.linalg.norm(label, 2, dim=1))
    cos_v = cos_v - torch.clamp(cos_v - 1 + 1e-7, 0, 1)
    cos_v = cos_v - torch.clamp(cos_v + 1 - 1e-7, -1, 0)
    return torch.mean(torch.acos(cos_v))


def AngularError_array(gaze, label):
    if gaze.shape[1] == 2:
        gaze = gazeto3d_batch(gaze)
    if label.shape[1] == 2:
        label  = gazeto3d_batch(label)

    total = torch.sum(gaze * label, 1)

    cos_v = total/(torch.linalg.norm(gaze, 2, dim=1)*torch.linalg.norm(label, 2, dim=1))
    cos_v = cos_v - torch.clamp(cos_v - 1 + 1e-7, 0, 1)
    cos_v = cos_v - torch.clamp(cos_v + 1 - 1e-7, -1, 0)
    return torch.acos(cos_v)

def gazeto3d_batch(gaze):
    gaze_gt = torch.zeros((gaze.shape[0], 3)).to(gaze.device)
    gaze_gt[:, 0] = -torch.cos(gaze[:, 1]) * torch.sin(gaze[:, 0])
    gaze_gt[:, 1] = -torch.sin(gaze[:, 1])
    gaze_gt[:, 2] = -torch.cos(gaze[:, 1]) * torch.cos(gaze[:, 0])
    return gaze_gt

