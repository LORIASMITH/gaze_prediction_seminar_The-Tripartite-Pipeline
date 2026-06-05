import numpy as np
import cv2
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch
import json
import random
import copy
import pathlib


class loader(Dataset):

    def __init__(self, dataset_name, dataset_paths, dataset_type, image_size):
        self.lines = []
        self.labels = {}
        self.dataset_path = dataset_paths[dataset_name]
        self.type = dataset_type
        self.dataset_name = dataset_name
        self.image_size = image_size

        self.g = []

        subjects = os.listdir(f'{self.dataset_path}/Image')
        subjects.sort()
        assert dataset_type in ['train', 'test', 'full']
        if dataset_name == 'Gaze360':
            if dataset_type == 'train' or dataset_type == 'test':
                subjects = [dataset_type]
            else:
                subjects = ['test']
        elif dataset_name == 'ETH':
            if self.type == 'train':
                subjects = subjects[:75]
            elif self.type == 'test' or self.type == 'full':
                subjects = subjects[75:]

        elif dataset_name == 'MPII':
            assert self.type == 'full'

        elif dataset_name == 'EyeDiapAll':
            assert self.type == 'full'
        else:
            print(f'Reader of dataset [{dataset_name}] not implemented!')
            raise ('Unknown Dataset ERROR')

        for subject in subjects:
            with open(os.path.join(self.dataset_path, 'Label', subject + '.label'), 'r') as label_file:
                label_content = label_file.readlines()
                label_content.pop(0)
                for label_single_line in label_content:
                    label_single_line = label_single_line.split(' ')
                    for i in range(1, 3):
                        label_single_line[i] = list(map(float, label_single_line[i].split(',')))
                    self.lines.append(label_single_line)


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # Header
        #   0    1      2      3       4     5      6   7    8     9       10         11        12
        # face gaze2d head2d origin gaze3d head3d face gaze head origin cam_index frame_index normmat
        # features after 3 are all strings.

        line = self.lines[idx]
        face_img = cv2.imread(os.path.join(self.dataset_path, 'Image', line[0]))
        if face_img.shape[0] != self.image_size:
            face_img = cv2.resize(face_img, (self.image_size, self.image_size))
        label = line[1]
        face_img = face_img / 255.0
        face_img = face_img.transpose(2, 0, 1)
        img = {"face": torch.from_numpy(face_img).type(torch.FloatTensor),
               "head_pose": torch.tensor(line[2]).type(torch.FloatTensor),
               "name": line[0]}

        return img, torch.tensor(label).type(torch.FloatTensor)


def loader_from_txt(dataset_name, dataset_path, dataset_type, image_size, batch_size, shuffle=False, num_workers=24):
    print(f"[{dataset_name} Dataset] {dataset_type} Set Loading......")
    print(f"[{dataset_name} Dataset] Path: [{dataset_path[dataset_name]}]")
    dataset = loader(dataset_name, dataset_path, dataset_type, image_size)
    print(f"[{dataset_name} Dataset] Data Loaded! [Image Num: {len(dataset)}][Batch: {batch_size}][Size:{image_size}][Shuffle:{shuffle}]")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load



if __name__ == "__main__":
    ETH_path = "/home/byw/Dataset/eth"
    Gaze360_path = "/home/byw/Dataset/Gaze360/"
    MPII_path = "/home/byw/Dataset/MPIIFaceGaze/"
    EyeDiap_path = "/home/byw/Dataset/EyeDiap_filters/"
    dataset_paths = {'ETH': ETH_path,
                     'Gaze360': Gaze360_path,
                     'ETHM': ETH_path,
                     'Gaze360M': Gaze360_path,
                     'MPII': MPII_path,
                     'EyeDiap': EyeDiap_path}
    # # path = "/home/byw/Dataset/eth"
    # dataset_type = "train"
    # loader = loader_from_txt('EyeDiap', dataset_paths, dataset_type, batch_size=2)
    # for i, data in enumerate(loader):
    #     print(data[0]['face'][0].shape)
    #     print(data[1])
    #     break
    # loader = loader_from_txt('Gaze360', dataset_paths, 'train', 224,
    #                        20, shuffle=False)

    loader = loader_from_txt('Gaze360', dataset_paths, 'test', 224,
                           20, shuffle=False)



    # # loader = Warmup_loader_from_txt('ETH', 'ETH', 'ETH-res18-448', '/home/byw/Experiments/GazeUDA/Evaluation', [1,2,3], 4, shuffle=False)
    # for i, batch in enumerate(loader):
    #     data, label = batch
    #     print(label[0])
    #     exit()
    # #
    # #     break