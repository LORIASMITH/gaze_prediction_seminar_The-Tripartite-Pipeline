import numpy as np
import os


def compare_GPM_param():
    base_dir = '/home/byw/Experiments/'
    param_path1 = os.path.join(base_dir, 'AGG_github/Checkpoint/GPM/[TrainSet][ETH-ETH][Epoch10][GPM].npy')
    param_path2 = os.path.join(base_dir, 'ISOMap_3/Checkpoint/IsoGaze/SphereModel/[TrainSet][ETH-ETH][Epoch10][SM].npy')

    param1 = np.load(param_path1)
    param2 = np.load(param_path2)
    print(param1)
    print(param2)


if __name__ == '__main__':
    compare_GPM_param()