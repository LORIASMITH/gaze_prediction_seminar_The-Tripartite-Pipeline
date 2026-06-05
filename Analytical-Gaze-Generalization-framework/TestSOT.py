import sklearn
import numpy as np
import json
import os
import cv2
import sys
import random
import DataProcessFuncs as funcs
from scipy.optimize import leastsq, least_squares
from cv2 import Rodrigues
import pickle
import argparse
import time
from SphereAlignment import FitGaze, ISOMap

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True



# self.save_path
class SOT_Test():
    def __init__(self, args):
        self.args = args
        print(
            f'[SOT_Test][Train: {self.args.source}-Test: {self.args.target} E{str(args.epoch[0]).zfill(2)}-{str(args.epoch[1]).zfill(2)}]')
        logs = []
        for epoch in range(args.epoch[0], args.epoch[1] + 1, args.epoch[2]):
            self.source_name = f'[TrainSet][{args.source}-{args.source}][Epoch{str(epoch).zfill(2)}]'
            self.target_name = f'[Evaluation][{args.source}-{args.target}][Epoch{str(epoch).zfill(2)}]'

            self.eval_path = os.path.join(args.eval_path, 'SOTTest')
            if not os.path.exists(self.eval_path):
                os.makedirs(self.eval_path)
            self.ISO_fitter_path = os.path.join(args.save_path, 'GPM', f'[TrainSet][{args.source}-{args.source}][Epoch10][ISO]')
            self.train_PGF_path = None
            self.test_PGF_path = os.path.join(args.eval_path, 'SOTTest', f'{self.target_name}[ISO].npy')
            self.test_log_path = os.path.join(args.eval_path, 'SOTTest', f'{self.target_name}[GPM].json')
            self.GPM_path = os.path.join(args.save_path, 'GPM', f'[TrainSet][{args.source}-{args.source}][Epoch10][GPM].npy')

            with open(self.ISO_fitter_path, 'rb') as f:
                self.ISO_fitter = pickle.load(f)
            print(f'[SOT_Test] Isomap Fitter Found and Load: {self.ISO_fitter_path}')
            self.GPM_param = np.load(self.GPM_path)
            logs.append(self.Test(epoch))
            with open(os.path.join(self.eval_path,
                                    f'[Evaluation][{self.args.source}-{self.args.target}][GPM][epochs].log'),
                        'a') as f:
                f.write(logs[-1])

            for log in logs:
                print(log, end='')
                # f.write(log)


    def Test(self, epoch):
        print(f'[SOT_Test][Test][{self.args.source}-{self.args.target}][e{str(epoch).zfill(0)}/{self.args.epoch[1]}]')
        # test_log_path = os.path.join(self.args.eval_path, 'Baseline', f'[Evaluation][{self.args.source}-{self.args.target} E{str(epoch).zfill(2)}]')
        test_log_path = os.path.join(self.args.eval_path, 'SOT',
                                     f'[Evaluation][{self.args.source}-{self.args.target}][Epoch{str(epoch).zfill(2)}]')
        feature = np.load(test_log_path+'.npy')
        with open(test_log_path + '.json', 'r') as f:
            test_log = json.load(f)
        print(f"[SOT_Test][Test]{len(test_log['names'])} features Load")
        print(f"[SOT_Test][Test]{len(test_log['names'])} {test_log_path}")
        label = np.array(test_log['gaze_label'])
        test_PGF, _ = ISOMap(feature, fitter=self.ISO_fitter)
        _, gaze_pred, model = FitGaze(test_PGF, label, self.GPM_param)

        np.save(self.test_PGF_path, test_PGF)
        print(f'[SOT_Test] Test PGF Saved: {self.test_PGF_path}')
        errors = funcs.angular_batch(label, gaze_pred)
        log_dict = {'names': test_log['names'], 'gaze_label': test_log['gaze_label'], 'gaze_GPM': gaze_pred.tolist(), 'gaze_error': errors.tolist()}
        with open(self.test_log_path, 'w') as f:
            json.dump(log_dict, f)
        log = f'[SOT_Test][Test][{self.args.source}-{self.args.target} E{str(epoch).zfill(2)}][TrainNum:{self.args.num}][Num:{feature.shape[0]}][Error {np.mean(errors)*180/np.pi:^5.2f}]\n'
        print(log+'\n\n')
        return log
    # @staticmethod
    # def Gen(gazes, labels, names, log_dict):

def get_args():
    parser = argparse.ArgumentParser(description='IsoGaze-GPM')
    parser.add_argument('--phase',  default='test')
    # parser.add_argument('--train_epoch', default=10, type=int)
    parser.add_argument('--epoch', default='1,10,1')
    parser.add_argument('--source', choices=['ETH', 'Gaze360'], default='ETH', type=str,
                        help="source dataset, eth/gaze360")
    parser.add_argument('--target', choices=['ETH', 'Gaze360', 'MPII', 'EyeDiapAll'], default='ETH', type=str,
                        help="source dataset, eth/gaze360")
    parser.add_argument('--num', default=2000, type=int)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--save_path', default=f'{current_dir}/Checkpoint/')
    parser.add_argument('--eval_path', default=f'{current_dir}/Evaluation/')
    args = parser.parse_args()
    args.phase = args.phase.split(',')
    args.epoch = list(map(int, args.epoch.split(',')))
    return args


if __name__ == '__main__':
    seed_everything(100)
    args = get_args()
    SOT_Test(args)