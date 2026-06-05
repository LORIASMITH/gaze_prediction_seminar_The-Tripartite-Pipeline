import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import os

import time
import argparse
import random
import reader
import model
import DataProcessFuncs as funcs
import json
from SphereAlignment import ISOMap, FitGaze
import pickle


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class AGG_Inference:
    def __init__(self, args):
        print(f'[Log] Test Phase: \033[0;32;40m\t{args.phase}\033[0m')
        self.args = args
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.GPU_num = torch.cuda.device_count()
            print(f'[Log] \033[0;32;40m\t{self.GPU_num} GPUs Detected \033[0m')
        else:
            self.device = torch.device('cpu')
            self.GPU_num = 0
            print(f'[Log] \033[0;32;40m\tNo GPU Detected. Run on CPU \033[0m')

        print(f'[Log] Model Building......', end='')

        
        sys.stdout.flush()
        if self.GPU_num > 1:
            self.model = nn.DataParallel(model.model(gaze_fc_num=3)).to(self.device)
        else:
            self.model = model.model(gaze_fc_num=3)

        ckpt_path = './ckpt'
        if args.phase == 'Baseline':
            self.model.load_state_dict(torch.load(f'{ckpt_path}/Baseline_{args.source}.pt')['model'])
            self.inference_method = self.Inference_baseline
        elif args.phase == 'GPM':
            self.model.load_state_dict(torch.load(f'{ckpt_path}/Baseline_{args.source}.pt')['model'])
            self.inference_method = self.Inference_AGG
        elif args.phase == 'SOT':
            self.model.load_state_dict(torch.load(f'{ckpt_path}/SOT_{args.source}.pt')['model'])
            self.inference_method = self.Inference_AGG
        else:
            raise ValueError(f'Invalid phase: {args.phase}')
        
        with open(f'{ckpt_path}/Isomap_{args.source}', 'rb') as f:
            self.ISO_fitter = pickle.load(f)
        self.GPM_param = np.load(f'{ckpt_path}/GPM_{args.source}.npy')

        print(f' Complete!')

        self.test_set = reader.loader_from_txt(args.target, args.dataset_paths, 'full', args.image_size,
                                                args.batch_size, shuffle=False)


    def run(self):
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()).replace(':', '-').replace(' ', '-')
        print(f"[Experiment Start Time] {start_time}")
        self.Test()
        print(f"[Experiment Finish Time] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")


    def Test(self):

        dataset = self.test_set
        self.model.eval()
        times = [time.time()]
        error_sum = 0
        count = 0
        feature_to_save = None
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                data, label = batch
                data["face"] = data["face"].to(self.device)
                labels = label.numpy()

                gazes = self.inference_method(data['face'])

                angular_error = np.sum(funcs.angular_batch(gazes, labels))

                
                count += labels.shape[0]
                times.append(time.time())
                if len(times) > 10:
                    times.pop(0)
                batch_time = (times[-1] - times[0]) / len(times)
                current_epoch_time = (len(dataset) - i - 1) * batch_time / 60
                # print(f'[{self.args.phase}][{self.args.source}-{self.args.target}][Batch {i+1}/{len(dataset)}]')
                # print(f'[Error {angular_error / labels.shape[0] *180/np.pi}] ')
                # print(f'[Time: {current_epoch_time:^5.2f}h] ')
                print(
                    f'\r[{self.args.phase}][{self.args.source}-{self.args.target}][Batch {i+1}/{len(dataset)}]'
                    f'[Error {angular_error / labels.shape[0] *180/np.pi:^5.2f}][Time: {current_epoch_time:^5.2f}min] ',
                    end='')
                error_sum += angular_error
        log = f'[{self.args.phase}][{self.args.source}-{self.args.target}][Total Num: {count}]' \
              f'[Error {error_sum / count *180/np.pi:^5.2f}]\n'
        print('\n' + log)
        return

    def Inference_baseline(self, faces):
        gazes, features = self.model(faces)
        return gazes.cpu().detach().numpy()
    
    def Inference_AGG(self, faces):
        gazes, features = self.model(faces)
        features = features.cpu().detach().numpy()
        PGF, fitter = ISOMap(features, dim=3, n_neighbors=300, fitter=self.ISO_fitter)
        _, gazes, _ = FitGaze(PGF, None, param=self.GPM_param, verbose=False)
        return gazes





def get_args():
    parser = argparse.ArgumentParser(description='AGGInference')
    parser.add_argument('--phase', choices=['Baseline', 'GPM', 'SOT'], default='ETH', type=str)
    parser.add_argument('--source', choices=['ETH', 'Gaze360'], default='ETH', type=str)
    parser.add_argument('--target', choices=['MPII', 'EyeDiapAll'], default='ETH', type=str)

    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--image_size', default=224, type=int)

    parser.add_argument('--ETH_path', default="/home/byw/Dataset/eth")
    parser.add_argument('--Gaze360_path', default="/home/byw/Dataset/Gaze360/")
    parser.add_argument('--MPII_path', default="/home/byw/Dataset/MPIIFaceGaze/")
    parser.add_argument('--EyeDiapAll_path', default="/home/byw/Dataset/EyeDiap/")

    args = parser.parse_args()
    args.dataset_paths = {'ETH'    : args.ETH_path,
                          'Gaze360': args.Gaze360_path,
                          'MPII'   : args.MPII_path,
                          'EyeDiapAll': args.EyeDiapAll_path}
    return args




if __name__ == "__main__":
    seed_everything(100)
    args = get_args()
    Tester = AGG_Inference(args)
    Tester.run()
