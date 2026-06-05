import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import yaml
import math
import time
import argparse
import random
import reader
from tensorboardX import SummaryWriter
import model
import DataProcessFuncs as funcs
import json


def set_seed(seed):
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass
    np.random.seed(seed)
    random.seed(seed)
    # cuda env
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def gazeto3d_batch(gaze):
    gaze_gt = torch.zeros((gaze.shape[0], 3)).to(gaze.device)
    gaze_gt[:, 0] = -torch.cos(gaze[:, 1]) * torch.sin(gaze[:, 0])
    gaze_gt[:, 1] = -torch.sin(gaze[:, 1])
    gaze_gt[:, 2] = -torch.cos(gaze[:, 1]) * torch.cos(gaze[:, 0])
    return gaze_gt


def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def angular(gaze, label):
    total = np.sum(gaze * label)
    return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi


def dis(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

class Trainer:
    def __init__(self, args):
        print(f'[Trainer Log] Model Name: \033[0;32;40m\t{args.model_name}\033[0m')
        self.args = args

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.GPU_num = torch.cuda.device_count()
            print(f'[Trainer Log] \033[0;32;40m\t{self.GPU_num} GPUs Detected \033[0m')
        else:
            self.device = torch.device('cpu')
            self.GPU_num = 0
            print(f'[Trainer Log] \033[0;32;40m\tNo GPU Detected. Run on CPU \033[0m')

        print(f'[Trainer Log] Model Building......', end='')
        sys.stdout.flush()

        if self.GPU_num > 1:
            self.model = nn.DataParallel(model.model(gaze_fc_num=3)).to(self.device)
        else:
            self.model = model.model(gaze_fc_num=3)

        print(f' Complete!')

        self.save_path = os.path.join(args.save_path, 'Baseline')
        self.eval_path = os.path.join(args.eval_path, 'Baseline')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)

        self.epoch = args.epoch[0]
        self.total_epoch = args.epoch[1]
        self.epoch_step = args.epoch[2]
        self.experiment_log = open(os.path.join(self.save_path, f'[{args.source}]Experiment.log'), 'a')
        if 'train' in args.phase:

            self.train_set = reader.loader_from_txt(args.source, args.dataset_paths, 'train', args.image_size,
                                                    args.batch_size, shuffle=True)
            assert self.epoch == 1
            self.loss = nn.L1Loss().to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), args.lr, betas=(0.5, 0.95))
            self.train_loss_writer = SummaryWriter(self.save_path)


        if 'test' in args.phase:
            if args.target == args.source:
                dataset_type = 'test'
            else:
                dataset_type = 'full'

            self.test_set = reader.loader_from_txt(args.target, args.dataset_paths, dataset_type, args.image_size,
                                                    300, shuffle=False)

        self.epochs_log = open(f'{self.eval_path}/[Baseline][{self.args.source}-{self.args.target}]Epochs.log', 'a')

    def run(self):
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()).replace(':', '-').replace(' ', '-')
        log = f"[Experiment Start Time] {start_time}"
        self.experiment_log.write(log)
        self.experiment_log.write(f'[Experiment args] {str(args)}')
        print(log)

        logs = []
        if 'train' in args.phase:
            if not os.path.exists(f'{self.save_path}/{start_time}'):
                os.mkdir(f'{self.save_path}/{start_time}')
            self.train_loss_writer = SummaryWriter(f"{self.save_path}/{start_time}")
        for self.epoch in range(self.args.epoch[0], self.args.epoch[1] + 1, self.args.epoch[2]):
            if 'train' in args.phase:
                self.Train()
                self.Test(prefix='TrainSet')
            if 'test' in args.phase:
                logs.append(self.Test())
            # if 'vali' in args.phase:
            #     logs.append(self.Test(prefix='Validation'))
        self.experiment_log.write(f"[Experiment Finish Time] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")
        self.epochs_log.close()
        if len(logs) > 0:
            for log in logs:
                print(log, end='')

    def Train(self):
        print(f'[Training][Epoch {self.epoch:^3}/{self.total_epoch:^3}] Training start...', end='')
        self.model.train()
        loss_sum = 0

        times = [time.time()]
        for i, batch in enumerate(self.train_set):
            data, label = batch
            data["face"] = data["face"].to(self.device)
            label = label.to(self.device)
            # print(data['face'].shape, '---------------------------------')
            gazes, _ = self.model(data['face'])

            loss = self.loss(gazes, gazeto3d_batch(label))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            times.append(time.time())
            if len(times) > 10:
                times.pop(0)
            batch_time = (times[-1] - times[0])/len(times)
            current_epoch_time = (len(self.train_set) - i - 1) * batch_time / 3600
            epochs_time = len(self.train_set) * batch_time * ((self.total_epoch - self.epoch)//self.epoch_step) / 3600
            print(f'\r[Training][E{self.epoch:^3}/{self.total_epoch:^3}][Batch {i}/{len(self.train_set)}]'
                  f'[L {loss:^6.3f}][Time: {current_epoch_time+epochs_time:^5.2f}h] ', end='')
            self.train_loss_writer.add_scalar('train_loss', loss, (self.epoch-1)*len(self.train_set)+i)
            loss_sum += loss.to('cpu').detach().numpy()

        print(f'\r[Training][E{self.epoch:^3}/{self.total_epoch:^3}][Batch {i}] Complete! Average loss: {loss_sum/(len(self.train_set)):^6.3f}')
        st_save_path = f'{self.save_path}/Iter_{str(self.epoch).zfill(2)}_Baseline_{self.args.source}.pt'
        if self.epoch % self.epoch_step == 0:
            torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, st_save_path)
        print(f'[Training][E {self.epoch:^3}/{self.total_epoch:^3}] Model saved at: {st_save_path}')

    def Test(self, prefix='Evaluation'):
        if not 'train' in self.args.phase:
            state_dict = torch.load(f'{self.save_path}/Iter_{str(self.epoch).zfill(2)}_Baseline_{self.args.source}.pt')
            self.model.load_state_dict(state_dict['model'])
        if prefix == 'Evaluation':
            dataset = self.test_set
        elif prefix == 'TrainSet':
            dataset = self.train_set
        self.model.eval()
        times = [time.time()]
        error_sum = 0
        feature_len_sum = 0
        count = 0
        feature_to_save = None
        log_dict = {'names': [],
                    'gaze_mlp': [],
                    'gaze_label': [],
                    'gaze_error': []
                    }
        # with open(f'{self.eval_path}/[{prefix}][{self.args.source}-{self.args.target}][Epoch{str(self.epoch).zfill(2)}].log', 'w') as current_epoch_log:
        #     current_epoch_log.write("name, x, y, labelx, labely, error, f_len\n")
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                data, label = batch
                data["face"] = data["face"].to(self.device)
                labels = label.numpy()

                gazes, feature = self.model(data['face'])
                if feature_to_save is None:
                    feature_to_save = feature.detach().cpu()
                else:
                    feature_to_save = torch.cat((feature_to_save, feature.detach().cpu()), dim=0)
                # angular_error = Trainer.Gen_test_log_batch(gazes, labels, data["name"], current_epoch_log)
                angular_error = Trainer.Gen_test_log_batch(gazes, labels, data["name"], log_dict)
                count += labels.shape[0]
                times.append(time.time())
                if len(times) > 10:
                    times.pop(0)
                batch_time = (times[-1] - times[0]) / len(times)
                current_epoch_time = (len(dataset) - i - 1) * batch_time / 3600
                epochs_time = len(dataset) * batch_time * (
                            (self.total_epoch - self.epoch) // self.epoch_step) / 3600
                print(
                    f'\r[{prefix} {self.args.target}][E {self.epoch:^3}/{self.total_epoch:^3}][Batch {i}/{len(dataset)}]'
                    f'[Error {angular_error / labels.shape[0] *180/np.pi:^5.2f}][Time: {current_epoch_time + epochs_time:^5.2f}h] ',
                    end='')
                error_sum += angular_error
                if prefix == 'TrainSet' and count >= args.feature_save_num:
                    break
        log = f'[{prefix} {self.args.source}-{self.args.target}][Epoch {self.epoch:^3}/{self.total_epoch:^3}][Total Num: {count}]' \
              f'[Error {error_sum / count *180/np.pi:^5.2f}]\n'
        # current_epoch_log.write(log)
        with open(f'{self.eval_path}/[{prefix}][{self.args.source}-{self.args.target}][Epoch{str(self.epoch).zfill(2)}].json', 'w') as f:
            json.dump(log_dict, f)
        if prefix == 'Evaluation':
            self.epochs_log.write(log)
        print('\n' + log)
        np.save(f'{self.eval_path}/[{prefix}][{self.args.source}-{self.args.target}][Epoch{str(self.epoch).zfill(2)}].npy', feature_to_save.numpy())
        return log

    @staticmethod
    def Gen_test_log_batch(gazes, labels, names, log_dict):

        gazes = gazes.cpu().detach().numpy()
        errors = funcs.angular_batch(gazes, labels)
        log_dict['names'] += names
        log_dict['gaze_mlp'] += gazes.tolist()
        log_dict['gaze_label'] += labels.tolist()
        log_dict['gaze_error'] += errors.tolist()
        return np.sum(errors)



def get_args():
    parser = argparse.ArgumentParser(description='IsoGaze')
    parser.add_argument('--phase',  default='test')

    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--epoch', default='1,10,1')
    # parser.add_argument('--decay_step', default=10, type=int)
    # parser.add_argument('--decay_ratio', default=0.1, type=float)
    parser.add_argument('--source', choices=['ETH', 'Gaze360', 'MPII', 'EyeDiapAll', 'ETHM', 'Gaze360M', 'Gaze360ISO', 'EVE'], default='ETH', type=str,
                        help="source dataset, eth/gaze360")
    parser.add_argument('--target', choices=['ETH', 'Gaze360', 'MPII', 'EyeDiap', 'ETHM', 'Gaze360M', 'EyeDiapAll', 'Gaze360ISO', 'EVE'], default='ETH', type=str,
                        help="source dataset, eth/gaze360")
    parser.add_argument('--feature_save_num', default=20000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--save_path', default=f'{current_dir}/Checkpoint/')
    parser.add_argument('--eval_path', default=f'{current_dir}/Evaluation/')

    parser.add_argument('--ETH_path', default="/home/byw/Dataset/eth")
    parser.add_argument('--Gaze360_path', default="/home/byw/Dataset/Gaze360/")
    parser.add_argument('--MPII_path', default="/home/byw/Dataset/MPIIFaceGaze/")
    parser.add_argument('--EyeDiapAll_path', default="/home/byw/Dataset/EyeDiap/")
    parser.add_argument('--model_name', default='AGG')

    parser.add_argument('--shuffle', default=True, type=bool)
    args = parser.parse_args()
    args.dataset_paths = {'ETH'    : args.ETH_path,
                          'Gaze360': args.Gaze360_path,
                          'MPII'   : args.MPII_path,
                          'EyeDiapAll': args.EyeDiapAll_path}
    args.phase = args.phase.split(',')
    args.epoch = list(map(int, args.epoch.split(',')))
    return args




if __name__ == "__main__":
    seed_everything(100)
    args = get_args()
    trainer = Trainer(args)
    trainer.run()
