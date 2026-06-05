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
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import model
import DataProcessFuncs as funcs
import json
from SphereAlignment import FitGaze


class loader(Dataset):

    def __init__(self, source, target, epoch, type):
        self.lines = []
        self.labels = []
        if type == 'train':
            prefix = 'TrainSet'
        elif type == 'test':
            prefix = 'Evaluation'
        else:
            raise ('Dataset Type not Defined Error')


        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.features = np.load(self.dir+f'/Evaluation/Baseline/[{prefix}][{source}-{target}][Epoch{str(epoch).zfill(2)}].npy')
        self.PGF = np.load(self.dir+f'/Evaluation/GPM/[{prefix}][{source}-{target}][Epoch{str(epoch).zfill(2)}][ISO].npy')
        self.features = self.features[:self.PGF.shape[0]]
        assert self.features.shape[0] == self.PGF.shape[0]
        assert self.features.shape[1] == 512
        assert self.PGF.shape[1] == 3
        print(f'[Training] Dataset Loaded: [{prefix}][{source}-{target}][Epoch{str(epoch).zfill(2)}]')
        print(f'[Training] Dataset length: {self.PGF.shape[0]}')

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.PGF[idx])


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
            self.IP = nn.DataParallel(model.IsoPropagator()).to(self.device)
        else:
            self.IP = model.IsoPropagator()

        print(f' Complete!')

        self.save_path = os.path.join(args.save_path, 'IsoPropagator')
        self.eval_path = os.path.join(args.eval_path, 'IsoPropagator')
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.epoch = args.epoch[0]
        self.total_epoch = args.epoch[1]
        self.epoch_step = args.epoch[2]
        self.experiment_log = open(os.path.join(self.save_path, f'[{args.source}]Experiment.log'), 'a')
        self.GPM_param = np.load(
            f'{args.save_path}/GPM/[TrainSet][{self.args.source}-{self.args.source}][Epoch{str(args.baseline_epoch).zfill(2)}][GPM].npy')
        with open(f'{args.eval_path}/Baseline/[Evaluation][{self.args.source}-{self.args.target}][Epoch{str(args.baseline_epoch).zfill(2)}].json', 'r') as f:
            self.baseline_log = json.load(f)
        # print(self.baseline_log.keys(), len(self.baseline_log['gaze_label']))
        if 'train' in args.phase:
            train_loader = loader(args.source, args.source, args.baseline_epoch, 'train')
            self.train_set = DataLoader(train_loader, batch_size=500, shuffle=True, num_workers=8)
            assert self.epoch == 1
            self.loss = nn.L1Loss().to(self.device)
            self.optimizer = torch.optim.Adam(self.IP.parameters(), args.lr, betas=(0.5, 0.95))
            self.train_loss_writer = SummaryWriter(self.save_path)



        if 'test' in args.phase:
            test_loader = loader(args.source, args.target, args.baseline_epoch, 'test')
            self.test_set = DataLoader(test_loader, batch_size=500, shuffle=False, num_workers=8)
        #     if 'vali' in args.phase:
        #         raise Exception('Phase Test and Vali conflict!')
        # elif 'vali' in args.phase:
        #     self.test_set = reader.loader_from_txt(args.target, args.dataset_paths, 'vali', args.image_size,
        #                                            300, shuffle=False)
        #     # self.test_set = self.train_set
        self.epochs_log = open(f'{self.eval_path}/[IP][{self.args.source}-{self.args.target}]Epochs.log', 'a')

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
        for self.epoch in range(self.args.epoch[0], self.args.epoch[1] + 1):
            if 'train' in args.phase:
                self.Train()
            if 'test' in args.phase and self.epoch%self.args.epoch[2]==0:

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
        self.IP.train()
        loss_sum = 0

        times = [time.time()]
        for i, batch in enumerate(self.train_set):
            data, label = batch
            data = data.to(self.device)
            label = label.to(self.device)
            # print(data['face'].shape, '---------------------------------')
            PGF = self.IP(data)

            loss = self.loss(PGF, label)
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
        st_save_path = f'{self.save_path}/Iter_{str(self.epoch).zfill(2)}_IP_{self.args.source}.pt'
        if self.epoch % self.epoch_step == 0:
            torch.save({'model': self.IP.state_dict(), 'optimizer': self.optimizer.state_dict()}, st_save_path)
        print(f'[Training][E {self.epoch:^3}/{self.total_epoch:^3}] Model saved at: {st_save_path}')

    def Test(self, prefix='Evaluation'):
        if not 'train' in self.args.phase:
            state_dict = torch.load(f'{self.save_path}/Iter_{str(self.epoch).zfill(2)}_IP_{self.args.source}.pt')
            self.IP.load_state_dict(state_dict['model'])

        dataset = self.test_set
        self.IP.eval()
        times = [time.time()]
        error_sum = 0
        count = 0
        feature_to_save = None
        log_dict = {'PGF_IP': [],
                    'PGF': [],
                    'error_IP': []
                    }

        with torch.no_grad():
            for i, batch in enumerate(dataset):
                data, label = batch
                data = data.to(self.device)
                labels = label.numpy()

                PGF_IP = self.IP(data)
                PGF_IP = PGF_IP.cpu().detach().numpy()


                log_dict['PGF_IP'] += PGF_IP.tolist()
                log_dict['PGF'] += labels.tolist()
                log_dict['error_IP'] += np.abs(PGF_IP-labels).tolist()
                angular_error = np.mean(np.abs(PGF_IP-labels))
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

        gaze_label = np.array(self.baseline_log['gaze_label'])
        _, gaze_PGF, _ = FitGaze(np.array(log_dict['PGF']), gaze_label, self.GPM_param)
        _, gaze_IP, _ = FitGaze(np.array(log_dict['PGF_IP']), gaze_label, self.GPM_param)
        gazeE_PGF2label = funcs.angular_batch(gaze_label, gaze_PGF)
        gazeE_IP2label = funcs.angular_batch(gaze_label, gaze_IP)
        gazeE_IP2PGF = funcs.angular_batch(gaze_IP, gaze_PGF)
        log_dict['gaze_PGF'] = gaze_PGF.tolist()
        log_dict['gaze_IP'] = gaze_IP.tolist()
        log_dict['gaze_label'] = gaze_label.tolist()
        log_dict['gaze_error_PGF2label'] = gazeE_PGF2label.tolist()
        log_dict['gaze_error_IP2label'] = gazeE_IP2label.tolist()
        log_dict['gaze_error_IP2PGF'] = gazeE_IP2PGF.tolist()

        log = f'[{prefix} {self.args.source}-{self.args.target}][Epoch {self.epoch:^3}/{self.total_epoch:^3}][Total Num: {count}]' \
              f'[Error {error_sum / count *180/np.pi:^5.2f}][PGF2label {np.mean(gazeE_PGF2label) *180/np.pi:^5.2f}][IP2label {np.mean(gazeE_IP2label) *180/np.pi:^5.2f}][IP2PGF {np.mean(gazeE_IP2PGF) *180/np.pi:^5.2f}]\n'

        with open(f'{self.eval_path}/[{prefix}][{self.args.source}-{self.args.target}][Epoch{str(self.epoch).zfill(2)}].json', 'w') as f:
            json.dump(log_dict, f)
        if prefix == 'Evaluation':
            self.epochs_log.write(log)
        print('\n' + log)
        return log


def get_args():
    parser = argparse.ArgumentParser(description='IsoGaze')
    parser.add_argument('--phase',  default='test')

    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--epoch', default='1,100,10')
    parser.add_argument('--baseline_epoch', default='10')
    # parser.add_argument('--decay_step', default=10, type=int)
    # parser.add_argument('--decay_ratio', default=0.1, type=float)
    parser.add_argument('--source', choices=['ETH', 'Gaze360'], default='ETH', type=str,
                        help="source dataset, eth/gaze360")
    parser.add_argument('--target', choices=['ETH', 'Gaze360', 'MPII', 'EyeDiapAll'], default='ETH', type=str,
                        help="source dataset, eth/gaze360")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--save_path', default=f'{current_dir}/Checkpoint/')
    parser.add_argument('--eval_path', default=f'{current_dir}/Evaluation/')

    parser.add_argument('--model_name', default='AGG')

    parser.add_argument('--shuffle', default=True, type=bool)
    args = parser.parse_args()

    args.phase = args.phase.split(',')
    args.epoch = list(map(int, args.epoch.split(',')))
    return args




if __name__ == "__main__":
    seed_everything(100)
    args = get_args()
    trainer = Trainer(args)
    trainer.run()
