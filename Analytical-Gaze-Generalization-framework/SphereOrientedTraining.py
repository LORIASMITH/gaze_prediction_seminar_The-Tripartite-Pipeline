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
from SphereAlignment import FitGaze


# 会改变gaze的值！！！！！！！！！！！！！！！！
def RecoverFeatureFitting(gaze, params):
    if gaze.shape[1]==3:
        gaze = funcs.gazeTo2d_array(gaze)
    assert len(params) == 14
    Rvec0 = params[:3]
    R0 = cv2.Rodrigues(Rvec0)[0]
    R0_inv = np.linalg.inv(R0)
    k0, b0, k1, b1 = params[6:10]

    sphere_center = params[10:13]
    sphere_radius = params[13]
    gaze[:, 0] = (gaze[:, 0]-b0)/k0
    # yaw_inv决定了在哪个象限
    gaze[:, 1] = (gaze[:, 1]-b1)/k1
    pts_rotated = funcs.gazeTo3d_array(gaze)

    # print(x)
    # feature = R0_inv.reshape(1, 3, 3) @ pts_rotated.reshape((gaze.shape[0], 3, 1)).reshape((gaze.shape[0], 3))
    feature = np.einsum('ij,bj->bi', R0_inv, pts_rotated)
    feature = feature / np.linalg.norm(feature, ord=2, axis=1, keepdims=True)
    feature = feature*sphere_radius + sphere_center
    return feature


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
            self.model = nn.DataParallel(model.model(gaze_fc_num=3)).to(self.device)
            self.IP = nn.DataParallel(model.IsoPropagator()).to(self.device)
        else:
            self.model = model.model(gaze_fc_num=3)
            self.IP = model.IsoPropagator()

        # state_dict = torch.load(f'{args.save_path}/Baseline/Iter_{args.baseline_epoch}_Baseline_{args.source}.pt')
        state_dict = torch.load(f'/home/byw/Experiments/ISOMap_3/Checkpoint/IsoGaze/Baseline/Iter_{args.baseline_epoch}_Baseline_{args.source}.pt')
        self.model.load_state_dict(state_dict['model'])

        # state_dict = torch.load(f'{args.save_path}/IsoPropagator/Iter_{args.IP_epoch}_IP_{args.source}.pt')
        state_dict = torch.load(f'/home/byw/Experiments/ISOMap_3/Checkpoint/IsoGaze/IsoPropagator/Iter_{args.IP_epoch}_IP_{args.source}.pt')
        self.IP.load_state_dict(state_dict['model'])

        # self.GPM_param = np.load(f'{args.save_path}/GPM/[TrainSet][{args.source}-{args.source}][Epoch{str(args.baseline_epoch).zfill(2)}][GPM].npy')
        self.GPM_param = np.load(f'/home/byw/Experiments/ISOMap_3/Checkpoint/IsoGaze/SphereModel/[TrainSet][{args.source}-{args.source}][Epoch{str(args.baseline_epoch).zfill(2)}][SM].npy')

        print(f' Complete!')

        # save_path: args.save_path/model_name

        self.save_path = os.path.join(args.save_path, 'SOT')
        self.eval_path = os.path.join(args.eval_path, 'SOT')

        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


        self.epoch = args.epoch[0]
        self.total_epoch = args.epoch[1]
        self.epoch_step = args.epoch[2]
        self.experiment_log = open(os.path.join(self.save_path, f'[{args.source}]Experiment.log'), 'a')

        self.train_set = reader.loader_from_txt(args.source, args.dataset_paths, 'train', args.image_size,
                                                args.batch_size, shuffle=True)
        if 'train' in args.phase:

            assert self.epoch == 1
            self.loss = nn.L1Loss().to(self.device)
            self.model.module.faceModel.fc1.weight.requires_grad = False
            self.model.module.faceModel.fc1.bias.requires_grad = False
            self.model.module.faceModel.fc2.weight.requires_grad = False
            self.model.module.faceModel.fc2.bias.requires_grad = False
            self.model.module.gaze_fc.weight.requires_grad = False
            self.model.module.gaze_fc.bias.requires_grad = False

            self.optimizer = torch.optim.Adam(self.model.parameters(), args.lr, betas=(0.5, 0.95))
            self.train_loss_writer = SummaryWriter(self.save_path)



        if 'test' in args.phase:
            if args.target == args.source:
                dataset_type = 'test'
            else:
                dataset_type = 'full'

            self.test_set = reader.loader_from_txt(args.target, args.dataset_paths, dataset_type, args.image_size,
                                                300, shuffle=False)

        self.epochs_log = open(f'{self.eval_path}/[SOT][{self.args.source}-{self.args.target}]Epochs.log', 'a')

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
            label = label.numpy()
            gazes, feature = self.model(data['face'])
            PGF_IP = self.IP(feature)
            PGF_label = torch.FloatTensor(RecoverFeatureFitting(copy.deepcopy(label), self.GPM_param)).to(self.device)

            loss = self.loss(PGF_IP, PGF_label)
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
        st_save_path = f'{self.save_path}/Iter_{str(self.epoch).zfill(2)}_SOT_{self.args.source}.pt'
        if self.epoch % self.epoch_step == 0:
            torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, st_save_path)
        print(f'[Training][E {self.epoch:^3}/{self.total_epoch:^3}] Model saved at: {st_save_path}')

    def Test(self, prefix='Evaluation'):
        if not 'train' in self.args.phase:
            state_dict = torch.load(f'{self.save_path}/Iter_{str(self.epoch).zfill(2)}_SOT_{self.args.source}.pt')
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
                    'gaze_MLP': [],
                    'gaze_IP_GPM': [],
                    'gaze_label': [],
                    'PGF_IP': [],
                    'PGF_label': [],
                    'gaze_error_MLP':[],
                    'gaze_error_IP_GPM': [],
                    'error_IP': []
                    }
        # with open(f'{self.eval_path}/[{prefix}][{self.args.source}-{self.args.target}][Epoch{str(self.epoch).zfill(2)}].log', 'w') as current_epoch_log:
        #     current_epoch_log.write("name, x, y, labelx, labely, error, f_len\n")
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                data, label = batch

                data["face"] = data["face"].to(self.device)
                labels = label.numpy()



                gazes, feature = self.model(data['face'])
                PGF_IP = self.IP(feature)

                PGF_label = RecoverFeatureFitting(copy.deepcopy(label), self.GPM_param)

                if feature_to_save is None:
                    feature_to_save = feature.detach().cpu()
                else:
                    feature_to_save = torch.cat((feature_to_save, feature.detach().cpu()), dim=0)


                log_dict['names'] += data['name']
                log_dict['gaze_MLP'] += gazes.detach().cpu().numpy().tolist()
                log_dict['gaze_label'] += labels.tolist()
                log_dict['PGF_IP'] += PGF_IP.detach().cpu().numpy().tolist()
                log_dict['PGF_label'] += PGF_label.tolist()

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
                    f'[Time: {current_epoch_time + epochs_time:^5.2f}h] ',
                    end='')
                if prefix == 'TrainSet' and count >= args.feature_save_num:
                    break
        labels = np.array(log_dict['gaze_label'])
        _, gaze_IP_GPM, _ = FitGaze(np.array(log_dict['PGF_IP']), labels, self.GPM_param, verbose=False)
        # print('target: ', log_dict['PGF_IP'][:2])
        # print('result: ', RecoverFeatureFitting(gaze_IP_GPM[:2], self.GPM_param))
        # exit()
        gaze_error_IP_GPM = funcs.angular_batch(labels, gaze_IP_GPM)
        gaze_error_MLP = funcs.angular_batch(labels, np.array(log_dict['gaze_MLP']))
        log_dict['gaze_IP_GPM'] = gaze_IP_GPM.tolist()
        log_dict['gaze_error_IP_GPM'] = gaze_error_IP_GPM.tolist()
        log_dict['gaze_error_MLP'] = gaze_error_MLP.tolist()
        error_IP = np.abs(np.array(log_dict['PGF_IP']) - np.array(log_dict['PGF_label']))
        log_dict['error_IP'] = error_IP.tolist()


        log = f'[{prefix} {self.args.source}-{self.args.target}][Epoch {self.epoch:^3}/{self.total_epoch:^3}][Total Num: {count}]' \
              f'[E_IP_GPM {np.mean(gaze_error_IP_GPM) *180/np.pi:^5.2f}][E_MLP {np.mean(gaze_error_MLP) *180/np.pi:^5.2f}][E_IP {np.mean(error_IP):^5.2f}]\n'
        # current_epoch_log.write(log)
        with open(f'{self.eval_path}/[{prefix}][{self.args.source}-{self.args.target}][Epoch{str(self.epoch).zfill(2)}].json', 'w') as f:
            json.dump(log_dict, f)
        # print(f'[Debugggggggggggg][Json path]:{self.eval_path}/[{prefix}][{self.args.source}-{self.args.target}][Epoch{str(self.epoch).zfill(2)}].json')
        # print(log_dict['gaze_label'][0])
        if prefix == 'Evaluation':
            self.epochs_log.write(log)
        print('\n' + log)
        np.save(f'{self.eval_path}/[{prefix}][{self.args.source}-{self.args.target}][Epoch{str(self.epoch).zfill(2)}].npy', feature_to_save.numpy())
        return log





def get_args():
    parser = argparse.ArgumentParser(description='IsoGaze')
    parser.add_argument('--phase',  default='test')

    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--epoch', default='1,10,1')
    parser.add_argument('--baseline_epoch', default=10, type=int)
    parser.add_argument('--IP_epoch', default=100, type=int)
    # parser.add_argument('--decay_step', default=10, type=int)
    # parser.add_argument('--decay_ratio', default=0.1, type=float)
    parser.add_argument('--source', choices=['ETH', 'Gaze360'], default='ETH', type=str,
                        help="source dataset, eth/gaze360")
    parser.add_argument('--target', choices=['ETH', 'Gaze360', 'MPII', 'EyeDiapAll'], default='ETH', type=str,
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
