import datetime
import os
import platform
import sys
from threading import Thread

import h5py as h5py
from sklearn.decomposition import PCA


class ModelInfo:
    def __init__(self, id_, name, meta=0, visible=True, weight=None):
        self.id = id_
        self.name = name
        self.meta = meta
        self.visible = visible
        self.weight = weight


MODELS = {x.id: x for x in [
    ModelInfo('na', 'CE'),
    ModelInfo('if', 'IF', weight='if'),
    ModelInfo('en', 'EN', weight='en'),
    ModelInfo('meta', 'DA', 10, weight='en'),
    ModelInfo('daen', 'DAEN', 10, weight='en'),
    ModelInfo('dadv', 'DADV', 10, weight='dv-N'),
    ModelInfo('sdv', 'DV', weight='dv-N'),
    ModelInfo('spv', 'PV', visible=True, weight='pv-N'),
    ModelInfo('sbv', 'BV', visible=True, weight='bv-N'),
]}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.utils.data.sampler import SubsetRandomSampler
# import matplotlib.pyplot as plt
from data_utils import *
from resnet import *
import shutil

TRAIN_FEATURE = 0
TEST_FEATURE = 1
META_FEATURE = 2
PRE_TRAIN_FEATURE = 3

"""
model: na, en, meta, dv, sdv, if, vv, b
  na: No class-balance weighting.
  if: Class-balance weighted by inverse frequency.
  en: Class-balance weighted by Effective Number.
  meta: Class-balance weighted by META training.
  daen: Domain Adaptation with Effective Number.
  dadv: Domain Adaptation with Distribution Volume.
  sdv: Class-balance weighted by Sqrt_n of Distribution Volume.
  spv: Sqrt_n of PCA volume.
  sbv: Sqrt_n of Boundary Volume.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Imbalanced Example')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset (cifar10 [default] or cifar100)')
    parser.add_argument('--model', default='na', type=str,
                        help='model (na [default], en, meta, sdv, if)')
    parser.add_argument('--w_epoch', type=int, default=0,
                        help='Epoch start to apply weights (default: 0)')
    parser.add_argument('--w_norm', type=str, default='N',
                        help='Normalize the weight'
                             '(N: [default], sum of weights = number of classes; '
                             'E: Expected value of weights = 1)')
    parser.add_argument('--n_pca', type=int, default=0,
                        help='Number of dimensions to keep for PCA volume, 0 means all. (default: 0)')
    parser.add_argument('--min_dv_sample', type=int, default=0,
                        help='Minimum number of samples to calculate distribution volume. (default: 0)')
    parser.add_argument('--up_sampling', default=False, type=bool,
                        help='Up-sampling minority classes to min_dv_sample samples.')
    parser.add_argument('--check_data', default=False, type=bool,
                        help='Check dataset.')
    parser.add_argument('--beta', default=0.9999, type=float,
                        help='beta (0, 1)')
    parser.add_argument('--loss', default='sm', type=str,
                        help='loss type (sm [default], sgm, or focal)')
    parser.add_argument('--feature_file', default='', type=str,
                        help='output feature file')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_meta', type=int, default=10,
                        help='The number of meta data for each class.')
    parser.add_argument('--add_meta', type=int, default=0,
                        help='Add meta data to training set if not used.')
    parser.add_argument('--imb_factor', type=float, default=0.1)
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_schedule', default='da', type=str,
                        help='learning rate schedule (da, en)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--split', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')

    args = parser.parse_args()
    model_info = MODELS[args.model]
    if model_info.meta:
        args.num_meta = args.num_meta or 10
        args.w_epoch = args.w_epoch or 160
    print('args:', args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    return args


class CommandManager:
    RUN = 'run'
    PAUSE = 'pause'
    STOP = 'stop'

    def __init__(self):
        self._thread = Thread(self.run)
        self._thread.start()
        self._save = False
        self._state = self.RUN

    def run(self):
        while not self.stop:
            line = sys.stdin.readline().strip()
            if line == 'pause':
                self._state = self.PAUSE
            elif line == 'resume':
                self._state = self.RUN
            elif line == 'stop':
                self._state = self.STOP
            elif line == 'save':
                self._save = True

    def should_save(self):
        while self._pause:
            return self._save

    def should_stop(self):
        return self._stop


class DistVolEstimator:
    def __init__(self, model, n_cls, n_pca=0, min_sample=0):
        self.n_pca = n_pca or n_cls
        self.min_sample = min_sample
        self.n_cls = n_cls
        self.y = [[] for _ in range(self.n_cls)]
        ts = model.weight.split('-')
        self.sqrt = 1
        if len(ts) == 2:
            self.model = ts[0]
            if ts[1] == 'N':
                self.sqrt = self.n_pca if self.model == 'pv' else (
                        self.n_cls - (self.model == 'bv'))
            elif ts[1] != '1':
                raise NotImplementedError()
        else:
            self.model = ts[0]
        if self.model in {'dv', 'pv', 'bv'}:
            self.stage = TRAIN_FEATURE
        elif self.model == 'vv':
            self.stage = TEST_FEATURE
        else:
            raise NotImplementedError()
        if self.model == 'bv':
            self.proj = np.zeros((self.n_cls, self.n_cls, self.n_cls - 1))
            for i in range(self.n_cls):
                self.proj[i, i, :] = 1
                for k in range(i):
                    self.proj[i, k, k] = -1
                for k in range(i + 1, self.n_cls):
                    self.proj[i, k, k - 1] = -1
        print('dist vol:', self.model, self.sqrt, self.stage)

    def add(self, stage, y_f, y):
        if self.stage != stage:
            return
        for cls, v in zip(y, y_f):
            self.y[cls].append(v)

    def update(self, stage, epoch, cls_n):
        if self.stage != stage:
            return
        # print('update:', stage, epoch, self.model, self.sqrt)
        mu, sig, vol = None, None, None
        # print(self.model, self.stage, [len(y) for y in self.y])
        if self.model in {'dv', 'vv'}:
            mu = np.asarray([np.mean(y, axis=0) for y in self.y])  # [1xn]
            sig = np.asarray([np.std(y, axis=0) for y in self.y])  # [1xn]
        elif self.model == 'pv':
            sig = np.empty((self.n_cls, self.n_pca))
            for cls, y in enumerate(self.y):
                pca = PCA()
                pca.fit(y)
                if pca.explained_variance_.shape[0] >= self.n_pca:
                    sig[cls] = pca.explained_variance_[:self.n_pca]
                else:
                    m = pca.explained_variance_.shape[0]
                    sig[cls][:m] = pca.explained_variance_
                    sig[cls][m:] = pow(np.prod(pca.explained_variance_), 1 / m)
        elif self.model in {'bv'}:
            mu = np.empty((self.n_cls, self.n_cls - 1))
            sig = np.empty((self.n_cls, self.n_cls - 1))
            for cls, y in enumerate(self.y):
                y_p = np.matmul(y, self.proj[cls])
                mu[cls] = np.mean(y_p, axis=0)
                sig[cls] = np.std(y_p, axis=0)
        else:
            raise NotImplementedError()
        vol = np.sum(np.log(sig), axis=1)
        if self.min_sample:
            ok = np.asarray([len(y) > self.min_sample for y in self.y])
            if not ok.all():
                avg_vol = np.mean(vol[ok])
                vol[np.logical_not(ok)] = avg_vol
        if self.sqrt == 1:
            w = np.exp(vol) / cls_n
        else:
            w = np.exp(vol / self.sqrt) / cls_n
        if np.isinf(w).any():
            print('got inf:')
            idx_list = [idx for idx, v in enumerate(np.isinf(w)) if v]
            for idx in idx_list:
                print(f'vol[{idx}]: {vol[idx]}')
                print(f'  sig[{idx}]: {sig[idx]}')
                print(f'  n[{idx}]: {self.y[idx].shape}')
        self.y = [[] for _ in range(self.n_cls)]
        return w, mu, sig


class DatasetManager:
    def __init__(self, args):
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(args.seed)

        self.args = args
        num_cls, train_data, meta_data, test_data = self._get_datasets(args)
        if args.min_dv_sample and args.up_sampling:
            self._up_sampling(train_data, num_cls, args.min_dv_sample)

        kwargs = {'num_workers': 4, 'pin_memory': True}
        if platform.system() == 'Windows':
            kwargs['num_workers'] = 1
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                        shuffle=True, **kwargs)
        self.meta_loader = torch.utils.data.DataLoader(meta_data, batch_size=args.batch_size,
                                                       shuffle=True,
                                                       **kwargs) if len(meta_data.data) else None
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                                       shuffle=False, **kwargs)

        print('#train samples:', len(train_data.data))
        print('#meta samples:', len(meta_data.data))
        print('#steps per epoch:', len(self.train_loader))

        self.cls_n = get_cls_num_images(train_data, num_cls)
        print('cls count:', self.cls_n)

    @staticmethod
    def _get_num_samples(loader):
        return len(loader.dataset.data) if loader else 0

    @property
    def n_meta(self):
        return self._get_num_samples(self.meta_loader)

    @property
    def n_train(self):
        return self._get_num_samples(self.train_loader)

    @property
    def n_test(self):
        return self._get_num_samples(self.test_loader)

    @classmethod
    def _get_datasets(cls, args):
        model = MODELS[args.model]
        meta_data, train_data, test_data, num_cls = build_dataset(args.dataset, args.num_meta * (
                model.id == 'meta' or args.add_meta))
        if platform.system() == 'Windows':
            cls._to_int64(train_data)
            cls._to_int64(meta_data)
        if not isinstance(train_data, EnImageDataset):
            data_list = get_cls_img_id_list(train_data, num_cls)
            if model.id == 'meta' or args.add_meta:
                img_num_list = get_img_num_per_cls(args.dataset, args.imb_factor,
                                                   args.num_meta * num_cls)
            else:
                img_num_list = get_img_num_per_cls(args.dataset, args.imb_factor, 0)
            train_take = []
            print('meta_data:', len(meta_data.data))
            meta_take = [] if args.num_meta > len(meta_data.data) else None
            for cls_idx, img_id_list in enumerate(data_list):
                random.shuffle(img_id_list)
                img_num = img_num_list[int(cls_idx)]
                if meta_take is None:
                    train_take.extend(img_id_list[:img_num])
                elif img_num > args.num_meta:
                    meta_take.extend(img_id_list[:args.num_meta])
                    train_take.extend(img_id_list[args.num_meta:img_num])
                else:
                    meta_take.extend(img_id_list[:img_num])
            meta_data = create_sub_dataset(train_data, meta_take) if meta_take else meta_data
            train_data = create_sub_dataset(train_data, train_take)
        img_num_list = get_cls_num_images(train_data, num_cls)
        if args.add_meta and model.id != 'meta':
            print('add meta')
            train_data = merge_dataset(train_data, meta_data)

        print('img_num_list:', img_num_list)
        print('total image_num:', sum(img_num_list))

        return num_cls, train_data, meta_data, test_data

    @staticmethod
    def _up_sampling(dataset, num_cls, min_sample):
        ext = []
        cls_img_id_list = get_cls_img_id_list(dataset, num_cls)
        for j, img_id_list in enumerate(cls_img_id_list):
            pad = min_sample - len(img_id_list)
            while pad > 0:
                random.shuffle(img_id_list)
                ext.extend(img_id_list[:min(pad, len(img_id_list))])
                pad -= len(img_id_list)
        print('up:', len(ext), ext)
        if ext:
            up_sample_dataset(dataset, ext)

    @staticmethod
    def _to_int64(dataset):
        if not isinstance(dataset, EnImageDataset) and dataset.targets.dtype == np.int32:
            dataset.targets = dataset.targets.astype(np.int64)


class WeightManager:
    def __init__(self, dataset):
        self.dist_vol = None
        self.w_epoch = dataset.args.w_epoch
        self.w_norm = dataset.args.w_norm
        self.cls_n = dataset.cls_n
        model = MODELS[dataset.args.model]
        per_cls_weights = np.ones((len(self.cls_n)))
        if model.weight is None:
            per_cls_weights = np.ones((len(self.cls_n)))
        elif model.weight == 'if':
            per_cls_weights = 1.0 / np.asarray(self.cls_n)
        elif model.weight == 'en':
            beta = dataset.args.beta
            effective_num = 1.0 - np.power(beta, self.cls_n)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
        else:
            self.dist_vol = DistVolEstimator(model, len(self.cls_n), n_pca=dataset.args.n_pca,
                                             min_sample=dataset.args.min_dv_sample)
        per_cls_weights = self._normalize_weight(per_cls_weights)
        print('per_cls_weights:', per_cls_weights)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        self.feature_writer = FeatureFileWriter(dataset, self)

    def add_features(self, type_, y_f, y, acc, epoch, step):
        if self.dist_vol and epoch >= self.w_epoch:
            self.dist_vol.add(type_, y_f.clone().detach().cpu().numpy(), y.numpy())
        self.feature_writer.write_features(type_, y_f, y, acc, epoch, step)

    def end_epoch(self, stage, epoch):
        if self.dist_vol and epoch >= self.w_epoch:
            w = self.dist_vol.update(stage, epoch, self.cls_n)
            if w is not None:
                w, mu, sig = w
                w = self._normalize_weight(w)
                self.feature_writer.write_dist_vol_weight(epoch, w, mu, sig)
                self.per_cls_weights = torch.FloatTensor(w).cuda()

    def _normalize_weight(self, w):
        if self.w_norm == 'N':
            c = len(self.cls_n) / np.sum(w)
        elif self.w_norm == 'E':
            c = sum(self.cls_n) / np.sum(w * self.cls_n)
        else:
            raise NotImplementedError()
        ret = w * c
        if np.isnan(ret).any():
            print('NAN found: ', w, c, ret)
            exit(1)
        return ret


class FeatureFileWriter:
    def __init__(self, dm, weight):
        if dm.args.feature_file:
            self.feature_file_is_csv = dm.args.feature_file.endswith('.csv')
            if self.feature_file_is_csv:
                self.feature_file = open(dm.args.feature_file, 'w')
                self.feature_file.write('type,epoch,step,acc,cls,features\n')
            else:
                self.feature_file = h5py.File(dm.args.feature_file, 'w')
                for field in ['dataset', 'model', 'beta', 'loss', 'batch_size', 'num_classes',
                              'num_meta', 'imb_factor',
                              'test_batch_size', 'epochs']:
                    self.feature_file.attrs[field] = getattr(dm.args, field)
                self.feature_file.attrs['n_train'] = dm.n_train
                self.feature_file.attrs['n_test'] = dm.n_test
                self.feature_file.attrs['n_meta'] = dm.n_meta
                self.feature_file.create_dataset('per_cls_weights',
                                                 data=weight.per_cls_weights.cpu().numpy())
        else:
            self.feature_file = None
            self.feature_file_is_csv = False

    def write_features(self, type_, y_f, y, acc, epoch, step):
        if self.feature_file:
            if self.feature_file_is_csv:
                feature_list = y_f.clone().detach().cpu().numpy()
                for cls, score in zip(y.numpy(), feature_list):
                    self.feature_file.write(
                        f'{type_},{epoch},{step},{acc},{cls},' + ','.join(
                            str(xi) for xi in score) + '\n')
            else:
                g = self.feature_file.create_group(f'{type_}-{epoch}-{step}')
                g.attrs['acc'] = acc
                g.create_dataset('y_f', data=y_f.clone().detach().cpu().numpy())
                g.create_dataset('y', data=y.numpy())

    def write_weights(self, pre_weights, eps, w, epoch, step):
        if self.feature_file:
            if not self.feature_file_is_csv:
                g = self.feature_file[f'{META_FEATURE}-{epoch}-{step}']
                g.create_dataset('pre_weights', data=pre_weights.clone().detach().cpu().numpy())
                g.create_dataset('eps', data=eps.clone().detach().cpu().numpy())
                g.create_dataset('weights', data=w.clone().detach().cpu().numpy())

    def write_dist_vol_weight(self, epoch, w, mu, sig):
        if self.feature_file:
            if not self.feature_file_is_csv:
                g = self.feature_file.create_group(f'dv_{TRAIN_FEATURE}-{epoch}')
                if mu is not None:
                    g.create_dataset('mu', data=mu)
                if sig is not None:
                    g.create_dataset('sigma', data=sig)
                g.create_dataset('w', data=w)

    def close(self):
        if self.feature_file:
            self.feature_file.close()


def main():
    args = parse_args()
    # cm = CommandManager()
    # cm.run()

    # ne = args.epochs
    # ns = 100
    # for e in range(args.epochs):
    #     for s in range(100):
    #         print(f'Epoch {e}/{ne}, Step {s}/{ns}')
    #         if cm.get_commands():

    dm = DatasetManager(args)
    if args.check_data:
        check_data(dm.test_loader, 'test')
        check_data(dm.train_loader, 'train')
        exit(1)
    wm = WeightManager(dm)
    model_info = MODELS[args.model]

    best_prec1 = 0

    # create model
    model = build_model(len(dm.cls_n))
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # print("=> loading checkpoint")
    # checkpoint = torch.load('/home/muhammad/Reweighting_samples/Meta-weight-net_class-imbalance-master/checkpoint/ours/ckpt.best.pth.tar', map_location='cuda:0')
    # args.start_epoch = checkpoint['epoch']
    # best_acc1 = checkpoint['best_acc1']
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer_a.load_state_dict(checkpoint['optimizer'])
    # print("=> loaded checkpoint")

    for epoch in range(args.epochs):
        if args.lr_schedule == 'en':
            adjust_learning_rate_en(optimizer_a, epoch + 1, dm.args.lr)
        else:
            adjust_learning_rate(optimizer_a, epoch + 1, dm.args.lr)

        if model_info.meta:
            fun = train if epoch < args.w_epoch else train_meta
        elif model_info.id != 'na':
            fun = train if epoch < args.w_epoch else train_en
        else:
            fun = train
        fun(dm, wm, model, optimizer_a, epoch)

        # tr_prec1, tr_preds, tr_gt_labels = validate(imbalanced_train_loader, model, criterion, epoch)
        # evaluate on validation set
        prec1, preds, gt_labels = validate(dm, wm, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save_checkpoint(args, {
        #   'epoch': epoch + 1,
        #   'state_dict': model.state_dict(),
        #   'best_acc1': best_prec1,
        #   'optimizer' : optimizer_a.state_dict(),
        # }, is_best)

    print('Best accuracy: ', best_prec1)
    wm.feature_writer.close()


def train(dm, wm, model, optimizer_a, epoch):
    """Train for one epoch on the training set"""
    train_loader = dm.train_loader
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        y_f = model(input_var)
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        l_f = torch.mean(cost_w)  # * w)
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        wm.add_features(TRAIN_FEATURE, y_f, target, prec_train.item(), epoch, i)

        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % dm.args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                  .format(epoch, i, len(train_loader), loss=losses, top1=top1))


def train_meta(dm, wm, model, optimizer_a, epoch):
    """Train for one epoch on the training set"""
    train_loader = dm.train_loader
    validation_loader = dm.meta_loader
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    weight_eps_class = [0 for _ in range(len(dm.cls_n))]
    total_seen_class = [0 for _ in range(len(dm.cls_n))]
    batch_w_eps = []
    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        target_var = target_var.cpu()

        # import pdb; pdb.set_trace()
        y = torch.eye(len(dm.cls_n))

        labels_one_hot = y[target_var].float().cuda()

        weights = torch.tensor(wm.per_cls_weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        # weights = weights.unsqueeze(1)
        # weights = weights.repeat(1, num_classes)

        meta_model = ResNet32(len(dm.cls_n))
        meta_model.load_state_dict(model.state_dict())

        meta_model.cuda()

        # compute output
        # Lines 4 - 5 initial forward pass to compute the initial weighted loss

        y_f_hat = meta_model(input_var)

        wm.add_features(PRE_TRAIN_FEATURE, y_f_hat, target, 0, epoch, i)

        target_var = target_var.cuda()
        cost = F.cross_entropy(y_f_hat, target_var, reduce=False)

        weights = to_var(weights)
        eps = to_var(torch.zeros(cost.size()))

        w_pre = weights + eps
        l_f_meta = torch.sum(cost * w_pre)
        meta_model.zero_grad()

        # Line 6-7 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = dm.args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
        meta_model.update_params(meta_lr, source_params=grads)
        # del grads

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        input_validation, target_validation = next(iter(validation_loader))
        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation, requires_grad=False)

        # import pdb; pdb.set_trace()
        y_g_hat = meta_model(input_validation_var)
        l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
        prec_metada = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]
        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

        dm.add_features(META_FEATURE, y_g_hat, target_validation, prec_metada.item(), epoch, i)

        # import pdb; pdb.set_trace()
        new_eps = eps - 0.01 * grad_eps
        w = weights + new_eps

        dm.feature_writer.write_weights(weights, new_eps, w, epoch, i)

        del grad_eps, grads

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f = model(input_var)
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)

        l_f = torch.mean(cost_w * w)

        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        dm.add_features(TRAIN_FEATURE, y_f, target, prec_train.item(), epoch, i)

        losses.update(l_f.item(), input.size(0))
        meta_losses.update(l_g_meta.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))
        # meta_top1.update(prec_meta.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        # import pdb; pdb.set_trace()

        if i % dm.args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            # 'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                # 'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses, top1=top1))

    wm.end_epoch(TRAIN_FEATURE, epoch)


def train_en(dm, wm, model, optimizer_a, epoch):
    """Train for one epoch on the training set"""
    train_loader = dm.train_loader
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        weights = to_var(wm.per_cls_weights[target_var.cpu()])
        y_f = model(input_var)
        cost = F.cross_entropy(y_f, target_var, reduce=False)
        l_f = torch.mean(cost * weights)
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        wm.add_features(TRAIN_FEATURE, y_f, target, prec_train.item(), epoch, i)

        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % dm.args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                  .format(epoch, i, len(train_loader), loss=losses, top1=top1))

    wm.end_epoch(TRAIN_FEATURE, epoch)


def validate(dm, wm, model, criterion, epoch):
    """Perform validation on the validation set"""
    val_loader = dm.test_loader
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    true_labels = []
    preds = []

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()  # async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = criterion(output, target_var)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target_var.data.cpu().numpy())
        preds += preds_output

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        wm.add_features(TEST_FEATURE, output, target.cpu(), prec1.item(), epoch, i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % dm.args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    # import pdb; pdb.set_trace()

    wm.end_epoch(TEST_FEATURE, epoch)

    return top1.avg, preds, true_labels


def build_model(num_classes):
    model = ResNet32(num_classes)
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR divided by 10 at 160th, and 180th epochs"""
    # lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120))* (0.2 ** int(epoch >= 160)))
    # lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
    lr = lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_en(optimizer, epoch, lr):
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch <= 30:
        lr = lr
    elif epoch <= 60:
        lr = lr * 0.1
    elif epoch <= 80:
        lr = lr * 0.01
    else:
        lr = lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(args, state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % ('checkpoint', 'ours')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def check_data(loader, name):
    n = len(loader)
    print(f'Check data: [{name}] {n} records')
    t_start = time.time()
    for i, (input, target) in enumerate(loader):
        elapsed = time.time() - t_start
        fps = (i + 1) / elapsed
        eta = datetime.timedelta(seconds=int((n - 1) / fps))
        print(f'  {elapsed:03.1f} {i}/{n}, {fps:.1f} fps, ETA {eta}')
    elapsed = time.time() - t_start
    print(f'{elapsed:03.1f} Finished.')


if __name__ == '__main__':
    main()
