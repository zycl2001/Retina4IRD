import argparse
import datetime
import json
import ast
import os

import time

from pathlib import Path
from copy import deepcopy

import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
import utils

from utils import NativeScalerWithGradNormCount as NativeScaler

from utils.pos_embed import interpolate_pos_embed_vit
from utils.datasets_multi_cls import build_dataset
from utils import Mixup,SoftTargetCrossEntropy
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_

import utils.lr_decay as lrd

from engine_finetune import train_one_epoch, evaluate, only_testing
import utils.misc as misc
import models_vit
from utils.focalLoss import FocalLoss

import warnings
warnings.filterwarnings("ignore")


def get_args():
    config_parser = parser = argparse.ArgumentParser(description='MultiMAE Finetune script', add_help=False)
    parser.add_argument('-c', '--config', default='cfgs/image_cls.yaml', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('MultiMAE fine-tuning script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--model', default='multivit_base', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--transform_type', default='', type=str)
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m5-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1), new: rand-m5-mstd0.5-inc1'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    parser.add_argument('--scale', type=str, default='None', help='New: (0.6, 1.0), old: None')
    parser.add_argument('--vflip', type=float, default=0., help='New: 0.5, old: 0.')
    parser.add_argument('--hflip', type=float, default=0.5, help='New: 0.5, old: 0.5')

    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')


    parser.add_argument('--data_path', default='', type=str, help='dataset path')
    parser.add_argument('--csv_path', default='', type=str)
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='output/finetune',
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--seed_count', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--in_domains', default='rgb', type=str,
                        help='Input domain names, separated by hyphen')

    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--eyenet', type=str, default='')
    parser.add_argument('--mutant_gene', type=str, default='')
    parser.add_argument('--label_column', default='', type=str)

    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--nowtime', default=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), type=str)

    parser.add_argument('--test', action='store_true', help='Perform testing only')
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='log training and validation metrics to wandb')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    if args.scale == 'None':
        args.scale = None
    else:
        try:
            args.scale = ast.literal_eval(args.scale)
        except (ValueError, SyntaxError):
            raise ValueError("Invalid format for --scale. Must be a tuple or 'None'.")

    return args

def main(args):
    utils.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    args.in_domains = args.in_domains.split('-')

    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)
    if args.having_test_set:
        dataset_test = build_dataset(is_train='test', args=args)
    else:
        dataset_test = None

    if True:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True,
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)

        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if args.having_test_set:
            if args.dist_eval:
                if len(dataset_test) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                          'This will slightly alter validation results as extra duplicate entries are added to achieve '
                          'equal num of samples per-process.')
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            else:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_tensorboard:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    elif global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    if args.having_test_set:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models_vit.__dict__['vit_large_patch16'](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.use_mean_pooling,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed_vit(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.use_mean_pooling:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    eff_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()

    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion= FocalLoss(gamma=2, weight=None)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.load_model(
        args=args,  model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        print("Only validation...")
        _, __, ___ = evaluate(data_loader_val, model, device, args.output_dir, epoch=-1, mode='val',num_class=args.nb_classes, args=args, save_confusion_matrix=True)
        exit(0)

    if args.test:
        print("Only testing...")
        _, __, ___ = evaluate(data_loader_test, model, device, args.output_dir, epoch=-1, mode='test',num_class=args.nb_classes, args=args, save_confusion_matrix=True)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_auc = 0.0
    best_val_epoch = 0

    metrics_val_list = []
    metrics_test_list = []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if log_writer is not None and args.log_wandb:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        val_stats, val_metric, _ = evaluate(data_loader_val, model, device, args.output_dir, epoch, mode='val',
                                          num_class=args.nb_classes, args=args)
        if max_auc < val_metric['AUROC']:
            max_auc = val_metric['AUROC']
            best_val_epoch = epoch
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                print(f"\nThe current best model is epoch {epoch}.\n")

        if args.traintest:
            test_stats, test_metric, _ = evaluate(data_loader_test, model, device, args.output_dir, epoch, mode='test',
                                           num_class=args.nb_classes, args=args)

        if log_writer is not None and args.log_tensorboard:
            log_writer.add_scalar('perf/val_auc', val_metric['AUROC'], epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters}

        if log_writer is not None and args.log_tensorboard:
            log_writer.update(log_stats)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None and args.log_tensorboard:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if log_writer is not None and args.log_wandb:
            log_metric = {'epoch': epoch, 'n_parameters': n_parameters}
            log_metric.update(**{f'val_{k}': v for k, v in val_metric.items()})
            if args.traintest:
                log_metric.update(**{f'test_{k}': v for k, v in test_metric.items()})
            log_writer.update(log_metric)

        metrics_val_list.append(val_metric)
        if args.traintest:
            metrics_test_list.append(test_metric)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    test_res_on_best_val = None
    if args.having_test_set:
        print('\nLoading model with best epoch in validation------------------------------------------------------\n')
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint-best.pth'), map_location='cpu')
        if args.distributed:
            msg = model.module.load_state_dict(checkpoint['model'], strict=False)
        else:
            msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        assert msg.missing_keys == []
        _, test_res_on_best_val, predict = evaluate(data_loader_test, model, device, args.output_dir, best_val_epoch, mode='test',
                                                   num_class=args.nb_classes, args=args, save_confusion_matrix=True)

    metrics_val_df = pd.DataFrame(metrics_val_list).sort_values(by='AUROC', ascending=False).round(4)

    if log_writer is not None and args.log_wandb:
        log_writer.wandb_finish()

    if test_res_on_best_val is not None:
        return test_res_on_best_val
    else:
        print('\nLoading model with best epoch in validation------------------------------------------------------\n')
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint-best.pth'), map_location='cpu')
        if args.distributed:
            msg = model.module.load_state_dict(checkpoint['model'], strict=False)
        else:
            msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        assert msg.missing_keys == []
        _, __, predict = evaluate(data_loader_val, model, device, args.output_dir,best_val_epoch, mode='val', num_class=args.nb_classes, args=args, save_confusion_matrix=True)
        return metrics_val_df.iloc[0].to_dict()


def run_one_time(opts,seed=0):

    opts.seed = seed
    dataset = str(Path(opts.csv_path).name)

    opts.wandb_project = f"{opts.wandb_project}-{dataset}"

    opts.output_dir = os.path.join(opts.output_dir, dataset)
    if opts.finetune == 'no_load_weights':
        opts.finetune = False
    else:
        opts.finetune = opts.__dict__[opts.finetune]

    opts.output_dir = os.path.join(opts.output_dir, f"seed_{opts.seed}")

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    res = main(opts)

    return res, str(Path(opts.output_dir).parent)

def set_mutant_gene_label(args):
    gene_mapping = {
        "ush2a": 1, "cyp4v2": 2, "abca4": 3, "rpgr": 4, "rpe65": 5,
        "rs1": 6, "prpf31": 7, "rho": 8, "rdh12": 9, "mertk": 10,
        "chm": 11, "cnga1": 12, "pde6b": 13, "pde6a": 14, "gucy2d": 15
    }
    file_list=['train.xlsx','val.xlsx','test.xlsx']
    for file in file_list:

        file_path = os.path.join(args.csv_path, file)
        df = pd.read_excel(file_path)
        gene_col_lower = df[args.mutant_gene].str.lower()

        is_nan = df[args.mutant_gene].isna()

        mapped = gene_col_lower.map(gene_mapping)

        if args.label_column not in df.columns:
            df = df.drop(columns=[args.label_column])

        df[args.label_column] = mapped.where(~is_nan, 0).fillna(16).astype(int)

        df.to_excel(file_path, index=False)


if __name__ == '__main__':

    if not hasattr(torch, "_six"):
        class _Six:
            string_classes = (str,)
            int_classes = (int,)
            inf = float('inf')
        torch._six = _Six()

    opts = get_args()

    if opts.in_domains=='rgb':
        opts.data_path = os.path.join(opts.data_path,'cfp')
        opts.csv_path = os.path.join(opts.csv_path, 'cfp')
    else:
        opts.data_path = os.path.join(opts.data_path,'oct')
        opts.csv_path = os.path.join(opts.csv_path, 'oct')

    set_mutant_gene_label(opts)
    seed_count=opts.seed_count
    path = ''

    if not opts.test and not opts.eval:
        for i in range(seed_count):
            res, path = run_one_time(opts,seed=i)
            opts.finetune=os.path.join(path, f'seed_{i}', 'checkpoint-best.pth')
            model_path=os.path.join(path, f'seed_{i}')
            only_testing(deepcopy(opts), model_path, phase='train')
            only_testing(deepcopy(opts),model_path, phase='val')
            only_testing(deepcopy(opts),model_path,phase='test')
        exit(0)
    elif opts.eval:
        only_testing(deepcopy(opts), opts.output_dir, phase='train')
        only_testing(deepcopy(opts), opts.output_dir,phase='val')
        only_testing(deepcopy(opts), opts.output_dir,phase='test')
    elif opts.test:
        only_testing(deepcopy(opts), opts.output_dir,phase='test')
