import torch

import matplotlib
matplotlib.use('agg')
import numpy as np
from utils.combine_both_eyes import combine_both_eyes
import torch.backends.cudnn as cudnn
from utils.datasets_multi_cls import build_dataset_only_testing
from engine_finetune import evaluate,save_pred_res
import argparse
import datetime
import ast
import os
from copy import deepcopy
import yaml
import utils
from utils.pos_embed import interpolate_pos_embed_vit
from torch.utils.tensorboard import SummaryWriter

import models_vit


import warnings
warnings.filterwarnings("ignore")

def only_testing(args, model_path,phase='test'):


    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.in_domains=='rgb':
        args.data_path = os.path.join(args.data_path,'cfp')
        finetune=opts.weight_cfp
    else:
        args.data_path = os.path.join(args.data_path,'oct')
        finetune = opts.weight_oct
    args.finetune=finetune


    args.in_domains = args.in_domains.split('-')

    cudnn.benchmark = True

    dataset_test = build_dataset_only_testing(is_train=phase, args=args)

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__['vit_large_patch16'](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.use_mean_pooling,
    )

    checkpoint = torch.load(args.finetune, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint['model']

    interpolate_pos_embed_vit(model, checkpoint_model)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    # print(msg)
    assert msg.missing_keys == []


    model.to(device)

    print("Only testing code ...")
    output_dir =os.path.join(args.output_dir,args.finetune.split('/')[-2])
    _, metrics_test_list, prediction = evaluate(data_loader_test, model, device, output_dir, epoch=f"only—{phase}-test", mode=f'only_{phase}',num_class=args.nb_classes, args=args, save_confusion_matrix=False, save_heatmap=True)

    csv_data_path=os.path.join(args.csv_path, f'{phase}.xlsx')
    save_pred_res(prediction, args, csv_data_path,csv_name=os.path.join(model_path, f'{phase}_predict.csv'))

    if args.combine_eyes:
        # combine both eyes
        root_dir = args.output_dir
        combine_both_eyes(root_dir)


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

    parser.add_argument('--weight_cfp', default='',
                        help='resume from checkpoint')
    parser.add_argument('--weight_oct', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--combine_eyes', action='store_true', help='Perform evaluation only')
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
    parser.add_argument('--gene', type=str, default='')
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

if __name__ == '__main__':
    opts = get_args()
    only_testing(deepcopy(opts), opts.output_dir, phase='test')