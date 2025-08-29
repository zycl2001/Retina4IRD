import math
import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from typing import Iterable, Optional
import utils
import utils.misc as misc
import utils.lr_sched as lr_sched
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score, roc_curve
import matplotlib
matplotlib.use('agg')
import numpy as np

import torch.backends.cudnn as cudnn

from utils.datasets_multi_cls import build_dataset_only_testing

from utils.pos_embed import interpolate_pos_embed_vit
import models_vit
from utils.focalLoss import FocalLoss


def classification_metrics(y_true, y_pred, y_score, nb_classes, use_youden_index=False):

    if nb_classes == 2 and use_youden_index:
        fpr, tpr, thresholds = roc_curve(y_true, y_score[:, 1])
        best_threshold_index = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_threshold_index]
        y_pred = (y_score[:, 1] >= best_threshold).astype(int)

    if nb_classes == 2:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        sen = recall_score(y_true, y_pred)
        spe = recall_score(y_true, y_pred, pos_label=0)
        pre = precision_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, weights='linear')
        auc_roc = roc_auc_score(y_true, y_score[:, 1])
        auc_pr = average_precision_score(y_true, y_score[:, 1])
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        sen = recall_score(y_true, y_pred, average='macro')
        spe = np.mean(specificity(y_true, y_pred))
        pre = precision_score(y_true, y_pred, average='macro')
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        auc_roc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        auc_pr = average_precision_score(y_true, y_score, average='macro')
    return dict(auc_roc=auc_roc, auc_pr=auc_pr, acc=acc, f1=f1, sen=sen, spe=spe, pre=pre, kappa=kappa, y_pred=y_pred)


def specificity(y_true: np.array, y_pred: np.array, classes: set = None):
    if classes is None:
        classes = set(np.concatenate((np.unique(y_true), np.unique(y_pred))))
    specs = []
    for cls in classes:
        y_true_cls = np.array((y_true == cls), np.int32)
        y_pred_cls = np.array((y_pred == cls), np.int32)
        specs.append(recall_score(y_true_cls, y_pred_cls, pos_label=0))
    return specs

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None and args.log_tensorboard:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in samples.items()
        }

        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in args.in_domains
        }
        with torch.cuda.amp.autocast():
            outputs = model(input_dict)
            outputs = outputs['cls']
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and args.log_tensorboard:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

        if log_writer is not None and args.log_wandb:
            log_writer.update(
                {
                    'loss': loss_value_reduce,
                    'lr': max_lr,
                }
            )
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, task, epoch, mode, num_class, args, save_confusion_matrix=False, save_heatmap=False):

    criterion=FocalLoss(gamma=2,weight=None)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []

    model.eval()
    image_name_list = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        image_name = batch[2]
        image_name_list.append(image_name)

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in images.items()
        }

        target = target.to(device, non_blocking=True)
        true_label = F.one_hot(target.to(torch.int64), num_classes=num_class)

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
        }

        with torch.cuda.amp.autocast():
            output = model(input_dict)
            output = output['cls']
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            _, prediction_decode = torch.max(prediction_softmax, 1)
            _, true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())


        metric_logger.update(loss=loss.item())


    true_label_decode_list_return = true_label_decode_list[:]
    prediction_list_return = prediction_list[:]

    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)

    res = classification_metrics(true_label_decode_list, prediction_decode_list, np.array(prediction_list), args.nb_classes,use_youden_index=True)
    acc, F1, precision, kappa = res['acc'], res['f1'], res['pre'], res['kappa']
    sensitivity, specificity = res['sen'], res['spe']
    auc_roc, auc_pr = res['auc_roc'], res['auc_pr']
    prediction_decode_list = res['y_pred']

    metric_logger.synchronize_between_processes()

    print(f'Epoch {epoch} {mode} Sklearn Metrics - AUC-roc: {auc_roc:.4f} AUC-pr: {auc_pr:.4f} Acc: {acc:.4f} SEN: {sensitivity:.4f} SPE: {specificity:.4f} PRE: {precision:.4f} F1-score: {F1:.4f} Kappa: {kappa:.4f}')

    metric_dict = {
        'epoch': epoch,
        'AUROC': auc_roc,
        'AUPR': auc_pr,
        'Accuracy': acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1': F1,
        'Kappa': kappa
        }

    predict = {
        'label': true_label_decode_list_return,
        'prob': prediction_list_return,
        'image_name': image_name_list,
        'predict': list(prediction_decode_list)
        }

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_dict, predict


def save_pred_res(predict_dict, args,csv_data_path, csv_name):
    image_name_list = predict_dict['image_name']
    metrics_test_list = predict_dict['prob']
    prediction = predict_dict['predict']
    metrics_test_df = pd.DataFrame()

    name_dict = dict()
    for domain in args.in_domains:
        name_dict[domain] = []

    for dic in image_name_list:
        for domain in args.in_domains:
            for item in dic[domain]:
                name_dict[domain].append(item)

    for domain in args.in_domains:
        metrics_test_df[domain] = name_dict[domain]

    metrics = pd.DataFrame(metrics_test_list)
    metrics_test_df = pd.concat([metrics_test_df, metrics], axis=1)

    if 'rgb' in args.in_domains:
        metrics_test_df = metrics_test_df.rename(columns={'rgb': 'cfp'})
    metrics_test_df['predict'] = prediction

    csv_data=pd.read_excel(csv_data_path)
    image_colums=''

    if args.in_domains[0]=='rgb':
       image_colums='cfp'
    elif args.in_domains[0]=='oct':
       image_colums='oct'
    df_merge = pd.merge(csv_data, metrics_test_df, on=image_colums, how="inner")

    df_merge.to_csv(csv_name, index=False, encoding="utf-8-sig")


def only_testing(args, model_path,phase='test'):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


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
    print(msg)
    assert msg.missing_keys == []


    model.to(device)

    print("Only testing code ...")
    output_dir =os.path.join(args.output_dir,args.finetune.split('/')[-2])
    _, metrics_test_list, prediction = evaluate(data_loader_test, model, device, output_dir, epoch=f"only—{phase}-test", mode=f'only_{phase}',num_class=args.nb_classes, args=args, save_confusion_matrix=False, save_heatmap=True)

    csv_data_path=os.path.join(args.csv_path, f'{phase}.xlsx')
    save_pred_res(prediction, args, csv_data_path,csv_name=os.path.join(model_path, f'{phase}_predict.csv'))


if __name__ == '__main__':
    pred = np.array([[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.]])
