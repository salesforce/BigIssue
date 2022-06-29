import os

from collections import OrderedDict

os.environ['OMP_NUM_THREADS'] = '1'

########################################################################################################
## imports

import os
import argparse
import random
import json
import math

from time import time

import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel

from resumable_loader import create_resumable_partition_iter, create_resumable_partition_iter_split

import wandb

from einops import reduce


########################################################################################################
## args

def create_args(args=argparse.Namespace()):
    args.rng_seed = 42

    args.debug = False

    args.gpu = 0
    args.gpu_enabled = True
    args.gpu_deterministic = False
    args.gpu_mixed_precision = False

    if args.debug:
        args.gpu_distributed_enabled = False
        args.gpu_distributed_port = 8888
        args.gpu_distributed_world_size = 1
        args.test_step = 10
    else:
        args.gpu_distributed_enabled = True
        args.gpu_distributed_port = 8888 + 3
        args.gpu_distributed_world_size = 16
        args.test_step = 2000

    args.data_path = "" # Link your pre-processed training dataset here
    args.data_path_val = "" # Link your pre-processed validation dataset here
    args.data_lines_len = 128 * 4
    args.num_lines_pad = 0
    args.num_lines_with_pad = 128 * 4
    args.data_seq_len = 2048 * 4
    args.data_pad = 1

    args.segment_len = args.data_seq_len // args.num_lines_with_pad
    args.data_val_batches = 64

    args.model_base_dir = 'microsoft/codebert-base'

    args.opt_lr = 5e-5
    args.opt_steps_warmup = 10_00
    args.opt_steps_train = 200_000
    args.opt_grad_clip_enabled = False
    args.opt_grad_clip = 1.0
    args.opt_weight_decay = 0.1
    args.opt_batch_size = 2
    args.opt_grad_acc = 1

    args.chkpoint_load = None
    args.out_prefix = '/tmp/pooling_noisy_java'
    args.out_ckpt_step = 10_000

    args.stats_step_log = 10

    args.wandb_enabled = False
    args.wandb_project = ''
    args.wandb_user = ''

    return args


########################################################################################################
## model

from torch import nn

from transformers import RobertaModel
from transformers import AutoConfig
from transformers.models.roberta.modeling_roberta import RobertaLayer

class ModelWrapper(nn.Module):

    def __init__(self,
                 base_model,
                 config,
                 max_seq_length=512):

        super().__init__()
        self.base_model = base_model

        self._encoding_size = self.base_model.config.hidden_size

        positional_embeddings = torch.zeros(max_seq_length, self._encoding_size)

        for position in range(max_seq_length):
            for i in range(0, self._encoding_size, 2):
                positional_embeddings[position, i] = (
                    math.sin(position / (10000 ** ((2 * i) / self._encoding_size)))
                )
                positional_embeddings[position, i + 1] = (
                    math.cos(position / (10000 ** ((2 * (i + 1)) / self._encoding_size)))
                )

        positional_embeddings = positional_embeddings.unsqueeze(0)

        self.positional_embeddings = positional_embeddings

        self.classification_layer = Classification(config)

    def forward(self, input_ids, attention_mask, chunk_len=512):
        x = input_ids
        xs = torch.split(x, chunk_len, dim=1)
        attention_masks = torch.split(attention_mask, chunk_len, dim=1)
        xs2 = []
        for y, attention_mask in zip(xs, attention_masks):
            output = self.base_model(y, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
            xs2.append(output)

        x = torch.cat(xs2, dim=1)

        positional_embeddings = torch.tile(self.positional_embeddings, (x.shape[0], 1, 1)).to(x.device)
        x = x + positional_embeddings

        x = self.classification_layer(x)

        return x


class Classification(nn.Module):
    def __init__(self, config, classifier_dropout=None, hidden_dropout_prob=0.1):
        super().__init__()
        self.attn = RobertaLayer(config)
        self.segment_len = config.segment_len
        hidden_size = config.hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        classifier_dropout = (
            classifier_dropout if classifier_dropout is not None else hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.token_proj = nn.Linear(hidden_size, 1)

    def forward(self, features):
        x = reduce(features, 'b (l l2) d -> b l d', 'mean', l2=self.segment_len)
        x = self.attn(x)[0]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.token_proj(x)
        x = torch.squeeze(x, 2)
        return x


########################################################################################################
## train

def train_distributed(rank, args):
    if args.gpu_distributed_enabled:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=f'env://',
            world_size=args.gpu_distributed_world_size,
            rank=rank)
    try:
        train(rank, args)
    finally:
        torch.distributed.destroy_process_group()


def train(rank, args):
    #######################
    ## gpu

    if args.gpu_enabled:
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')

    is_master = True if not args.gpu_distributed_enabled else args.gpu_distributed_enabled and rank == 0

    #######################
    ## preamble

    set_seed(args.rng_seed)
    if args.gpu_enabled:
        set_cuda(deterministic=args.gpu_deterministic)

    output_dir = f'{args.output_dir}/{rank}'
    os.makedirs(output_dir, exist_ok=False)

    if is_master and args.wandb_enabled:
        print('creating wandb')

        run = wandb.init(project=args.wandb_project, entity=args.wandb_user, name=args.exp_id, config=args)
        print(f'wandb_run_id={run.id}')
        wandb.save(__file__)

    #######################
    ## data

    process_id = rank if args.gpu_distributed_enabled else 0
    process_n = args.gpu_distributed_world_size if args.gpu_distributed_enabled else 1
    data_train_iter, data_train_files_partition, data_test_iter, test_size = create_resumable_partition_iter_split(
        path=args.data_path, batch_size=args.opt_batch_size, process_id=process_id, process_n=process_n,
        seq_len=args.data_seq_len, resume_at_file=None, resume_at_batch=None)  # change these if resuming training

    #######################
    ## model

    print('creating model')

    config = AutoConfig.from_pretrained(args.model_base_dir)
    config.segment_len = args.segment_len
    base_model = RobertaModel.from_pretrained(args.model_base_dir, add_pooling_layer=False)
    model = ModelWrapper(base_model=base_model, config=config, max_seq_length=args.data_seq_len)
    model = model.to(device)

    if args.chkpoint_load:
        state_dict = torch.load(os.path.join(args.chkpoint_load, "model"), map_location=device)
        new_state_dict = OrderedDict(
            {k[k.index('.') + 1:]: v for k, v in state_dict.items()})  # need to massage the keys
        model.load_state_dict(new_state_dict, strict=False)

    #######################
    ## optimizer

    print('creating optimizer')

    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step):
            learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
            learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
            return learning_rate

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def get_params_without_weight_decay_ln(named_params, weight_decay):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        return optimizer_grouped_parameters

    optimizer = torch.optim.AdamW(
        get_params_without_weight_decay_ln(model.named_parameters(), weight_decay=args.opt_weight_decay),
        lr=args.opt_lr, betas=(0.9, 0.999), eps=1e-08)
    model = model if not args.gpu_distributed_enabled else DistributedDataParallel(model, device_ids=[rank],
                                                                                   output_device=rank,
                                                                                   find_unused_parameters=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.opt_steps_warmup,
                                                num_training_steps=args.opt_steps_train)
    scaler = torch.cuda.amp.GradScaler(enabled=args.gpu_mixed_precision)

    if args.chkpoint_load:
        optimizer.load_state_dict(torch.load(os.path.join(args.chkpoint_load, "optimizer"), map_location=device))
        scheduler.load_state_dict(torch.load(os.path.join(args.chkpoint_load, "scheduler"), map_location=device))
        scaler.load_state_dict(torch.load(os.path.join(args.chkpoint_load, "scaler"), map_location=device))

    ##############
    ## helper function

    def compute_loss_and_logits(data_iter, eval=False):
        # deserialize data

        with torch.no_grad():
            (record, resume_at_file, resume_at_batch) = next(data_iter)
            record = torch.tensor(record.astype(np.int32)).long().to(device)

            n_seq, n_lines = args.data_seq_len, args.data_lines_len

            assert record.shape[1:] == torch.Size([n_seq + n_lines]), f'{record.shape[1:]} == {[n_seq + n_lines]}'

            input_ids = record[:, :n_seq]
            labels = record[:, n_seq:]

            assert input_ids.shape == torch.Size([args.opt_batch_size, n_seq])
            assert labels.shape == torch.Size([args.opt_batch_size, n_lines])

            """
            Padded lines are marked as -100 in the data, but we treat them as non-buggy
            """
            labels[labels == -100] = 0

            if args.num_lines_pad > 0:
                bs = labels.shape[0]
                label_pad = torch.zeros(bs, args.num_lines_pad, dtype=labels.dtype, device=labels.device)
                labels = torch.cat((labels, label_pad), dim=-1)

        with torch.cuda.amp.autocast(enabled=args.gpu_mixed_precision):
            attention_mask = (input_ids != args.data_pad)

            if args.debug or not eval:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits = model.module(input_ids=input_ids, attention_mask=attention_mask)

            weight = torch.tensor([50.], device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(input=logits, target=labels.float(),
                                                                        pos_weight=weight)
            loss = loss / float(args.opt_grad_acc)

        return loss, labels, logits, resume_at_file, resume_at_batch

    #######################
    ## train

    print('starting training')

    t, steps_s, eta_m = time(), 0., 0

    for step in range(args.opt_steps_train + 1):
        if not args.debug:
            torch.distributed.barrier()

        loss_total = 0.0
        tp, fp, fn = 0, 0, 0
        train_acc, train_acc_0, train_acc_1 = 0.0, 0.0, 0.0
        train_n_0, train_n_1 = 0, 0
        train_n_0_p, train_n_1_p = 0, 0

        for batch in range(args.opt_grad_acc):
            loss, labels, logits, resume_at_file, resume_at_batch = compute_loss_and_logits(data_train_iter)

            scaler.scale(loss).backward()

            loss_total += loss.detach()

            with torch.no_grad():
                labels_pred = torch.round(torch.sigmoid(logits))

                tp += torch.sum(torch.mul(labels, labels_pred))
                fp += torch.sum(torch.mul(1 - labels, labels_pred))
                fn += torch.sum(torch.mul(labels, 1 - labels_pred))

                train_acc += torch.mean((labels_pred == labels).float()) / args.opt_grad_acc
                train_acc_0 += torch.mean((labels_pred[labels == 0] == 0).float()) / args.opt_grad_acc
                train_acc_1 += torch.mean((labels_pred[labels == 1] == 1).float()) / args.opt_grad_acc

                train_n_0 += (labels == 0).sum()
                train_n_1 += (labels == 1).sum()

                train_n_0_p += (labels_pred == 0).sum()
                train_n_1_p += (labels_pred == 1).sum()

        # step

        if args.opt_grad_clip_enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.opt_grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        # checkpoint

        if step > 0 and step % args.out_ckpt_step == 0:

            output_dir_step = os.path.join(output_dir, str(step))
            os.makedirs(output_dir_step, exist_ok=False)

            if is_master:
                print(f'saving state')

                torch.save(optimizer.state_dict(), os.path.join(output_dir_step, 'optimizer'))
                torch.cuda.empty_cache()

                torch.save(model.state_dict(), os.path.join(output_dir_step, 'model'))
                torch.cuda.empty_cache()

                torch.save(scheduler.state_dict(), os.path.join(output_dir_step, 'scheduler'))
                torch.cuda.empty_cache()

                torch.save(scaler.state_dict(), os.path.join(output_dir_step, 'scaler'))
                torch.cuda.empty_cache()

            with open(os.path.join(output_dir_step, 'data.json'), 'w') as f:
                json.dump({'data_train_files_partition': data_train_files_partition, 'resume_at_file': resume_at_file,
                           'resume_at_batch': resume_at_batch}, f)

        # stats

        if not args.debug:
            torch.distributed.barrier()
        if is_master:

            train_stats_pres = tp / (tp + fp)
            train_stats_recall = tp / (tp + fn)
            train_stats_f1 = 2 * tp / (2 * tp + fp + fn)

            metrics = {
                'step': (step, '{:8d}'),
                'loss': (loss_total.item(), '{:8.5f}'),
                'lr': (scheduler.get_last_lr()[0], '{:8.7f}'),
                'steps': (steps_s, '{:4.1f}/s'),
                'eta': (eta_m, '{:4d}m'),
                'pres': (train_stats_pres, '{:6.3f}'),
                'recall': (train_stats_recall, '{:6.3f}'),
                'f1': (train_stats_f1, '{:6.3f}'),
                'acc': (train_acc, '{:6.3f}'),
                'acc_0': (train_acc_0, '{:6.4f}'),
                'acc_1': (train_acc_1, '{:6.4f}'),
                'n_0': (train_n_0, '{:4d}'),
                'n_1': (train_n_1, '{:4d}'),
                'n_0_p': (train_n_0_p, '{:4d}'),
                'n_1_p': (train_n_1_p, '{:4d}'),
                'tp': (tp, '{:4f}'),
                'fp': (fp, '{:4f}'),
                'fn': (fn, '{:4f}'),
            }

            if args.wandb_enabled:
                wandb.log({k: v[0] for k, v in metrics.items()}, step)

            if step % args.stats_step_log == 0:
                sep = ' ' * 2
                print(sep.join([f'{k}: {v[1].format(v[0])}' for (k, v) in metrics.items()]))

            if step > 0 and step % 100 == 0:
                t2 = time()
                steps_s = 100. / (t2 - t)
                eta_m = int(((args.opt_steps_train - step) / steps_s) // 60)
                t = t2

            if step > 0 and step % args.test_step == 0 and is_master:
                model.eval()
                with torch.no_grad():

                    data_train_iter_val = \
                        create_resumable_partition_iter(path=args.data_path_val, batch_size=args.opt_batch_size,
                                                        process_id=0, process_n=1, seq_len=args.data_seq_len)[0]

                    loss_total = 0.0
                    tp, fp, fn = 0, 0, 0

                    for i in range(args.data_val_batches):
                        loss, labels, logits, _, _ = compute_loss_and_logits(data_train_iter_val, eval=True)

                        loss_total += loss.detach()

                        labels_pred = torch.round(torch.sigmoid(logits))

                        tp += torch.sum(torch.mul(labels, labels_pred))
                        fp += torch.sum(torch.mul(1 - labels, labels_pred))
                        fn += torch.sum(torch.mul(labels, 1 - labels_pred))

                    metrics["val_loss"] = (loss_total, '{:6.3f}')
                    metrics["val_pres"] = (tp / (tp + fp), '{:6.3f}')
                    metrics["val_recall"] = (tp / (tp + fn), '{:6.3f}')
                    metrics["val_f1"] = ((2 * tp) / (2 * tp + fp + fn), '{:6.3f}')

                    print("-" * 20)
                    print("Eval Results")
                    print(
                        f"Total loss {loss_total}, precision {(tp) / (tp + fp)}, recall {tp / (tp + fn)}, f1 {(2 * tp) / (2 * tp + fp + fn)}")
                    print("-" * 20)

                model.train()

                if args.wandb_enabled:
                    wandb.log({k: v[0] for k, v in metrics.items()}, step)


########################################################################################################
## preamble

def set_gpus(gpu):
    torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    import datetime
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    return output_dir


def copy_source(file, output_dir):
    import shutil
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


########################################################################################################
## main

def main():
    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)

    # args
    args = create_args()
    args.output_dir = os.path.join(args.out_prefix, output_dir)
    args.exp_id = exp_id

    # files
    os.makedirs(args.output_dir, exist_ok=True)
    copy_source(__file__, args.output_dir)

    # distributed
    if args.gpu_distributed_enabled:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.gpu_distributed_port)
        torch.multiprocessing.spawn(train_distributed, nprocs=args.gpu_distributed_world_size, args=(args,), join=True)
    else:
        train(rank=args.gpu, args=args)


if __name__ == '__main__':
    main()
