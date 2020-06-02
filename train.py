""" #!/usr/bin/env python3 -u """
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import itertools
import os
import sys
import math
import random
import subprocess
import time
import numpy as np
import pandas as pd

import torch

from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.utils import import_user_module
from fairseq.models import ema_reverse, ema_restore

from allennlp.models.semantic_role_labeler import convert_bio_tags_to_conll_format

import logging
logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

sys.setrecursionlimit(2000)


def main(args, init_distributed=False):
    import_user_module(args)

    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    logger.info('Setup task')
    task = tasks.setup_task(args)

    # Load dataset splits
    logger.info('Load dataset splits')
    load_dataset_splits(task, ['train', 'valid'])

    # Initialize distributed training (after data loading)
    logger.info('Initialize distributed training')
    if init_distributed:
        import socket
        args.distributed_rank = distributed_utils.distributed_init(args)
        print('| initialized host {} as rank {}'.format(socket.gethostname(), args.distributed_rank))

    # Build model and criterion
    logger.info('Build model and criterion')
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)
    oom_batch = task.dataset('train').get_dummy_batch(1, max_positions)

    model.copy_pretrained_params(args)

    # Build trainer
    logger.info('Build trainer')
    trainer = Trainer(args, task, model, criterion, dummy_batch, oom_batch)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Initialize dataloader
    logger.info('Initialize dataloader')
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    )

    # Load the latest checkpoint if one is available
    if not load_checkpoint(args, trainer, epoch_itr):
        logger.info('Load the latest checkpoint')
        trainer.dummy_train_step([dummy_batch])

    # Train until the learning rate gets too small
    logger.debug('TRAIN')
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    logger.debug("lr ... %lf" % lr)
    logger.debug("min_lr ... %lf" % args.min_lr)
    logger.debug("epoch_itr.epoch ... %d" % epoch_itr.epoch)
    logger.debug("max_epoch ... %d" % max_epoch)
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        
        # train for one epoch
        logger.info('===== train =====')
        train(args, trainer, task, epoch_itr)

        if epoch_itr.epoch % args.validate_interval == 0:
            logger.info('===== validate =====')
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            logger.info("epoch_itr.epoch ... %d" % epoch_itr.epoch)
            logger.debug("valid_losses ... ".format(valid_losses[0]))

            # ema process (直近の比重を大きく?)
            logger.info('ema process')
            if not args.no_ema:
                old_data = ema_restore(trainer.ema, trainer.model)
                logger.info('validate')
                valid_losses_ema = validate(args, trainer, task, epoch_itr, valid_subsets)
                if epoch_itr.epoch % args.save_interval == 0:
                    logger.info('save checkpoint ... {}'.format(epoch_itr))
                    save_checkpoint(args, trainer, epoch_itr, valid_losses_ema[0], suffix='ema')
                ema_reverse(trainer.ema, trainer.model, old_data)

        # only use first validation loss to update the learning rate
        logger.info('update lr')
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            logger.debug('save checkpoint ... {}'.format(epoch_itr))
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
    
    train_meter.stop()
    print('| DONE training in {:.1f} seconds'.format(train_meter.sum))
    logger.info('destdir is ... {}'.format(args.save_dir))
    sys.exit()


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
            if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    logger.info('Initialize data iterator')
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]   # 'valid'
    max_update = args.max_update or math.inf        # inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        """ samples: List[Dict] (sizeof 1)
        - id: tensor([48])
        - nsentences: 1
        - ntokens: 43
        - net_input: {}
            - src_tokens: tensor([1, 39])
            - src_lengths: tensor([39])
            - prev_output_tokens: tensor([1, 43])
        - target: tensor([1, 43])
        - source_label: None
        - target_label: None
        """

        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0:
            valid_losses = validate(args, trainer, task, epoch_itr, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    logger.info('reset trainint meters')
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['copy_alpha'] = trainer.get_meter('copy_alpha')
    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets: # subset == 'valid'
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        #TODO ここで計算する
        fo_gold, fo_pred = open('work/gold.prop', 'a'), open('work/pred.prop', 'a')
        logger.info('open gold prop ... {}'.format(fo_gold))
        logger.info('open pred prop ... {}'.format(fo_pred))

        for sample in progress:
            """
            sample.keys()
            * id            ::: tensor([126, 127, 121, 62, 103, 66, 63, 57])
            * nsentences    ::: 8
            * ntokens       ::: 63
            * net_input     ::: dict_keys(['src_tokens, src_length, prev_output_tokens'])
            * target        ::: torch.Size([8, 18])
            * source_label  ::: None
            * target_label  ::: None
            """
            ### develop ###
            nsents = sample['nsentences']
            gold, pred = trainer.eval_step(sample)
            gold = gold.cpu()
            pred = torch.argmax(pred, -1).cpu()
            gold_tokens = np.array(list(dict_itos(task.tgt_dict, gold))).reshape(nsents, -1).tolist()
            pred_tokens = np.array(list(dict_itos(task.tgt_dict, pred))).reshape(nsents, -1).tolist()
            assert len(gold_tokens) == nsents
            assert len(pred_tokens) == nsents
            wrap_write_prop(fo_gold, gold_tokens, gold_tokens)
            wrap_write_prop(fo_pred, pred_tokens, gold_tokens)


            ### original ###
            log_output = trainer.valid_step(sample) # trainer.valid_step 内で srl score を計算？

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)
        
        fo_gold.close(), fo_pred.close()
        ### srl-eval.pl ###
        path_pl = os.path.abspath('/home/miyawaki_shumpei/soft/srlconll-1.1/bin/srl-eval.pl')
        path_gldp = os.path.abspath(fo_gold.name)
        path_prdp = os.path.abspath(fo_pred.name)
        command = f"perl {path_pl} {path_gldp} {path_prdp}"
        process = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = process.stdout.decode("utf8")
        result = [o.split() for o in result.split('\n') if len(o.split()) > 1]
        statistics, scores = result[:3], result[3:]
        column, *scores = scores
        column.insert(0, 'label')
        overall = ( 
            pd.DataFrame(scores, columns=column)
            .set_index('label')
            .loc['Overall']
        )
        
        logger.debug('overall scores ::: {}' % overall)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats['loss'].avg)
    return valid_losses


# decorator
def develop(tag):
    def _develop(func):
        def wrapper(*args, **kwargs):
            logger.info('{} ... {}'.format(tag, func))
            func(*args, **kwargs)
        return wrapper
    return _develop


def dict_itos(dict, indices):
    return map(lambda x: dict[int(x)], indices)

def is_start(t, excepts=('<unk>', '<pad>', '<EN-SRL>', '<DE-SRL>', '</s>')):
    return t.startswith('<') and t.endswith('>') and not t.startswith('</') and (t not in excepts)

def is_end(t, excepts=('<unk>', '<pad>', '<EN-SRL>', '<DE-SRL>', '</s>')):
    return t.startswith('</') and t.endswith('>') and (t not in excepts)

def extract_index(tokens: list, query='<V>', pos=1):
    index = pos+tokens.index(query) if query in tokens else -1
    return index if (index+1 <= len(tokens)) and (index != -1) else -1

def is_closed(tokens: list):
    isInvalid = 0
    for token in tokens:
        if isInvalid < 0 or 1 < isInvalid: return False
        if is_start(token): isInvalid += 1
        elif is_end(token): isInvalid -= 1
    return True

#@develop('dev.1 convert into bio')
def to_bio(tokens: list, init_label='O', excepts=('<pad>', '<EN-SRL>', '<DE-SRL>', '</s>')) -> list:
    # is_close(tokens)
    is_B, bios, label = False, [], init_label
    for token in tokens:
        if is_start(token):
            is_B, label = True, token[1:-1]
        elif is_end(token): 
            is_B, label = False, init_label
            continue
        elif label == init_label: # O
            bios.append(label)
        elif token in excepts:
            continue
        else: # token==label -> BIを付与
            if is_B:
                bios.append(f"B-{label}")
                is_B = False
            else:
                bios.append(f"I-{label}")
    return bios

#@develop('dev.2 convert into prop')
def to_prop(bios:list):
    #assert bios is not None
    return convert_bio_tags_to_conll_format(bios)

# dev なので，同一文でも新しく 2cols を作成
#@develop('dev.3 write prop')
def write_prop(fo, tokens:list, golds:list, oracle='min'):
    assert oracle in ('min', 'max'), "OracleError"
    gold_props = to_prop(to_bio(golds))
    verb_idx = extract_index(golds, query='<V>', pos=1)
    col_0 = ['-' for _ in gold_props]
    col_1 = []

    if verb_idx == -1:
        # 述語なし
        col_1 = []
    else:
        def extract_index_for_cols(golds, target='<V>'):
            count = 0
            for token in golds:
                if (not is_start(token)) and (not is_end(token)): count+=1
                if token == target: return count
        verb = golds[verb_idx]
        col_vix = extract_index_for_cols(golds)
        try:
            col_0[col_vix] = verb
        except IndexError:
            import ipdb; ipdb.set_trace()
        if len(tokens) != len(golds):
            # len 不一致
            import ipdb; ipdb.set_trace()
            col_1 = ['*' if oracle=='min' else gold for gold in gold_props]
        elif is_closed(tokens):
            # unclosed (不適切 bracket)
            col_1 = ['*' if oracle=='min' else gold for gold in gold_props]
        else:
            # 問題なし
            col_1 = to_prop(to_bio(tokens))
            if len(col_0) != len(col_1):
                # len 不一致
                col_1 = ['*' if oracle=='min' else gold for gold in gold_props]
    
    try:
        assert (len(col_0)==len(col_1)) or (set(col_0) == {'-'})
    except:
        import ipdb; ipdb.set_trace()
    for c0, c1 in itertools.zip_longest(col_0, col_1):
        print(f"{c0}\t{c1}", file=fo, end='\n')
    print("\n", file=fo)


def wrap_write_prop(fo, sents:list, golds:list):
    # write_prop 関数は一文専用なので，複数文で loop
    for sent, gold in zip(sents, golds):
        sent = [token for token in sent if token!='<pad>']
        gold = [token for token in gold if token!='<pad>']
        write_prop(fo, sent, gold, oracle='min')


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best_loss'] = min(save_checkpoint.best, stats['loss'].avg)
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss, suffix=''):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}{}.pt'.format(suffix, epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint{}_{}_{}.pt'.format(suffix, epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and len(suffix) == 0 and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint{}_last.pt'.format(suffix)] = True  # keep this last so that it's a symlink

    if len(suffix) == 0: # update best only when suffix is empty
        prev_best = getattr(save_checkpoint, 'best', val_loss)
        if val_loss is not None:
            save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint' +  suffix + '\d+\.pt')
        for old_chk in checkpoints[args.keep_last_epochs:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    if os.path.isabs(args.restore_file):
        checkpoint_path = args.restore_file
    else:
        checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path, args.reset_optimizer, args.reset_lr_scheduler,
                                              eval(args.optimizer_overrides))
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']
        return True
    else:
        print('| no existing checkpoint found {}'.format(checkpoint_path))
    return False


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e


def distributed_main(i, args):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    try:
        git_branch = subprocess.check_output(['git', 'symbolic-ref', '--short', 'HEAD'])
        git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    except:
        git_branch = 'unknown'
        git_revision = 'unknown'
    print('GIT: {} {}'.format(git_branch, git_revision))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('-' * 80)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
