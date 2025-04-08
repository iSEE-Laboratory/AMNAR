##############################################################################################
# The code is modified from ActionFormer: https://github.com/happyharrycn/actionformer_release
##############################################################################################

# python imports
import argparse
import os
import time
import datetime
from pprint import pformat, pprint
from copy import deepcopy
import pickle
import logging

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our implementation
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    
    config_name = os.path.basename(args.config).replace('.yaml', '')
    log_subfolder = f'./logs/{config_name}'
    os.makedirs(log_subfolder, exist_ok=True)
    log_file = f'{log_subfolder}/train.log'  # Specify log file path
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    # Add console output to log file
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(pformat(cfg))

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])



    """2. create dataset / dataloader"""
    assert args.dataset_buffer is None or args.dataset_buffer == 'save' or args.dataset_buffer == 'load', "Invalid dataset buffer option"
    taskname = cfg['dataset_name'] + '_' + cfg['dataset']['task']
    if args.dataset_buffer == 'save' or args.dataset_buffer is None:
        cfg_dataset = deepcopy(cfg['dataset'])
        cfg_dataset['ckpt_folder'] = ckpt_folder
        train_dataset = make_dataset(
            cfg['dataset_name'], True, cfg['train_split'], **cfg_dataset
        )

        if args.dataset_buffer == 'save':
            os.makedirs('dataset_buffer', exist_ok=True)
            with open(os.path.join(f'dataset_buffer/{taskname}_TRAIN_train_dataset.pkl'), "wb") as f:
                pickle.dump(train_dataset, f)
    else:
        with open(os.path.join(f'dataset_buffer/{taskname}_TRAIN_train_dataset.pkl'), "rb") as f:
            train_dataset = pickle.load(f)

    # data loaders
    train_dataset_full_seq = deepcopy(train_dataset)
    train_dataset_full_seq.is_training = False
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])
    cfg_full_seq = deepcopy(cfg['loader'])
    cfg_full_seq['batch_size'] = 1
    train_loader_full_seq = make_data_loader(
        train_dataset_full_seq, False, rng_generator, **cfg_full_seq)


    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])


    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    logging.info("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'], strict=False)

            # load additional class member states
            if 'action_clusters' in checkpoint and checkpoint['action_clusters'] is not None:
                action_clusters = checkpoint['action_clusters']
                if 'use_norm' in cfg['model']['rebuild_model_cfg']:
                    if hasattr(action_clusters, 'use_norm'):
                        if action_clusters.use_norm != cfg['model']['rebuild_model_cfg']['use_norm']:
                            action_clusters.use_norm = cfg['model']['rebuild_model_cfg']['use_norm']
                    else:
                        action_clusters.use_norm = cfg['model']['rebuild_model_cfg']['use_norm']
                if 'distance_type' in cfg['model']['rebuild_model_cfg']:
                    if hasattr(action_clusters, 'distance_type'):
                        if action_clusters.distance_type != cfg['model']['rebuild_model_cfg']['distance_type']:
                            action_clusters.distance_type = cfg['model']['rebuild_model_cfg']['distance_type']
                    else:
                        action_clusters.distance_type = cfg['model']['rebuild_model_cfg']['distance_type']
                model_ema.module.module.action_clusters = action_clusters
                model.module.action_clusters = action_clusters
            logging.info("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint    
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
            return
    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    if model.module.use_rebuild_expected_feat:
        train_one_epoch(
            train_loader_full_seq,
            model,
            optimizer,
            scheduler,
            -1,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer = tb_writer,
            print_freq = args.print_freq,
            cluster_accumulate_method = cfg['model']['rebuild_model_cfg']['cluster_init_method'],
        )


    """4. training / validation loop"""
    logging.info("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    logging.info('Maximum epoch: {:d}'.format(max_epochs))
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer = tb_writer,
            print_freq = args.print_freq,
        )

        if cfg['model']['rebuild_model_cfg']['cluster_update_interval'] > 0 and \
            (epoch + 1) % cfg['model']['rebuild_model_cfg']['cluster_update_interval'] == 0 and \
            epoch < max_epochs * cfg['model']['rebuild_model_cfg']['cluster_update_stop_epoch']:


            next_update_epoch = epoch + cfg['model']['rebuild_model_cfg']['cluster_update_interval']
            if next_update_epoch >= max_epochs * cfg['model']['rebuild_model_cfg']['cluster_update_stop_epoch']:
                # last update
                model.module.rebuild_model_cfg['action_feat_to_center_distance_loss_dist'] = -1
                model.module.rebuild_model_cfg['diversity_in_training'] = 0

            train_one_epoch(
                train_loader_full_seq,
                model,
                optimizer,
                scheduler,
                -1,
                model_ema = model_ema,
                clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
                tb_writer = tb_writer,
                print_freq = args.print_freq,
            )
            
        # save ckpt once in a while
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'action_clusters': model.module.action_clusters if hasattr(model.module, 'action_clusters') else None,
                'state_dict_ema': model_ema.module.state_dict()
            }

            if (epoch + 1) > int(max_epochs * 0.90):
                logging.info('saving checkpoint...')
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
                )
            
            
            
    # wrap up
    tb_writer.close()
    logging.info("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=2, type=int,
                        help='print frequency (default: 2 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--dataset-buffer', default=None, type=str, help='save | load | None')
    args = parser.parse_args()
    main(args)
