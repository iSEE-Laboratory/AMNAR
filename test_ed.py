# python imports
import os
import glob
import time
import pickle
import argparse
from pprint import pformat, pprint
from copy import deepcopy
import time
import logging

# torch imports
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

# our code
from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.datasets import make_dataset, make_data_loader, to_frame_wise, to_segments
from libs.utils import valid_one_epoch, fix_random_seed

################################################################################
def main(args):
    """0. load config"""
    st_time = time.time()
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    
    config_name = os.path.basename(args.config).replace('.yaml', '')
    log_subfolder = f'./logs/{config_name}'
    os.makedirs(log_subfolder, exist_ok=True)
    log_file = f'{log_subfolder}/test_ed.log'  # Specify log file path
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    # 将控制台输出也添加到日志中
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # 移除 validation set 的强制要求
    use_val_set = cfg['val_split'] is not None

    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
        args.ckpt = os.path.dirname(args.ckpt)
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    logging.info(f'Init Config Time: {time.time()-st_time:.3f}s')
    st_time = time.time()


    """2. create dataset / dataloader"""
    # ------------------------------------------------------------------------------------
    assert args.dataset_buffer is None or args.dataset_buffer == 'save' or args.dataset_buffer == 'load', "Invalid dataset buffer option"
    taskname = cfg['dataset_name'] + '_' + cfg['dataset']['task']
    if args.dataset_buffer == 'save' or args.dataset_buffer is None:
        # set cfg
        temp_train_cfg = deepcopy(cfg['dataset'])
        temp_train_cfg['online_test'] = False
        temp_train_cfg['ckpt_folder'] = args.ckpt
        
        train_dataset = make_dataset(
            cfg['dataset_name'], False, cfg['train_split'], **temp_train_cfg
        )
        
        # 条件性创建验证集
        val_dataset = None
        if use_val_set:
            temp_val_cfg = deepcopy(cfg['dataset'])
            temp_val_cfg['ckpt_folder'] = args.ckpt
            val_dataset = make_dataset(
                cfg['dataset_name'], False, cfg['val_split'], **temp_val_cfg
            )
        
        # 创建测试集
        temp_test_cfg = deepcopy(cfg['dataset'])
        temp_test_cfg['ckpt_folder'] = args.ckpt
        
        
        test_dataset = make_dataset(
            cfg['dataset_name'], False, cfg['test_split'], **temp_test_cfg
        )

        
        if args.dataset_buffer == 'save':
            os.makedirs('dataset_buffer', exist_ok=True)
            with open(os.path.join(f'dataset_buffer/{taskname}_train_dataset.pkl'), "wb") as f:
                pickle.dump(train_dataset, f)
            if use_val_set:
                with open(os.path.join(f'dataset_buffer/{taskname}_val_dataset.pkl'), "wb") as f:
                    pickle.dump(val_dataset, f)
            with open(os.path.join(f'dataset_buffer/{taskname}_test_dataset.pkl'), "wb") as f:
                pickle.dump(test_dataset, f)
    else:
        with open(os.path.join(f'dataset_buffer/{taskname}_train_dataset.pkl'), "rb") as f:
            train_dataset = pickle.load(f)
        if use_val_set:
            with open(os.path.join(f'dataset_buffer/{taskname}_val_dataset.pkl'), "rb") as f:
                val_dataset = pickle.load(f)
        with open(os.path.join(f'dataset_buffer/{taskname}_test_dataset.pkl'), "rb") as f:
            test_dataset = pickle.load(f)
            


    # set bs = 1, and disable shuffle
    train_loader = make_data_loader(
        train_dataset, False, None, 1, cfg['loader']['num_workers']
    )
    val_loader = None
    if use_val_set:
        val_loader = make_data_loader(
            val_dataset, False, None, 1, cfg['loader']['num_workers']
        )
    test_loader = make_data_loader(
        test_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    logging.info(f'Prepare Data Time: {time.time()-st_time:.3f}s')
    st_time = time.time()
    # ------------------------------------------------------------------------------------





    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    
    # model.init_prototypes()

    # not ideal for multi GPU training
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    logging.info("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    logging.info("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    # Load additional class member states
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
        model.module.action_clusters = action_clusters
    del checkpoint

    output_name = 'pred_seg_results_'

    logging.info(f'Load Model Time: {time.time()-st_time:.3f}s')




    """5. Test the model"""
    logging.info("\nStart testing model {:s} ...".format(cfg['model_name']))
    print_freq = args.print_freq
    """Test the model on the validation set"""
    model.eval()
    
    
    # loop over training set
    for iter_idx, video_list in enumerate(train_loader, 0):
        with torch.no_grad():
            if iter_idx != len(train_loader) - 1:
                output = model(video_list, mode='calculate_feature_difference')
            else:
                # 如果没有验证集，在训练集最后一个batch计算统计量
                if not use_val_set:
                    output = model(video_list, mode='calculate_feature_difference calculate_mean_std build_final_mean_std', weight=0.99)
                else:
                    output = model(video_list, mode='calculate_feature_difference')

        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            torch.cuda.synchronize()
            logging.info('Train: [{0:05d}/{1:05d}]\t'.format(iter_idx, len(train_loader)))

    logging.info('Training set done!')

    # 条件性处理验证集
    if use_val_set:
        logging.info(f'Size of validation set: {len(val_loader)}')
        for iter_idx, video_list in enumerate(val_loader, 0):
            with torch.no_grad():
                if iter_idx != len(val_loader) - 1:
                    output = model(video_list, mode='calculate_feature_difference')
                else:
                    output = model(video_list, mode='calculate_feature_difference calculate_mean_std build_final_mean_std', weight=0.99)

            if (iter_idx != 0) and iter_idx % (print_freq) == 0:
                torch.cuda.synchronize()
                logging.info('Val: [{0:05d}/{1:05d}]\t'.format(iter_idx, len(val_loader)))


    for ratio in range(-20, 21):
        threshold = ratio / 10
        output_file = os.path.join(args.ckpt, output_name+'%.2f.pkl'%(threshold))

        # loop over test set
        inference_results = {}
        for iter_idx, video_list in enumerate(test_loader, 0):
            # forward the model (wo. grad)
            with torch.no_grad():
                
                # error detection
                output = model(video_list, mode='inference', threshold=threshold)

                for sample_output in output:
                    video_id = sample_output['video_id']
                    if video_id not in inference_results:
                        inference_results[video_id] = {
                            'segments': sample_output['segments'],
                            'score': sample_output['scores'].numpy(),
                            # 'label': sample_output['labels'],
                            'diff_and_threshold': sample_output['diff_and_threshold'],
                            'all_diff_and_threshold': sample_output['all_diff_and_threshold'],
                            'original_labels': sample_output['original_labels'],
                            'corrected_labels': sample_output['corrected_labels'],
                            'quantile_labels': sample_output['quantile_labels']
                        }

            # printing
            if (iter_idx != 0) and iter_idx % (print_freq) == 0:
                torch.cuda.synchronize()
                logging.info('Threshold:%.3f, Test: [%05d/%05d]\t'%(threshold, iter_idx, len(test_loader)))

        with open(output_file, "wb") as f:
            pickle.dump(inference_results, f)
    logging.info('Test set done!')

    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str,
                        help='path to a checkpoint. If it is a folder, the latest checkpoint will be used. If it is a .pth.tar file, it will be used directly.')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        help='print frequency (default: 20 iterations)')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--mode', default='similarity', type=str)  
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--dataset-buffer', default=None, type=str, help='save | load | None')
    args = parser.parse_args()
    main(args)
