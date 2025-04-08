# python imports
import argparse
import os
import glob
import time
from pprint import pformat, pprint
from copy import deepcopy
import pickle
import logging

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, fix_random_seed


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    
    config_name = os.path.basename(args.config).replace('.yaml', '')
    log_subfolder = f'./logs/{config_name}'
    os.makedirs(log_subfolder, exist_ok=True)
    log_file = f'{log_subfolder}/test.log'  # Specify log file path
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    # Add console output to log file
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # assert len(cfg['val_split']) > 0, "Test set must be specified!"
    assert len(cfg['test_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            if args.is_best:
                ckpt_file = os.path.join(args.ckpt, 'best.pth.tar')
            else:
                ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
                ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    logging.info(pformat(cfg))

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    assert args.dataset_buffer is None or args.dataset_buffer == 'save' or args.dataset_buffer == 'load', "Invalid dataset buffer option"
    taskname = cfg['dataset_name'] + '_' + cfg['dataset']['task']
    if args.dataset_buffer == 'save' or args.dataset_buffer is None:
        cfg_dataset = deepcopy(cfg['dataset'])
        cfg_dataset['ckpt_folder'] = args.ckpt
        val_dataset = make_dataset(
                cfg['dataset_name'], False, cfg['test_split'], **cfg_dataset
            )
        if args.dataset_buffer == 'save':
            with open(os.path.join(f'dataset_buffer/{taskname}_test_dataset.pkl'), "wb") as f:
                pickle.dump(val_dataset, f)
    else:
        with open(os.path.join(f'dataset_buffer/{taskname}_test_dataset.pkl'), "rb") as f:
            val_dataset = pickle.load(f)

    # Print length of val_dataset
    logging.info(f"length of val_dataset: {len(val_dataset)}")

    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
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
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    
    output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    logging.info("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()

    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq
    )
    end = time.time()
    logging.info("All done! Total time: {:0.2f} sec".format(end - start))

    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('--is_best', action='store_true')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('--dataset-buffer', default=None, type=str, help='save | load | None')
    args = parser.parse_args()
    main(args)
