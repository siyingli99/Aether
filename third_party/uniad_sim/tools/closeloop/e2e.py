import logging, sys

# 将 logging 的默认输出从 stdout 改到 stderr
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,                # <= 这里改成 stderr
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


import fractions, math
fractions.gcd = math.gcd
import sys
sys.path.append('')
import argparse
import cv2
import torch
import sklearn
import mmcv
import os
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
# from mmdet3d.datasets import build_dataset
# from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
# from projects.mmdet3d_plugin.uniad.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
from queue import Queue
import pickle
from mmdet3d.core import Box3DMode
from dataparser import parse_raw
from tools.closeloop.visualizer import draw_bev, draw_proj
from visualizer import save_visualize_img, to_video
import multiprocessing

warnings.filterwarnings("ignore")

# UNIAD_VIS=0 disables the per-step BEV/camera visualization (saves CPU time).
VISUALIZE = os.environ.get('UNIAD_VIS', '1') != '0'
MAX_VIS_WORKERS = max(1, int(os.environ.get('UNIAD_VIS_WORKERS', '4')))


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def _safe_visualize(outputs, raw_images, save_fn, cam_params=None):
    try:
        save_visualize_img(outputs, raw_images, save_fn, cam_params)
    except Exception as error:  # visualization must never break the loop
        print(f'[uniad-vis] {save_fn}: {type(error).__name__}: {error}', flush=True)


def _finish_visualization(processes, output_folder):
    for process in processes:
        process.join()
    try:
        to_video(output_folder)
    except Exception as error:
        print(f'[uniad-vis] video export failed: {error}', flush=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--output', type=str, required=True)
    # parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']

    model = MMDataParallel(model, device_ids=[0]).eval()
    
    cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
               'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    img_norm_cfg = {
        'mean': np.array([103.530, 116.280, 123.675]), 
        'std': np.array([1.0, 1.0, 1.0]), 
        'to_rgb': False
    }
    history = Queue()
    
    os.makedirs(args.output, exist_ok=True)
    obs_pipe = os.path.join(args.output, 'obs_pipe')
    plan_pipe = os.path.join(args.output, 'plan_pipe')
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print('Ready for recieving')
    cnt = 0
    vis_process_pool = []
    output_folder = os.path.join(args.output, 'uniad')
    os.makedirs(output_folder, exist_ok=True)
    while True:
        with open(obs_pipe, "rb") as pipe:
            payload = pipe.read()
        try:
            raw_data = pickle.loads(payload)
        except Exception:
            print(f"FAILED TO UNPICKLE observation, length={len(payload)}, "
                  f"head={payload[:50]!r}, tail={payload[-50:]!r}")
            raise
        print('received')

        if isinstance(raw_data, str) and raw_data == 'Done':
            _finish_visualization(vis_process_pool, output_folder)
            exit(0)

        data = parse_raw(raw_data, cameras, img_norm_cfg, history)
        raw_images = data['raw_imgs']
        del data['raw_imgs']

        try:
            with torch.no_grad():
                results = model(return_loss=False, rescale=True, **data)
        except RuntimeError as e:
            results = None
            print(f'UniAD inference failed: {e}', flush=True)

        if results is None:
            # Tell AETHER to stop instead of leaving it waiting on plan_pipe.
            with open(plan_pipe, "wb") as pipe:
                pipe.write(pickle.dumps(None))
            print('Waiting for visualize tasks...')
            _finish_visualization(vis_process_pool, output_folder)
            exit(1)

        plan_traj = results[0]['planning']['result_planning']['sdc_traj'][0].detach().cpu().numpy()
        with open(plan_pipe, "wb") as pipe:
            pipe.write(pickle.dumps(plan_traj))
        print('sent')

        if VISUALIZE:
            # results post processing for the visualizer; everything is moved to
            # CPU first because the worker is a forked process and must not
            # touch the parent's CUDA context.
            vis = _to_cpu(results[0])
            vis['planning_traj'] = vis['planning']['result_planning']['sdc_traj'].numpy()
            vis['pts_bbox']['score_list'] = vis['pts_bbox']['score_list'].numpy()
            vis['pts_bbox']['lane_score'] = vis['pts_bbox']['lane_score'].numpy()
            vis['command'] = data['command'][0].cpu().numpy()
            save_fn = osp.join(output_folder, f'{str(cnt).zfill(4)}')
            # Bound the number of concurrent workers.
            while len([p for p in vis_process_pool if p.is_alive()]) >= MAX_VIS_WORKERS:
                time.sleep(0.05)
            process = multiprocessing.Process(target=_safe_visualize,
                                              args=(vis, raw_images, save_fn,
                                                    raw_data[1].get('cam_params')))
            vis_process_pool.append(process)
            process.start()

        cnt += 1


if __name__ == '__main__':
    main()
