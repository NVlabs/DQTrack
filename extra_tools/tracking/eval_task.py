# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE

import mmcv
import argparse
import json
from os import path as osp

CLASSES_10 = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'barrier')

CLASSES_7 = (
    'car', 'truck', 'bus', 'trailer',
    'bicycle', 'motorcycle', 'pedestrian',
)

ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking to Det Evaluation")
    parser.add_argument('track_result', help='tracking result file path')
    parser.add_argument('--eval_track', action="store_true", help='eval detection results')
    parser.add_argument('--all_cate', action="store_true", help='eval results with 7 categories')
    args = parser.parse_args()

    return args

def _evaluate_single(result_path,
                    eval_track,
                    all_cate,
                    logger=None,
                    metric='bbox',
                    result_name='pts_bbox',
                    data_root='data/nuscenes/',
                    version='v1.0-trainval',
                    eval_version='detection_cvpr_2019',):
    """Evaluation for a single model in nuScenes protocol.

    Args:
        result_path (str): Path of the result file.
        logger (logging.Logger | str, optional): Logger used for printing
            related information during evaluation. Default: None.
        metric (str, optional): Metric name used for evaluation.
            Default: 'bbox'.
        result_name (str, optional): Result name in the metric prefix.
            Default: 'pts_bbox'.

    Returns:
        dict: Dictionary of evaluation details.
    """
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory as track_configs
    eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
    output_dir = osp.join(*osp.split(result_path)[:-1])

    if not eval_track:
        eval_detection_configs = config_factory(eval_version)

        if not all_cate:
            class_range = {}
            for _key in CLASSES_7:
                class_range[_key] = eval_detection_configs.class_range[_key]
            # eval_detection_configs.class_range = class_range
            eval_detection_configs.class_names = class_range.keys()
            CLASSES = CLASSES_7
        else:
            CLASSES = CLASSES_10

        nusc = NuScenes(
            version=version, dataroot=data_root, verbose=False)
        
        nusc_eval = NuScenesEval(
            nusc,
            config=eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'

        for name in CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                    ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
    
    else:
        cfg = track_configs("tracking_nips_2019")
        nusc_eval = TrackingEval(
            config=cfg,
            result_path=result_path,
            eval_set=eval_set_map[version],
            output_dir=output_dir,
            verbose=True,
            nusc_version=version,
            nusc_dataroot=data_root
        )
        metrics = nusc_eval.main()

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        print(metrics)
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        keys = ['amota', 'amotp', 'recall', 'motar',
                'gt', 'mota', 'motp', 'mt', 'ml', 'faf',
                'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
        for key in keys:
            detail['{}/{}'.format(metric_prefix, key)] = metrics[key]
    
    return detail


if __name__ == '__main__':
    args = parse_args()
    result_files = args.track_result
    eval_track = args.eval_track
    all_cate = args.all_cate

    print("Start evaluating.....\n")

    results_dict = _evaluate_single(result_files, eval_track, all_cate)

    for _key in results_dict:
        print(_key, ": ", results_dict[_key], "\n")