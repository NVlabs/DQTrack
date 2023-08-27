# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS, build_loss
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.track_dq_utils import FFN, QueryInteractionX
from projects.mmdet3d_plugin.core.track.instances import Instances as Tracklet
from nuscenes.eval.detection.config import config_factory

@DETECTORS.register_module()
class DETR3DTrackerDQ(MVXTwoStageDetector):
    """DETR3DTrackerDQ."""
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 bbox_coder=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 loss_cfg=None,
                 tracker_cfg=None,
                 pretrained=None,
                 load_img=None,
                 load_pts=None):
        super(DETR3DTrackerDQ,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        if self.with_img_backbone:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
            in_channels = self.img_neck.out_channels
            out_channels = self.pts_bbox_head.in_channels
            if isinstance(in_channels, list):
                in_channels  = in_channels[0]
            self.use_grid_mask = use_grid_mask
        self.load_img = load_img
        self.load_pts = load_pts
        self.tracker_cfg = tracker_cfg
        self.loss = build_loss(loss_cfg)
        self.test_detector = None
        self.test_tracker = None
        self.scene_token = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.class_dict = tracker_cfg.get("class_dict", None)
        self.train_det_only = tracker_cfg.get("train_det_only", False)
        self.pos_cfg = tracker_cfg.get("pos_cfg", dict())
        self.zero_val = nn.Embedding(tracker_cfg.num_query, 1)
        self.rel_dist_embed = nn.Linear(1, tracker_cfg.embed_dims)
        if not self.train_det_only:
            ffn_channel = tracker_cfg.get("ffn_dims", tracker_cfg.embed_dims)
            self.detector_trans = FFN(tracker_cfg.embed_dims,
                                    ffn_channel,
                                    tracker_cfg.ema_drop)
            self.tracklet_trans = FFN(tracker_cfg.embed_dims,
                                    ffn_channel,
                                    tracker_cfg.ema_drop)
            self.query_inter = QueryInteractionX(
                                    tracker_cfg.embed_dims,
                                    ffn_channel,
                                    **self.tracker_cfg.query_trans)

            # add ffn to position transform
            if 'ffn' in self.pos_cfg.get('pos_trans', 'linear'):
                self.rel_dist_embed = nn.Sequential(
                    self.rel_dist_embed,
                    FFN(tracker_cfg.embed_dims,
                        ffn_channel,
                        tracker_cfg.ema_drop))

            # final trans to embedding
            if 'linear' in self.pos_cfg.get('final_trans', 'sum'):
                self.embed_trans = nn.Linear(tracker_cfg.embed_dims, 1)
            else:
                self.embed_trans = None

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_track = tracker_cfg.num_track
        self.det2track = self.trans_track_cate()
        self.max_track_id = 0
        self.eval_configs = config_factory('detection_cvpr_2019')
        if pts_middle_encoder:
            self.pts_fp16 = self.pts_middle_encoder.fp16_enabled

    def init_weights(self):
        """Initialize weights of the depth head."""
        if self.with_img_backbone:
            if getattr(self.img_backbone, 'init_cfg'):
                self.img_backbone.init_weights()


    def trans_track_cate(self):
        # transfer det cate to track cate if not identical
        if self.tracker_cfg.get('class_dict', None):            
            all_classes = self.tracker_cfg.class_dict['all_classes']
            track_classes = self.tracker_cfg.class_dict['track_classes']
            det2track = [all_classes.index(_cate) for _cate in track_classes]
            det2track = torch.Tensor(det2track)
        else:
            det2track = torch.arange(0,10)
        return det2track.to(dtype=int, device='cuda')

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if img is not None:
            B = img.size(0)
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def embedding_interaction(self, detector):
        det_embedding = detector.output_embedding.clone()    
        track_embedding = self.tracklet_trans(det_embedding)
        obj_embedding = self.detector_trans(det_embedding)
        if self.tracker_cfg.get('with_qim', True):
            # use position embedding
            pos_embedding = None
            # query interaction
            track_embedding, obj_embedding = self.query_inter(track_embedding, 
                                                                obj_embedding,
                                                                pos_embedding)
        return track_embedding, obj_embedding

    def query_asso(self, detector, tracker, obj_embedding, track_embedding, time_delta):
        # update pos embed to current frame
        track_pts = tracker.pos_velo[:,:2].clone()
        track_velo = tracker.pos_velo[:,2:].clone()
        obj_pts = detector.pos_velo[:,:2].clone()
        # detach position
        track_pts = track_pts.detach()
        obj_pts = obj_pts.detach()
        # detach velocity
        track_velo = track_velo.detach()
        fuse_type = self.pos_cfg.get('fuse_type', 'sum')
        # update to current position
        track_pts_now = track_pts + track_velo * time_delta.float()
        
        # calculate relative distance
        rel_dist = (obj_pts[:,None] - track_pts_now[None])**2
        rel_dist = rel_dist.sum(-1, keepdim=True).sqrt()

        motion_embedding = self.rel_dist_embed(rel_dist)
        appear_embedding = obj_embedding[:,None] * track_embedding[None]
        # fuse embedding according to type
        if 'sum' in fuse_type:
            fused_embedding = appear_embedding + motion_embedding
        else:
            raise NotImplementedError
        
        # calculate association matrix
        if self.embed_trans is not None:
            det2track_mat = self.embed_trans(fused_embedding)
            det2track_mat = det2track_mat.sum(-1)
        else:
            det2track_mat = fused_embedding.sum(-1) / \
                            (fused_embedding.shape[-1]**0.5)
        
        return det2track_mat, track_pts_now

    def tracklet_asso_aug(self, tracker, detector, match_info):
        aug_cfg = self.tracker_cfg.get('track_aug', 
                                       dict(drop_prob=0.0, fp_ratio=0.0, trans_noise=0.0))
        # random add FP tracklet during training
        if aug_cfg['fp_ratio'] > 0:
            # undo previous fp track
            fp_track = (tracker.obj_idxes == -2)
            if fp_track.sum() > 0:
                tracker = self._delete_tracks(tracker, fp_track)
            # select background det embedding
            det_mask = torch.ones_like(detector.scores).bool()
            det_mask[match_info['pos_inds']] = False
            # select track cate only
            if aug_cfg.get('track_cate', False) and self.tracker_cfg.get('class_dict', None):
                class_dict = self.tracker_cfg.class_dict
                pred_cls = detector.pred_logits.argmax(dim=1)
                class_mask = [class_dict['all_classes'][_idx] in class_dict['track_classes']
                              for _idx in pred_cls]
                class_mask = torch.Tensor(class_mask).to(dtype=bool, device=det_mask.device)
                det_mask = det_mask & class_mask
            # select fp embedding according to prob
            score_mask = detector.scores > self.tracker_cfg.det_thres
            fp_mask = det_mask & score_mask
            fp_num = int(tracker.valid_track.sum() * aug_cfg['fp_ratio'])
            if fp_mask.sum() > fp_num:
                score_sort = torch.argsort(detector.scores, descending=True)
                fp_mask[score_sort[fp_num:]] = False
            fp_det = detector[fp_mask]
            if len(fp_det) > 0:
                # embedding interaction
                track_embedding, obj_embedding = self.embedding_interaction(fp_det)
                track_idx = (tracker.obj_idxes==-1).nonzero(as_tuple=True)[0][:len(fp_det)]
                tracker.valid_track[track_idx] = True
                # set idx of FP track to -2
                tracker.obj_idxes[track_idx] = -2
                tracker.query_track[track_idx] = track_embedding
                tracker.track_score[track_idx] = fp_det.scores
                tracker.pred_logits[track_idx] = fp_det.pred_logits[:, self.det2track]
                tracker.pred_boxes[track_idx] = fp_det.pred_boxes
                tracker.ref_pts[track_idx] = fp_det.ref_pts
                tracker.pos_velo[track_idx] = fp_det.pos_velo

        # random drop tracklet during training
        if aug_cfg['drop_prob'] > 0:
            # select fn track according to prob
            dead_mask = torch.rand_like(tracker.track_score) < aug_cfg['drop_prob']
            dead_mask = dead_mask & tracker.valid_track

            if self.tracker_cfg.track_aug.get('drop_all', False):
                tracker = self._delete_tracks(tracker, dead_mask)
            else:
                # undo previous fn track
                fn_track = (tracker.obj_idxes < -2)
                if fn_track.sum() > 0:
                    tracker.valid_track[fn_track] = True
                    tracker.obj_idxes[fn_track] *= -1
                # mark fn track
                tracker.valid_track[dead_mask] = False
                tracker.obj_idxes[dead_mask] *= -1                

        # random transional noise
        if 'trans_noise' in aug_cfg:
            trans_noise = torch.rand(tracker.valid_track.sum(),2).to(
                                     device=tracker.valid_track.device)
            trans_noise = (trans_noise-0.5) * 2 * aug_cfg['trans_noise']
            tracker.pos_velo[tracker.valid_track, :2] += trans_noise

        return tracker

    def tracklet_update_train(self, tracker, detector, track_embedding, match_info):
        ema_decay_rate = self.tracker_cfg.ema_decay
        pos_inds = match_info['pos_inds']
        gt_inds = match_info['gt_inds']
        obj_ids = match_info['obj_ids']    
        # update tracklets info
        is_new_born = torch.full((len(gt_inds),), False).to(tracker.obj_idxes.device)
        for _idx, obj_id in enumerate(obj_ids):
            # update track query after asso mat calculation
            if obj_id.item() in tracker.obj_idxes:
                track_idx = (tracker.obj_idxes==obj_id.item()).nonzero(as_tuple=True)[0][0]
                tracker.query_track[track_idx] = ema_decay_rate * tracker.query_track[track_idx] + \
                                                (1 - ema_decay_rate) * track_embedding[_idx]
            else:
                # init new tracklet
                is_new_born[_idx] = True
                track_idx = (tracker.obj_idxes==-1).nonzero(as_tuple=True)[0][0]
                tracker.valid_track[track_idx] = True
                tracker.obj_idxes[track_idx] = obj_id
                tracker.query_track[track_idx] = track_embedding[_idx]

            tracker.track_score[track_idx] = detector.scores[pos_inds[_idx]]
            tracker.pred_logits[track_idx] = detector.pred_logits[pos_inds[_idx]][self.det2track]
            tracker.pred_boxes[track_idx] = detector.pred_boxes[pos_inds[_idx]]
            tracker.ref_pts[track_idx] = detector.ref_pts[pos_inds[_idx]]
            tracker.pos_velo[track_idx] = detector.pos_velo[pos_inds[_idx]]
        
        # use tracklet aug
        tracker = self.tracklet_asso_aug(tracker, detector, match_info)
        
        return tracker

    def tracklet_update_test(self, tracker, detector, track_embedding_nb, time_delta, 
                             match_mat, valid_det, valid_track):
        ema_decay_rate = self.tracker_cfg.ema_decay
        # add position error for constrain
        velo_error = self.tracker_cfg.get('velo_error', [])
        if len(velo_error) > 0 and (time_delta is not None):
            det_pos_prev = detector.pos_velo[valid_det][...,:2] - \
                           detector.pos_velo[valid_det][...,2:] * time_delta
            track_pos = tracker.pos_velo[valid_track][...,:2]
            det_track_dist = ((det_pos_prev[:,None] - track_pos[None])**2).sum(dim=2)
            det_track_dist = det_track_dist.sqrt()
            velo_error = torch.tensor(velo_error).to(tracker.query_track.device)
        else:
            det_track_dist = torch.zeros(valid_det.sum(), valid_track.sum())

        det_cate = detector.pred_logits[valid_det].argmax(1)
        track_cate = tracker.pred_logits[valid_track].argmax(1)
        if self.tracker_cfg.get('class_dict', None):
            class_dict = self.tracker_cfg.class_dict
            invalid = []
            for _det in det_cate:
                _invalid = [class_dict['all_classes'][_det] != class_dict['track_classes'][_track] 
                            for _track in track_cate]
                invalid.append(_invalid)
            invalid = torch.Tensor(invalid).to(dtype=bool, device=det_cate.device)
            det_cate = [class_dict['track_classes'].index(class_dict['all_classes'][_cate]) 
                        for _cate in det_cate]
            det_cate = torch.Tensor(det_cate).to(dtype=int, device=invalid.device)
        else:
            invalid = (det_cate[:,None] != track_cate[None])
        
        invalid = invalid + (match_mat < self.tracker_cfg.asso_thres)
        if len(velo_error) > 0:
            invalid = invalid + (det_track_dist > velo_error[det_cate,None])
        
        assign_mat = match_mat - invalid * 10
        match_pairs = linear_sum_assignment(assign_mat.cpu().numpy(), maximize=True)
        match_pairs = np.array([match_pairs[0], match_pairs[1]]).T
        match_pairs = torch.tensor(match_pairs).to(tracker.query_track.device)
        
        # refine invalid calculate
        valid_mask = ~invalid[match_pairs[:,0],match_pairs[:,1]]
        
        # update info for matched tracklets
        if len(match_pairs) > 0:
            match_dets = match_pairs[valid_mask,0]
            match_tracks = match_pairs[valid_mask,1]
            tmp_tracker = tracker[valid_track]
            tmp_detector = detector[valid_det]
            tmp_tracker.pred_logits[match_tracks] = tmp_detector.pred_logits[match_dets][:, self.det2track]
            tmp_tracker.pred_boxes[match_tracks] = tmp_detector.pred_boxes[match_dets]
            tmp_tracker.ref_pts[match_tracks] = tmp_detector.ref_pts[match_dets]
            tmp_tracker.pos_velo[match_tracks] = tmp_detector.pos_velo[match_dets]
            tmp_tracker.valid_track[match_tracks] = True
            tmp_tracker.long_track[match_tracks] = False
            tmp_tracker.active[match_tracks] = True
            tmp_tracker.track_age[match_tracks] += 1
            tmp_tracker.disappear_time[match_tracks] = 0
            assert (tmp_tracker.obj_idxes[match_tracks] == 0).sum() < 2
            tmp_tracker.query_track[match_tracks] = ema_decay_rate * tmp_tracker.query_track[match_tracks] + \
                                                        (1 - ema_decay_rate) * track_embedding_nb[match_dets]

            tmp_tracker.track_score[match_tracks] = tmp_detector.scores[match_dets]

            raw_tracker = tracker.get_fields()
            new_tracker = tmp_tracker.get_fields()
            for _key in raw_tracker:
                raw_tracker[_key][valid_track.clone()] = new_tracker[_key]

        return tracker, match_pairs, valid_mask

    def tracklet_asso_train(self, tracker, detector, match_info, time_delta=None):
        pos_inds = match_info['pos_inds']
        # embedding interaction
        track_embedding, obj_embedding = self.embedding_interaction(detector[pos_inds])

        # calculate asso mat with new-born object & previous tracklets
        valid_track = tracker.valid_track
        if valid_track.sum() > 0:
            track_embedding_mat = tracker.query_track[valid_track]
            # update position to current system 
            det2track_mat, track_pts_now = self.query_asso(detector[pos_inds],
                                                            tracker[valid_track],
                                                            obj_embedding,
                                                            track_embedding_mat,
                                                            time_delta)
            tracker.pos_velo[valid_track,:2] = track_pts_now
        else:
            det2track_mat = None
        
        # update match info
        match_info['is_new_born'] = None
        if det2track_mat is not None:
            match_info['match_mat'] = det2track_mat.detach()
        else:
            match_info['match_mat'] = det2track_mat
        return det2track_mat, track_embedding, match_info

    def tracklet_asso_test(self, tracker, detector, time_delta=None, img_metas=None):
        # sort det results according to predicted score
        detector = self.sort_and_filter(detector, img_metas)
        # active track at each frame
        tracker.active[:] = False
        # select valide track according to det score
        valid_det = (detector.scores >= self.tracker_cfg.det_thres)
        valid_track = tracker.valid_track.clone()
        # select positive classes for tracking
        if self.tracker_cfg.get('class_dict', None):
            class_dict = self.tracker_cfg.class_dict
            pred_cls = detector.pred_logits.argmax(dim=1)
            det_mask = [class_dict['all_classes'][_idx] in class_dict['track_classes']
                        for _idx in pred_cls]
            det_mask = torch.Tensor(det_mask).to(dtype=bool, device=valid_det.device)
            valid_det = valid_det & det_mask
        
        det_num = valid_det.sum()
        if det_num == 0:
            return tracker

        # embedding interaction
        track_embedding_nb, obj_embedding = self.embedding_interaction(detector[valid_det])
        
        # init new tracker
        if valid_track.sum() == 0:
            tracker.query_track[:det_num] = track_embedding_nb
            tracker.track_score[:det_num] = detector.scores[valid_det]
            tracker.pred_logits[:det_num] = detector.pred_logits[valid_det][:,self.det2track]
            tracker.pred_boxes[:det_num] = detector.pred_boxes[valid_det]
            tracker.ref_pts[:det_num] = detector.ref_pts[valid_det]
            tracker.pos_velo[:det_num] = detector.pos_velo[valid_det]
            tracker.valid_track[:det_num] = True
            tracker.long_track[:det_num] = False
            tracker.active[:det_num] = True
            tracker.disappear_time[:det_num] = 0
            tracker.obj_idxes[:det_num] = self.max_track_id + \
                            torch.arange(0,det_num).to(tracker.obj_idxes.device)
            tracker.track_age[:det_num] += 1
            self.max_track_id += det_num
            
            return tracker

        track_embedding = tracker.query_track[valid_track]
        # use update 
        det2track_mat, track_pts_now = self.query_asso(detector[valid_det],
                                                        tracker[valid_track],
                                                        obj_embedding,
                                                        track_embedding,
                                                        time_delta)
        # update pos embed of all tracks
        tracker[valid_track].pos_velo[:,:2] = track_pts_now
        det2track_mat = det2track_mat.softmax(1)
        tracker, match_pairs, valid_pair = self.tracklet_update_test(
                                                            tracker, 
                                                            detector, 
                                                            track_embedding_nb, 
                                                            time_delta,
                                                            det2track_mat,
                                                            valid_det,
                                                            valid_track)
        # calculate number of det and track
        det_num = det2track_mat.shape[0]
        track_num = det2track_mat.shape[1]
        
        if len(match_pairs) > 0:
            unmatched_dets = torch.tensor([_det for _det in range(det_num) \
                            if not (_det in match_pairs[valid_pair, 0])]).to(det2track_mat.device)
            unmatched_tracks = torch.tensor([_track for _track in range(track_num) \
                            if not (_track in match_pairs[:, 1])]).to(det2track_mat.device)
        else:
            unmatched_dets = torch.arange(0, det_num).to(det2track_mat.device)
            unmatched_tracks = torch.arange(0, track_num).to(det2track_mat.device)

        # generate new-born tracks
        if len(unmatched_dets) > 0:
            # select empty track to fill
            empty_track = (valid_track == False)
            assert empty_track.sum() > 0
            # use higher threshold for new-born objects
            if self.tracker_cfg.get('new_born_thres', None):
                new_born_mask = (detector.scores[valid_det][unmatched_dets] > \
                                self.tracker_cfg.new_born_thres)
                unmatched_dets = unmatched_dets[new_born_mask]

            tmp_mask = empty_track[empty_track]
            tmp_mask[len(unmatched_dets):] = False
            empty_track[empty_track.clone()] = tmp_mask
            # fill empty track
            tracker.query_track[empty_track] = track_embedding_nb[unmatched_dets]
            tracker.track_score[empty_track] = detector.scores[valid_det][unmatched_dets]
            tracker.pred_logits[empty_track] = detector.pred_logits[valid_det][unmatched_dets][:, self.det2track]
            tracker.pred_boxes[empty_track] = detector.pred_boxes[valid_det][unmatched_dets]
            tracker.ref_pts[empty_track] = detector.ref_pts[valid_det][unmatched_dets]
            tracker.pos_velo[empty_track] = detector.pos_velo[valid_det][unmatched_dets]
            tracker.valid_track[empty_track] = True
            tracker.long_track[empty_track] = False
            tracker.active[empty_track] = True
            tracker.disappear_time[empty_track] = 0
            tracker.obj_idxes[empty_track] = self.max_track_id + \
                        torch.arange(0,len(unmatched_dets)).to(tracker.obj_idxes.device)
            tracker.track_age[empty_track] += 1
            self.max_track_id += len(unmatched_dets)
        
        # process unmatched tracks
        if len(unmatched_tracks) > 0:
            tmp_tracker = tracker[valid_track]
            tmp_tracker.active[unmatched_tracks] = False
            tmp_tracker.disappear_time[unmatched_tracks] += 1
            tracker.active[valid_track] = tmp_tracker.active
            tracker.disappear_time[valid_track] = tmp_tracker.disappear_time

        track_mask = tracker.disappear_time > self.tracker_cfg.miss_thres
        dead_track = tracker.valid_track & track_mask
        if dead_track.sum() > 0:
            tracker = self._delete_tracks(tracker, dead_track)

        return tracker

    def sort_and_filter(self, detector, img_metas):
        cls_range_map = self.eval_configs.class_range
        cls_dict = self.tracker_cfg.class_dict
        pred_cls = detector.pred_logits.argmax(dim=1)
        cls_range = [cls_range_map[cls_dict['all_classes'][_idx]] for _idx in pred_cls]
        cls_range = torch.tensor(cls_range).to(pred_cls.device)
        l2e_mat = torch.from_numpy(img_metas[0]['lidar2ego'][0]).to(pred_cls.device)
        ego_pts = torch.cat([detector.ref_pts, 
                                torch.ones(len(detector),1).to(detector.ref_pts.device)],
                            dim=1)
        ego_pts = ego_pts @ l2e_mat.T
        radius = (ego_pts[:,:2] ** 2).sum(dim=1).sqrt()
        detector.scores[radius>cls_range] *= 0.0
        
        return detector


    def trans2real_world(self, detector, img_metas):
        # transfer object center to real world
        # reference points (x,y,z)
        ref_pts = torch.cat([detector.pred_boxes[...,:2],detector.pred_boxes[...,4:5]], dim=1)
        # predicted velocity (vx, vy)
        pred_velo = torch.cat([detector.pred_boxes[...,8:10], 
                              torch.zeros_like(detector.pred_boxes[...,8:9])], dim=1)

        # undo bev-level augmentation
        if 'uni_rot_aug' in img_metas[0]:
            uni_rot_aug = img_metas[0]['uni_rot_aug'].to(device=ref_pts.device)
            uni_rot_aug = uni_rot_aug.inverse()
            ref_pts = ref_pts @ uni_rot_aug
            pred_velo = pred_velo @ uni_rot_aug

        l2g_r = img_metas[0]['l2g_r_mat'].to(device=ref_pts.device)
        l2g_t = img_metas[0]['l2g_t'].to(device=ref_pts.device)
        # transfer to global system
        global_pts = ref_pts @ l2g_r.T + l2g_t
        global_velo = pred_velo @ l2g_r.T

        # position and velocity (x, y, vx, vy)
        detector.pos_velo[...,:2] = global_pts[...,:2]
        detector.pos_velo[...,2:] = global_velo[...,:2]
        detector.ref_pts = ref_pts

        return detector

    def forward_detector(self, img, img_metas, detector):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        pred_dict = self.pts_bbox_head(img_feats, img_metas)

        return pred_dict, detector

    def forward_single(self, img, detector, img_metas, depth_map=None, frame_idx=0):
        pred_dict, detector = self.forward_detector(img, img_metas, detector)
        output_classes = pred_dict['all_cls_scores'][:,0].float()
        output_coords = pred_dict['all_bbox_preds'][:,0].float()
        last_ref_pts = pred_dict['last_ref_points'][0].float()
        last_query_feat = pred_dict['last_query_feat'][0].float()

        # Track score from last stage
        with torch.no_grad():
            det_scores = output_classes[-1].sigmoid().max(dim=-1).values

        # change reference points
        detector.scores = det_scores
        detector.pred_logits = output_classes[-1]  # [300, num_cls]
        detector.pred_boxes = output_coords[-1]  # [300, box_dim]
        detector.output_embedding = last_query_feat  # [300, feat_dim]
        # transfer predicted coord to real world
        detector = self.trans2real_world(detector, img_metas)

        if self.training:
            det_dict = {'pred_logits': output_classes[:,None],
                        'pred_boxes': output_coords[:,None]}
            loss_det, match_info = self.loss.loss_det(det_dict, frame_idx)    
            
            return detector, loss_det, match_info
        else:
            return detector

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      instance_inds=None,
                      depth_map=None,
                      img=None,
                      timestamp=None):
        """Forward training function.
        """
        num_frame = self.tracker_cfg.track_frame
        detector = self.generate_empty_detector()

        if points is not None:
            batch_num = len(points)
        else:
            batch_num = len(img)

        loss_dict = {}
        for batch_idx in range(batch_num):
            # only support single inference
            tracker = self.generate_empty_tracker()
            timestamp_single = [_time[batch_idx] for _time in timestamp]
            # init gt instances!
            loss_dict_single = {}
            gt_tracklet_list = []
            for frame_idx in range(num_frame):
                gt_tracklets = Tracklet((1, 1))
                gt_tracklets.boxes = gt_bboxes_3d[batch_idx][frame_idx]
                gt_tracklets.labels = gt_labels_3d[batch_idx][frame_idx]
                gt_tracklets.obj_ids = instance_inds[batch_idx][frame_idx]
                gt_tracklet_list.append(gt_tracklets)

            # init track loss for single clip
            self.loss.clip_init(gt_tracklet_list)
            self.timestamp = None
            for frame_idx in range(num_frame):
                img_single = img[batch_idx][frame_idx][None]
                img_metas_single = [deepcopy(img_metas[batch_idx])]
                img_metas_single[0]['l2g_r_mat'] = img_metas_single[0]['l2g_r_mat'].data[frame_idx]
                img_metas_single[0]['l2g_t'] = img_metas_single[0]['l2g_t'].data[frame_idx]
                if 'uni_rot_aug' in img_metas_single[0]:
                    img_metas_single[0]['uni_rot_aug'] = img_metas_single[0]['uni_rot_aug'][frame_idx]
                
                img_metas_single[0]['pad_shape'] = img_metas_single[0]['pad_shape'][frame_idx]
                img_metas_single[0]['img_shape'] = img_metas_single[0]['img_shape'][frame_idx]
                img_metas_single[0]['timestamp'] = img_metas_single[0]['timestamp'][frame_idx]
                img_metas_single[0]['lidar2img'] = img_metas_single[0]['lidar2img'][frame_idx]
                img_metas_single[0]['box_type_3d'] = img_metas_single[0]['box_type_3d'][frame_idx]
                img_metas_single[0]['gt_bboxes_3d'] = img_metas_single[0]['gt_bboxes_3d']._data[frame_idx]
                img_metas_single[0]['gt_labels_3d'] = img_metas_single[0]['gt_labels_3d']._data[frame_idx]
                
                
                if frame_idx == 0:
                    time_delta = None
                else:
                    time_delta = timestamp_single[frame_idx] - self.timestamp

                self.timestamp = timestamp_single[frame_idx]
                
                if self.tracker_cfg.get('train_track_only', False):
                    with torch.no_grad():
                        # calculate detection loss and matched pairs for each frame
                        detector, loss_single, info_single = self.forward_single(
                                                                    img_single,
                                                                    detector, 
                                                                    img_metas_single,
                                                                    depth_map,
                                                                    frame_idx)
                else:
                    # calculate detection loss and matched pairs for each frame
                    detector, loss_single, info_single = self.forward_single(
                                                                img_single,
                                                                detector, 
                                                                img_metas_single,
                                                                depth_map,
                                                                frame_idx)
                    # update loss dict and match info
                    loss_dict_single.update(loss_single)
               
                if self.train_det_only:
                    continue
                
                # calculate tacklet asso mat
                det2track_mat, track_embedding, info_single = self.tracklet_asso_train(
                                                                    tracker, 
                                                                    detector, 
                                                                    info_single,
                                                                    time_delta)
                tracker = self._copy_tracks(tracker)
                # calculae tracking loss with the matched pairs
                loss_track = self.loss.loss_track(tracker, 
                                                det2track_mat, 
                                                info_single, 
                                                zero_val=(0*self.zero_val.weight.sum()),
                                                frame_idx=frame_idx)
                loss_dict_single.update(loss_track)
                
                # update tracklet
                tracker = self.tracklet_update_train(tracker, detector, track_embedding, info_single)

            # update loss
            for loss_key in loss_dict_single:
                loss_dict['batch.{}.{}'.format(batch_idx,loss_key)] = loss_dict_single[loss_key]
        
        return loss_dict
    
    def forward_test(self, 
                     img_metas, 
                     points=None, 
                     img=None, 
                     timestamp=None,
                     **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        if img is not None:
            num_frame = img.shape[1]
        else:
            num_frame = len(points[0])
        
        timestamp = timestamp[0]
        scene_token = img_metas[0]['scene_token'][0]
        img_metas[0]['scene_change'] = False
        init_frame = False
        # The initial scene
        if self.test_tracker is None:
            self.scene_token = scene_token
            self.timestamp = timestamp[0]
            init_frame = True

        # Start new scene
        if scene_token != self.scene_token or init_frame:
            detector = self.generate_empty_detector()
            tracker = self.generate_empty_tracker()
            img_metas[0]['scene_change'] = True
            time_delta = None
        else:
            detector = self.test_detector
            tracker = self.test_tracker
            time_delta = timestamp[0] - self.timestamp

        self.timestamp = timestamp[-1]
        assert timestamp[-1] == timestamp[0]
        self.scene_token = scene_token

        results = []
        for frame_idx in range(num_frame):
            img_single = img[frame_idx]
            img_metas_single = deepcopy(img_metas)
            img_metas_single[0]['l2g_r_mat'] = img_metas_single[0]['l2g_r_mat'].data[frame_idx]
            img_metas_single[0]['l2g_t'] = img_metas_single[0]['l2g_t'].data[frame_idx]
            if 'uni_rot_aug' in img_metas_single[0]:
                img_metas_single[0]['uni_rot_aug'] = img_metas_single[0]['uni_rot_aug'][frame_idx]

            img_metas_single[0]['pad_shape'] = img_metas_single[0]['pad_shape'][frame_idx]
            img_metas_single[0]['img_shape'] = img_metas_single[0]['img_shape'][frame_idx]
            img_metas_single[0]['timestamp'] = img_metas_single[0]['timestamp'][frame_idx]
            img_metas_single[0]['lidar2img'] = img_metas_single[0]['lidar2img'][frame_idx]
            img_metas_single[0]['box_type_3d'] = img_metas_single[0]['box_type_3d'][frame_idx]

            detector = self.forward_single(img_single, detector, img_metas_single, frame_idx=frame_idx)
            
            # eval detection results only
            if self.tracker_cfg.get('eval_det_only', False):
                frame_res = self.tracklet2results(detector, img_metas)
            else:
                tracker = self.tracklet_asso_test(tracker, 
                                                  detector,
                                                  time_delta,
                                                  img_metas_single)
                frame_res = self.tracklet2results(tracker, img_metas)
            results.append(frame_res)
        
        self.test_detector = detector
        self.test_tracker = tracker
        return results
    
    def generate_empty_detector(self):
        detector = Tracklet((1, 1))
        num_det = self.tracker_cfg.get('num_query', 300)
        dim = self.tracker_cfg.get('embed_dims', 256) * 2
        device = 'cuda'
        class_dict = self.tracker_cfg.get('class_dict', None)
        num_classes = 10 if class_dict is None else len(class_dict['all_classes'])
        # initialize detector part
        detector.output_embedding = torch.zeros(
            (num_det, dim//2), device=device)
        detector.scores = torch.zeros(
            (num_det,), dtype=torch.float, device=device)
        # x, y, w, l, z, h, sin, cos, vx, vy
        detector.pred_boxes = torch.zeros(
            (num_det, 10), dtype=torch.float, device=device)
        detector.pred_logits = torch.zeros(
            (num_det, num_classes),
            dtype=torch.float, device=device)
        detector.ref_pts = torch.zeros(
            (num_det, 3), dtype=torch.float, device=device)
        detector.pos_velo = torch.zeros(
            (num_det, 4), dtype=torch.float, device=device)
        return detector.to(device)
    
    def generate_empty_tracker(self):
        device = 'cuda'
        class_dict = self.tracker_cfg.get('class_dict', None)
        num_classes = 7 if class_dict is None else len(class_dict['track_classes'])
        self.max_track_id = 0
        # initialize tracker part
        tracker = Tracklet((1, 1))
        num_track = self.tracker_cfg.num_track
        tracker.query_track = torch.zeros(
            (num_track, self.tracker_cfg.embed_dims), device=device)
        tracker.track_score = torch.zeros(
            (num_track,), dtype=torch.float, device=device)
        tracker.pred_logits = torch.zeros(
            (num_track,num_classes), dtype=torch.float, device=device)
        tracker.pred_boxes = torch.zeros(
            (num_track, 10), dtype=torch.float, device=device)
        tracker.ref_pts = torch.zeros(
            (num_track, 3), dtype=torch.float, device=device)
        tracker.pos_velo = torch.zeros(
            (num_track, 4), dtype=torch.float, device=device)
        tracker.valid_track = torch.full(
            (num_track,), False, dtype=torch.bool, device=device)
        tracker.long_track = torch.full(
            (num_track,), False, dtype=torch.bool, device=device)
        tracker.obj_idxes = torch.full(
            (num_track,), -1, dtype=torch.long, device=device)
        tracker.track_age = torch.zeros(
            (num_track,), dtype=torch.float, device=device)
        tracker.active = torch.full(
            (num_track,), False, dtype=torch.bool, device=device)
        tracker.disappear_time = torch.zeros(
            (num_track,), dtype=torch.float, device=device)
        return tracker.to(device)

    def _copy_tracks(self, tgt_tracker):
        tracker = Tracklet((1, 1))
        device = 'cuda'
        tracker.query_track = tgt_tracker.query_track.clone()
        tracker.track_score = tgt_tracker.track_score.clone()
        tracker.pred_logits = tgt_tracker.pred_logits.clone()
        tracker.pred_boxes = tgt_tracker.pred_boxes.clone()
        tracker.ref_pts = tgt_tracker.ref_pts.clone()
        tracker.pos_velo = tgt_tracker.pos_velo.clone()
        tracker.valid_track = tgt_tracker.valid_track.clone()
        tracker.long_track = tgt_tracker.long_track.clone()
        tracker.obj_idxes = tgt_tracker.obj_idxes.clone()
        tracker.track_age = tgt_tracker.track_age.clone()
        tracker.active = tgt_tracker.active.clone()
        tracker.disappear_time = tgt_tracker.disappear_time.clone()
        return tracker.to(device)

    def _delete_tracks(self, tracker, dead_track):
        device = tracker.query_track.device
        num_track = dead_track.sum().item()
        class_dict = self.tracker_cfg.get('class_dict', None)
        num_classes = 7 if class_dict is None else len(class_dict['track_classes'])
        tracker.query_track[dead_track] = torch.zeros(
            (num_track, self.tracker_cfg.embed_dims), device=device)
        tracker.track_score[dead_track] = torch.zeros(
            (num_track,), dtype=torch.float, device=device)
        tracker.pred_logits[dead_track] = torch.zeros(
            (num_track, num_classes), dtype=torch.float, device=device)
        tracker.pred_boxes[dead_track] = torch.zeros(
            (num_track, 10), dtype=torch.float, device=device)
        tracker.ref_pts[dead_track] = torch.zeros(
            (num_track, 3), dtype=torch.float, device=device)
        tracker.pos_velo[dead_track] = torch.zeros(
            (num_track, 4), dtype=torch.float, device=device)
        tracker.valid_track[dead_track] = torch.full(
            (num_track,), False, dtype=torch.bool, device=device)
        tracker.long_track[dead_track] = torch.full(
            (num_track,), False, dtype=torch.bool, device=device)
        tracker.obj_idxes[dead_track] = torch.full(
            (num_track,), -1, dtype=torch.long, device=device)
        tracker.track_age[dead_track] = torch.zeros(
            (num_track,), dtype=torch.float, device=device)
        tracker.active[dead_track] = torch.full(
            (num_track,), False, dtype=torch.bool, device=device)
        tracker.disappear_time[dead_track] = torch.zeros(
            (num_track,), dtype=torch.float, device=device)
        return tracker

    def tracklet2results(self, tracker, img_metas):
        '''
        Outs:
        active_tracklets. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        '''
        if not self.tracker_cfg.get('eval_det_only', False):
            # select activate track
            valid_mask = tracker.valid_track & tracker.active
            tracker = tracker[valid_mask]

        if len(tracker) == 0:
            return None
        
        bbox_dict = dict(
            cls_scores=tracker.pred_logits,
            bbox_preds=tracker.pred_boxes,
        )

        if self.tracker_cfg.get('eval_det_only', False):
            bbox_dict['track_scores'] = tracker.scores
            bbox_dict['obj_idxes'] = torch.full((len(tracker),), -1, 
                                        dtype=torch.long, device=tracker.scores.device)
            bbox_dict['num_classes'] = 10
        else:
            bbox_dict['track_scores'] = tracker.track_score
            bbox_dict['obj_idxes'] = tracker.obj_idxes
            bbox_dict['num_classes'] = 7

        bboxes_dict = self.bbox_coder.decode(bbox_dict)[0]

        bboxes = bboxes_dict['bboxes']
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        bboxes = img_metas[0]['box_type_3d'][0](bboxes, 9)
        labels = bboxes_dict['labels']
        scores = bboxes_dict['scores']

        track_scores = bboxes_dict['track_scores']
        obj_idxes = bboxes_dict['obj_idxes']
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            track_ids=obj_idxes.cpu(),
            eval_det=self.tracker_cfg.get('eval_det_only', False)
        )

        return result_dict