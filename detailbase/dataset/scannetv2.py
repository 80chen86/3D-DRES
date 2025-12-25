import glob, json, os
import math
import numpy as np
import os.path as osp
import pointgroup_ops
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import torch
import torch_scatter
from torch.utils.data import Dataset
from typing import Dict, Sequence, Tuple, Union
from tqdm import tqdm
from gorilla import is_main_process
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./backbone/mpnet-base')

class ScanNetDataset_sample_graph_edge(Dataset):

    def __init__(self,
                 data_root,
                 prefix,
                 suffix,
                 voxel_cfg=None,
                 training=True,
                 with_label=True,
                 mode=4,
                 with_elastic=True,
                 aug=False,
                 use_xyz=True,
                 logger=None,
                 max_des_len=78,
                 lang_num_max=8,
                 ):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.with_label = with_label
        self.mode = mode
        self.with_elastic = with_elastic
        self.aug = aug
        self.use_xyz = use_xyz
        self.logger = logger
        self.max_des_len = max_des_len
        self.sp_filenames = self.get_sp_filenames()
        
        np.random.seed(1999)
        
        # load scanrefer
        if self.prefix == 'train':
            self.scanrefer = json.load(open('data/DetailRefer/DetailRefer_train.json'))
            if is_main_process(): self.logger.info(f'Load {self.prefix} scanrefer: {len(self.scanrefer)} samples')
        elif self.prefix == 'val':
            self.scanrefer = json.load(open('data/DetailRefer/DetailRefer_val.json.json'))
            if is_main_process(): self.logger.info(f'Load {self.prefix} scanrefer: {len(self.scanrefer)} samples')
        elif self.prefix == 'test':
            self.scanrefer = json.load(open('data/DetailRefer/DetailRefer_test.json.json'))
            if is_main_process(): self.logger.info(f'Load {self.prefix} scanrefer: {len(self.scanrefer)} samples')
        else:
            raise ValueError('ScanRefer only support train and val split, not support %s' % self.prefix)
        
        self.scanrefer.sort(key=lambda x: x['scene_id'])
        scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))
        scanrefer_new = []
        scanrefer_new_scene = []
        scene_id = ""
        for data in self.scanrefer:
            if data["scene_id"] in scene_list:
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_new_scene) > 0:
                        scanrefer_new.append(scanrefer_new_scene)
                    scanrefer_new_scene = []
                if len(scanrefer_new_scene) >= lang_num_max:
                    scanrefer_new.append(scanrefer_new_scene)
                    scanrefer_new_scene = []
                scanrefer_new_scene.append(data)
        scanrefer_new.append(scanrefer_new_scene)
        self.scene_inputs = scanrefer_new

    def get_sp_filenames(self):
        if self.prefix == 'test': filenames = glob.glob(osp.join(self.data_root, 'scannetv2', 'val', '*' + '_refer.pth'))
        else: filenames = glob.glob(osp.join(self.data_root, 'scannetv2', self.prefix, '*' + '_refer.pth'))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
        return filenames
        
    def load(self, filename):
        if self.with_label:
            return torch.load(filename)
        else:
            xyz, rgb, superpoint = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, superpoint, dummy_sem_label, dummy_inst_label
        
    def transform_train(self, xyz, rgb, superpoint, semantic_label, instance_label):
        if self.aug:
            xyz_middle = self.data_aug(xyz, True, True, True)
        else:
            xyz_middle = xyz.copy()
        rgb += np.random.randn(3) * 0.1
        xyz = xyz_middle * self.voxel_cfg.scale
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        xyz = xyz - xyz.min(0)
        valid_idxs = self.sample_rdn(xyz)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = instance_label[valid_idxs]
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def transform_test(self, xyz, rgb, superpoint, semantic_label, instance_label):
        xyz_middle = xyz
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = instance_label[valid_idxs]
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def sample_rdn(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        if xyz.shape[0] > self.voxel_cfg.max_npoint:
            valid_idxs = np.random.choice(
                xyz.shape[0],
                self.voxel_cfg.max_npoint,
                replace=xyz.shape[0] < self.voxel_cfg.max_npoint
            )
            return valid_idxs
        else:
            valid_idxs = np.ones(xyz.shape[0], dtype=bool)
            return valid_idxs

    def elastic(self, xyz, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])

        return xyz + g(xyz) * mag
    
    def get_ref_mask(self, coord_float, instance_label, superpoint, t2o): 
        gt_pmasks, gt_spmasks = [], []
        for k in t2o.keys():
            obj_ids = t2o[k]
            for i in range(len(obj_ids)):
                if i == 0: ref_lbl = instance_label == obj_ids[i]
                else: ref_lbl = ref_lbl | (instance_label == obj_ids[i])
            gt_spmask = torch_scatter.scatter_mean(ref_lbl.float(), superpoint, dim=-1)
            gt_spmask = (gt_spmask > 0.5).float()
            gt_pmask = ref_lbl.float()
            
            gt_pmasks.append(gt_pmask)
            gt_spmasks.append(gt_spmask)
        return gt_pmasks, gt_spmasks
    
    def get_aux_info(self, t2o, scan_id, lang_tokens):
        is_long, is_complex = [], []
        for k in t2o.keys():
            if len(lang_tokens) > 50: is_long.append(1)
            else: is_long.append(0)
            if len(t2o.keys()) >= 4: is_complex.append(1)
            else: is_complex.append(0)
        return is_long, is_complex
    
    def __len__(self):
        return len(self.scene_inputs)
    
    def __getitem__(self, index: int) -> Tuple:
        ann_ids, gt_pmasks, gt_spmasks, lang_tokenss, t2os, is_longs, is_complexs = [],[],[],[],[],[],[]
        scene_input = self.scene_inputs[index]
        for i in range(len(scene_input)):
            data = scene_input[i]
            scan_id = data['scene_id']
            
            if i==0:
                for fn in self.sp_filenames:
                    if scan_id in fn:
                        sp_filename = fn
                        break
                scene = self.load(sp_filename)
                scene = self.transform_train(*scene) if self.training else self.transform_test(*scene)
                xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label = scene
                coord = torch.from_numpy(xyz).long()
                coord_float = torch.from_numpy(xyz_middle).float()
                feat = torch.from_numpy(rgb).float()
                superpoint = torch.from_numpy(superpoint)
                semantic_label = torch.from_numpy(semantic_label).long()
                instance_label = torch.from_numpy(instance_label).long()
            
            ann_id = int(data['ann_id'])
            lang_tokens = data['token']
            if 't2o' in data.keys():
                t2o = data['t2o']
            else:
                t2o = {"-1":[int(data['object_id'])]}
            gt_pmask, gt_spmask = self.get_ref_mask(coord_float, instance_label, superpoint, t2o)
            is_long, is_complex = self.get_aux_info(t2o, scan_id, lang_tokens)
            
            is_longs.append(is_long)
            is_complexs.append(is_complex)
            ann_ids.append(ann_id)
            t2os.append(t2o)
            gt_pmasks.extend(gt_pmask)
            gt_spmasks.extend(gt_spmask)
            lang_tokenss.append(lang_tokens)

        return ann_ids, scan_id, coord, coord_float, feat, superpoint, t2os, gt_pmasks, gt_spmasks, lang_tokenss, is_longs, is_complexs
    
    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        ann_ids, scan_ids, coords, coords_float, feats, superpoints, t2os, gt_pmasks, gt_spmasks, lang_tokenss, lang_masks, lang_words = [], [], [], [], [], [], [], [], [], [], [], []
        batch_offsets = [0]
        scenes_len = []
        superpoint_bias = 0
        is_longs, is_complexs = [],[]
        lang_tokens = []

        for i, data in enumerate(batch):
            ann_id, scan_id, coord, coord_float, feat, src_superpoint, t2o, gt_pmask, gt_spmask, lang_token, is_long, is_complex = data
            
            superpoint = src_superpoint + superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            scenes_len.append(len(ann_id))
            batch_offsets.append(superpoint_bias)

            is_longs.extend(is_long)
            is_complexs.extend(is_complex)
            ann_ids.extend(ann_id)
            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            
            t2os.extend(t2o)
            
            gt_pmasks.extend(gt_pmask)
            gt_spmasks.extend(gt_spmask)
            
            lang_tokens.extend(lang_token)
            
        max_len = 0
        for lang_token in lang_tokens:
            if len(lang_token) > max_len:
                max_len = len(lang_token)
        for txt_id, lang_token in enumerate(lang_tokens):
                # mpnet
            token_dict = tokenizer(lang_token, is_split_into_words=False, add_special_tokens=True, truncation=True, max_length=max_len+2, padding='max_length', return_attention_mask=True,return_tensors='pt',)
            token_dict['input_ids'][0][1:len(lang_token)+1] = token_dict['input_ids'][:,1]
            token_dict['input_ids'][0][len(lang_token)+1] = 2
            token_dict['attention_mask'][0][token_dict['input_ids'][0]!=1]=1
            token_dict['input_ids'] = token_dict['input_ids'][0].unsqueeze(0)
            token_dict['attention_mask'] = token_dict['attention_mask'][0].unsqueeze(0)
            
            lang_words.append(lang_token)
            lang_tokenss.append(token_dict['input_ids']) 
            lang_masks.append(token_dict['attention_mask'])

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]
        scenes_len = torch.tensor(scenes_len, dtype=torch.int) #int [B]
        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]
        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)
        # voxelize
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0], None)  # long [3]
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords, len(batch), self.mode)

        lang_tokenss = torch.cat(lang_tokenss, 0)
        lang_masks = torch.cat(lang_masks, 0).int()

        return {
            'ann_ids': ann_ids,
            'scan_ids': scan_ids,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'spatial_shape': spatial_shape,
            'feats': feats,
            'superpoints': superpoints,
            'batch_offsets': batch_offsets,
            't2os': t2os,
            'gt_pmasks': gt_pmasks,
            'gt_spmasks': gt_spmasks,
            'lang_tokenss': lang_tokenss,
            'lang_masks': lang_masks,
            'coords_float': coords_float,
            'scenes_len': scenes_len,
            'is_longs': is_longs,
            'is_complexs': is_complexs,
        }