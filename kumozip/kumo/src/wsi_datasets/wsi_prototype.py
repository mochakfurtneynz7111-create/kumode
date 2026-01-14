from __future__ import print_function, division
from os.path import join as j_
import torch
import numpy as np
import pandas as pd
import sys
import os

from torch.utils.data import Dataset
import h5py
sys.path.append('../')
from utils.pandas_helper_funcs import df_sdir, series_diff

class WSIProtoDataset(Dataset):
    """WSI Custer Dataset."""

    def __init__(self,
                 df,
                 data_source,
                 sample_col='slide_id',
                 slide_col='slide_id'):
        """
        Args:
        """
        self.data_source = []
        for src in data_source:
            assert os.path.basename(src) in ['feats_h5', 'feats_pt']
            self.use_h5 = True if os.path.basename(src) == 'feats_h5' else False
            self.data_source.append(src)
        
        self.data_df = df['histo']      # df has both 'histo' and 'gene
        assert 'Unnamed: 0' not in self.data_df.columns
        self.sample_col = sample_col
        self.slide_col = slide_col
        self.data_df[sample_col] = self.data_df[sample_col].astype(str)
        self.data_df[slide_col] = self.data_df[slide_col].astype(str)
        self.X = None
        self.y = None

        self.idx2sample_df = pd.DataFrame({'sample_id': self.data_df[sample_col].astype(str).unique()})
        self.set_feat_paths_in_df()
        self.data_df.index = self.data_df[sample_col].astype(str)
        self.data_df.index.name = 'sample_id'

    def __len__(self):
        return len(self.idx2sample_df)

    def set_feat_paths_in_df(self):
        """
        Sets the feature path (for each slide id) in self.data_df. At the same time, checks that all slides 
        specified in the split (or slides for the cases specified in the split) exist within data source.
        """
        self.feats_df = pd.concat([df_sdir(feats_dir, cols=['fpath', 'fname', self.slide_col]) for feats_dir in self.data_source]).drop(['fname'], axis=1).reset_index(drop=True)
        missing_feats_in_split = series_diff(self.data_df[self.slide_col], self.feats_df[self.slide_col])

        ### Assertion to make sure that there are not any missing slides that were specified in your split csv file
        try:
            assert len(missing_feats_in_split) == 0
        except:
            print(f"Missing Features in Split:\n{missing_feats_in_split}")
            sys.exit()

        ### Assertion to make sure that all slide ids to feature paths have a one-to-one mapping (no duplicated features).
        try:
            self.data_df = self.data_df.merge(self.feats_df, how='left', on=self.slide_col, validate='1:1')
            assert self.feats_df[self.slide_col].duplicated().sum() == 0
        except:
            print("Features duplicated in data source(s). List of duplicated features (and their paths):")
            print(self.feats_df[self.feats_df[self.slide_col].duplicated()].to_string())
            sys.exit()

        self.data_df = self.data_df[list(self.data_df.columns[-1:]) + list(self.data_df.columns[:-1])]

    def get_sample_id(self, idx):
        return self.idx2sample_df.loc[idx]['sample_id']

    def get_feat_paths(self, idx):
        feat_paths = self.data_df.loc[self.get_sample_id(idx), 'fpath']
        if isinstance(feat_paths, str):
            feat_paths = [feat_paths]
        return feat_paths

    def __getitem__(self, idx):
        feat_paths = self.get_feat_paths(idx)

        # Read features (and coordinates, Optional) from pt/h5 file
        all_features = []
        all_coords = []

        for feat_path in feat_paths:
            if self.use_h5:
                with h5py.File(feat_path, 'r') as f:
                    features = f['features'][:]

                    # 🔥 新增：读取坐标
                    if 'coords_patching' in f.keys():
                        coords = f['coords_patching'][:]  # (N, 2)
                    elif 'coords' in f.keys():
                        # 如果没有coords_patching，尝试coords（需要去掉batch维度）
                        coords = f['coords'][:]
                        if len(coords.shape) == 3:  # (1, N, 2)
                            coords = np.squeeze(coords, axis=0)  # → (N, 2)
                    else:
                        # 如果完全没有坐标，使用假坐标
                        print(f"[Warning] No coords found in {feat_path}, using fake coords")
                        coords = None
            else:
                # PT格式
                features = torch.load(feat_path)
                coords = None  # PT格式通常没有坐标

            # 处理features的维度
            if len(features.shape) > 2:
                assert features.shape[0] == 1, f'{features.shape} is not compatible!'
                features = np.squeeze(features, axis=0)

            all_features.append(features)

            # 🔥 新增：收集坐标
            if coords is not None:
                all_coords.append(coords)
            else:
                # 使用假坐标（顺序索引）
                fake_coords = np.arange(len(features)).reshape(-1, 1)
                fake_coords = np.tile(fake_coords, (1, 2)).astype(np.float32)
                all_coords.append(fake_coords)

        # 拼接所有features和coords
        all_features = torch.from_numpy(np.concatenate(all_features, axis=0))
        all_coords = torch.from_numpy(np.concatenate(all_coords, axis=0))  # 🔥 修改

        out = {'img': all_features,
               'coords': all_coords}  # 🔥 现在有真实坐标了！

        return out
        
    
       
