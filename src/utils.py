import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.model_selection import train_test_split

def set_seed(seed):
    """
    Sets the seed for generating random numbers for Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
                
class MMSTILDataset(Dataset):
    def __init__(self, root_dir='./', data_file='data/stils/data_stils.csv', split=None, use_rand_splits=True, rand_seed=42):
        """
        Args:
            root_dir (string): path to data directory
            csv_file (string): Path to the file w/ the dataset information.
            split (string): 'train', 'val', or 'test'.
            use_rand_splits (bool): Whether to use random splits or fixed splits.
            rand_seed (int): Random seed to use for splitting the dataset.
        """
        self.root_dir = root_dir
        df = pd.read_csv(data_file)
        
        # Split the dataset into train+val set and test set
        if use_rand_splits:
            train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=rand_seed)
        else:
            train_val_df, test_df = df[df['split'] == 'train'], df[df['split'] == 'test']
        # Further split the train+val set into separate training and validation sets
        train_df, val_df = train_test_split(train_val_df, test_size=0.11, random_state=rand_seed)  # 0.11 x 0.9 = 0.099 ~ 0.1

        self.df = train_df if split == 'train' else val_df if split == 'val' else test_df if split == 'test' else df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the tensors
        wsi_feats_path = os.path.join(self.root_dir, self.df.iloc[idx]['wsi_feat_path'])
        report_feats_path = os.path.join(self.root_dir, self.df.iloc[idx]['report_feat_path'])
        wsi_feats = torch.load(wsi_feats_path)
        report_feats = torch.load(report_feats_path)
        
        # Extract labels
        stil_score = self.df.iloc[idx]['stil_score']
        stil_level = self.df.iloc[idx]['stil_lvl']
        
        return wsi_feats, report_feats, stil_score, stil_level
        
    def mm_collate_fn(batch):
        wsi_feats, report_feats, stil_scores, stil_levels = zip(*batch)
        return list(wsi_feats), list(report_feats), default_collate(stil_scores).float(), default_collate(stil_levels)
    
class MMDataset(Dataset):
    def __init__(self, root_dir='./', data_file='data/stils/data_mm_subtype_grade.csv', split=None, use_rand_splits=True, rand_seed=42):
        """
        Args:
            root_dir (string): path to data directory
            csv_file (string): Path to the file w/ the dataset information.
            split (string): 'train', 'val', or 'test'.
            use_rand_splits (bool): Whether to use random splits or fixed splits.
            rand_seed (int): Random seed to use for splitting the dataset.
        """
        self.root_dir = root_dir
        df = pd.read_csv(data_file)
        
        # Split the dataset into train+val set and test set
        if use_rand_splits:
            train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=rand_seed)
        else:
            train_val_df, test_df = df[df['split'] == 'train'], df[df['split'] == 'test']
        # Further split the train+val set into separate training and validation sets
        train_df, val_df = train_test_split(train_val_df, test_size=0.11, random_state=rand_seed)  # 0.11 x 0.9 = 0.099 ~ 0.1

        self.df = train_df if split == 'train' else val_df if split == 'val' else test_df if split == 'test' else df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the tensors
        wsi_feats_path = os.path.join(self.root_dir, self.df.iloc[idx]['wsi_feat_path'])
        report_feats_path = os.path.join(self.root_dir, self.df.iloc[idx]['report_feat_path'])
        wsi_feats = torch.load(wsi_feats_path)
        report_feats = torch.load(report_feats_path)
        
        # Extract labels
        region = self.df.iloc[idx]['region']
        localization = self.df.iloc[idx]['localization']
        grade = self.df.iloc[idx]['grade']
        
        return wsi_feats, report_feats, region, localization, grade
        
    def mm_collate_fn(batch):
        wsi_feats, report_feats, regions, localizations, grades = zip(*batch)
        return list(wsi_feats), list(report_feats), default_collate(regions), default_collate(localizations), default_collate(grades)

