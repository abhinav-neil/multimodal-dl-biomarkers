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
    
    def mm_collate_fn(self, batch):
        wsi_feats, report_feats, stil_scores, stil_levels = zip(*batch)
        return list(wsi_feats), list(report_feats), default_collate(stil_scores).float(), default_collate(stil_levels)
    
class MMDataset(Dataset):
    def __init__(self, root_dir='./', data_file='data/data_subtype_grade.csv', split=None, use_rand_splits=True, rand_seed=42):
        """
        Args:
            root_dir (string): path to data directory
            csv_file (string): Path to the file w/ the dataset information.
            split (string): 'train', 'val', or 'test'.
            use_rand_splits (bool): Whether to use random splits or fixed splits.
            rand_seed (int): Random seed to use for splitting the dataset.
        """
        self.root_dir = root_dir
        df = pd.read_csv(data_file, keep_default_na=False)
        
        # Split the dataset into train+val set and test set
        if use_rand_splits:
            train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=rand_seed)
            # Further split the train+val set into separate training and validation sets
            train_df, val_df = train_test_split(train_val_df, test_size=0.11, random_state=rand_seed)  # 0.11 x 0.9 = 0.099 ~ 0.1
        else:
            train_df, val_df, test_df = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']

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
        
        # Extract and convert labels
        region_to_idx = {'ductal': 0, 'lobular': 1, 'mixed': 2, 'NA': 3}
        localization_to_idx = {'in situ': 0, 'invasive': 1, 'metastatic': 2, 'NA': 3}
        grade_to_idx = {'1': 0, '2': 1, '3': 2, 'NA': 3}
        
        region = torch.tensor(region_to_idx[self.df.iloc[idx]['region']], dtype=torch.long)
        local = torch.tensor(localization_to_idx[self.df.iloc[idx]['localization']], dtype=torch.long)
        grade = torch.tensor(grade_to_idx[self.df.iloc[idx]['grade']], dtype=torch.long)
       
        # Return as a dictionary
        return {
            'wsi_feats': wsi_feats,
            'report_feats': report_feats,
            'region': region,
            'local': local,
            'grade': grade
        }

    def mm_collate_fn(self, batch):
        # Extract data from the dictionaries
        wsi_feats = [item['wsi_feats'] for item in batch]
        report_feats = [item['report_feats'] for item in batch]
        regions = [item['region'] for item in batch]
        local = [item['local'] for item in batch]
        grades = [item['grade'] for item in batch]

        # Return as a dict
        batch = {
            'wsi_feats': wsi_feats,
            'report_feats': report_feats,
            'region': default_collate(regions),
            'local': default_collate(local),
            'grade': default_collate(grades)
        }
        return batch

def create_dataloaders(data_type, data_file, root_dir='./', split=None, use_rand_splits=True, rand_seed=42, bsz=64):
    '''
    Creates dataloaders for the specified dataset.
    Inputs:
        data_type (str): 'stils' or 'subtype_grade'
        data_file (str): path to the csv file containing the dataset information
        root_dir (str): path to the root directory containing the dataset
        split (str): 'train', 'val', or 'test'
        use_rand_splits (bool): whether to use random splits or fixed splits
        rand_seed (int): random seed to use for splitting the dataset
        bsz (int): batch size
    '''
    
    assert data_type in ['stils', 'subtype_grade'], f'invalid data type: {data_type}, must be one of stils, subtype_grade'
    data_class = MMSTILDataset if data_type == 'stils' else MMDataset
    
    train_data = data_class(root_dir, data_file, 'train', use_rand_splits, rand_seed=rand_seed)
    val_data = data_class(root_dir, data_file, 'val', use_rand_splits, rand_seed=rand_seed)
    test_data = data_class(root_dir, data_file, 'test', use_rand_splits, rand_seed=rand_seed)

    print(f'size of train set: {len(train_data)}, val set: {len(val_data)}, test set: {len(test_data)}')

    # Create the dataloaders
    collate_fn = data_class.mm_collate_fn
    train_loader = DataLoader(train_data, batch_size=bsz, shuffle=True, num_workers=12, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=bsz, shuffle=False, num_workers=12, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=bsz, shuffle=False, num_workers=12, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader