import os
import random
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
import torch
from torch.utils.data import Dataset, DataLoader, default_collate, WeightedRandomSampler
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
                   
class MMDataset(Dataset):
    def __init__(self, target, data_file, wsi_feats_dir, report_feats_dir, split=None, use_rand_splits=True, rand_seed=42):
        """
        Args:
            target (string): 'stils', 'subtype_grade', or 'msi'.
            wsi_feats_dir (string): Path to the directory containing the WSI features.
            report_feats_dir (string): Path to the directory containing the report features.
            csv_file (string): Path to the file w/ the dataset information.
            split (string): 'train', 'val', or 'test'.
            use_rand_splits (bool): Whether to use random splits or fixed splits.
            rand_seed (int): Random seed to use for splitting the dataset.
        """
        assert target in ['stils', 'msi', 'region', 'local', 'grade'], f'invalid target: {target}, must be one of stils, msi, region, local, or grade'
        self.target = target
        self.wsi_feats_dir = wsi_feats_dir
        self.report_feats_dir = report_feats_dir
        
        # Define mappings for each target
        self.label2idx = {
            'stils': None,
            'msi': {'MSS': 0, 'MSI-L': 0, 'MSI-H': 1},
            'region': {'ductal': 0, 'lobular': 1, 'mixed': 2, 'NA': 3},
            'local': {'in situ': 0, 'invasive': 1, 'metastatic': 2, 'NA': 3},
            'grade': {'1': 0, '2': 1, '3': 2, 'NA': 3}
        }
        
        df = pd.read_csv(data_file)
        
        # drop rows w/ missing labels
        df = df.dropna(subset=['stil_score']) if target == 'stils' else df.dropna(subset=['msi_sensor']) if target == 'msi' else df.dropna(subset=[target])
        
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

    def get_labels(self):
        if self.target == 'stils':
            return self.df['stil_score'].values
        elif self.target == 'msi':
            return self.df['msi_sensor'].map(self.label2idx['msi']).values
        else:
            return self.df[self.target].map(self.label2idx[self.target]).values

        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the tensors
        wsi_feats_path = f'{self.wsi_feats_dir}/{self.df.iloc[idx]["wsi_id"]}.wsi.pt'
        report_feats_path = f'{self.report_feats_dir}/{self.df.iloc[idx]["case_id"]}.report.pt'
        wsi_feats = torch.load(wsi_feats_path).float()  # Convert to Float tensor
        report_feats = torch.load(report_feats_path).float()  # Convert to Float tensor
        
        item = {'wsi_feats': wsi_feats, 'report_feats': report_feats}
        
        if self.target == 'stils':
            item['stil_score'] = torch.tensor(self.df.iloc[idx]['stil_score'], dtype=torch.float)  # Convert to Float tensor
        elif self.target == 'msi':
            item['msi'] = torch.tensor(self.label2idx['msi'][self.df.iloc[idx]['msi_sensor']], dtype=torch.long)
        else:
            item[self.target] = torch.tensor(self.label2idx[self.target][self.df.iloc[idx][self.target]], dtype=torch.long)

        return item
    
def mm_collate_fn(batch):
    # Extract data from the dictionaries
    batch_keys = batch[0].keys()
    batch = {key: default_collate([item[key] for item in batch]) if key != 'wsi_feats' and key != 'report_feats' else [item[key] for item in batch] for key in batch_keys}
    return batch

def create_dataloaders(target, data_file, wsi_feats_dir, report_feats_dir,  use_rand_splits=True, stratify=None, rand_seed=42, bsz=64, resample=False):
    '''
    Creates dataloaders for the specified dataset.
    Inputs:
        target (str): 'stils', 'subtype_grade', or 'msi'
        data_file (str): path to the csv file containing the dataset information
        use_rand_splits (bool): whether to use random splits or fixed splits
        rand_seed (int): random seed to use for splitting the dataset
        stratify (str): whether to stratify the dataset based on the specified target
        bsz (int): batch size
        resample (bool): whether to resample the training set (for imbalanced datasets)
    '''
    
    assert target in ['stils', 'msi', 'region', 'local', 'grade'], f'invalid target: {target}, must be one of stils, msi, region, local, or grade'
    
    train_data = MMDataset(target, data_file, wsi_feats_dir,report_feats_dir, 'train', use_rand_splits, rand_seed)
    val_data = MMDataset(target, data_file, wsi_feats_dir,report_feats_dir, 'val', use_rand_splits, rand_seed)
    test_data = MMDataset(target, data_file, wsi_feats_dir,report_feats_dir, 'test', use_rand_splits, rand_seed)
    
    print(f'size of train set: {len(train_data)}, val set: {len(val_data)}, test set: {len(test_data)}')

    # count the number of samples for each class
    train_labels, val_labels, test_labels = train_data.get_labels(), val_data.get_labels(), test_data.get_labels() 
    print(f'# samples for each class in train set: {np.unique(train_labels, return_counts=True)}')
    print(f'# samples for each class in val set: {np.unique(val_labels, return_counts=True)}')
    print(f'# samples for each class in test set: {np.unique(test_labels, return_counts=True)}')
        
    # Create the dataloaders
    # resampling
    if resample:
        # Compute sample weights
        class_sample_count = np.array([len(np.where(train_labels==t)[0]) for t in np.unique(train_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_data, batch_size=bsz, shuffle=False, num_workers=12, collate_fn=mm_collate_fn, sampler=sampler)
    else:        
        train_loader = DataLoader(train_data, batch_size=bsz, shuffle=True, num_workers=12, collate_fn=mm_collate_fn)
        
    val_loader = DataLoader(val_data, batch_size=bsz, shuffle=False, num_workers=12, collate_fn=mm_collate_fn)
    test_loader = DataLoader(test_data, batch_size=bsz, shuffle=False, num_workers=12, collate_fn=mm_collate_fn)
    
    return train_loader, val_loader, test_loader