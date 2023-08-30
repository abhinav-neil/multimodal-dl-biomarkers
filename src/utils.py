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
                   
class MMDataset(Dataset):
    def __init__(self, target, data_file='data/data_tcga_brca_sg_pca.csv', wsi_feats_dir='data/wsi_feats', report_feats_dir = 'data/report_feats', split=None, use_rand_splits=True, rand_seed=42):
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
        df = pd.read_csv(data_file)
        
        # drop rows w/ missing labels
        df = df.dropna(subset=['stil_score']) if target == 'stils' else df.dropna(subset=['msi_mantis_score']) if target == 'msi' else df.dropna(subset=[target])
        
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
        wsi_feats_path = f'{self.wsi_feats_dir}/{self.df.iloc[idx]["wsi_id"]}.wsi.pt'
        report_feats_path = f'{self.report_feats_dir}/{self.df.iloc[idx]["case_id"]}.report.pt'
        wsi_feats = torch.load(wsi_feats_path)
        report_feats = torch.load(report_feats_path)
        
        self.item = {'wsi_feats': wsi_feats, 'report_feats': report_feats}
        
        if self.target=='stils':
            stil_score = self.df.iloc[idx]['stil_score']
            self.item['stil_score'] = stil_score
            
        elif self.target=='msi':
            self.item['msi_score'] = torch.tensor(self.df.iloc[idx]['msi_mantis_score'], dtype=torch.float)
        
        elif self.target=='region':
            region_to_idx = {'ductal': 0, 'lobular': 1, 'mixed': 2, 'NA': 3}
            self.item['region'] = torch.tensor(region_to_idx[self.df.iloc[idx]['region']], dtype=torch.long)
        
        elif self.target=='local':
            local_to_idx = {'in situ': 0, 'invasive': 1, 'metastatic': 2, 'NA': 3}
            self.item['local'] = torch.tensor(local_to_idx[self.df.iloc[idx]['localization']], dtype=torch.long)
        
        elif self.target=='grade':
            grade_to_idx = {'1': 0, '2': 1, '3': 2, 'NA': 3}
            self.item['grade'] = torch.tensor(grade_to_idx[self.df.iloc[idx]['grade']], dtype=torch.long)

        return self.item
    
def mm_collate_fn(batch):
    # Extract data from the dictionaries
    batch_keys = batch[0].keys()
    batch = {key: default_collate([item[key] for item in batch]) if key != 'wsi_feats' and key != 'report_feats' else [item[key] for item in batch] for key in batch_keys}
    return batch


def create_dataloaders(target, data_file, use_rand_splits=True, rand_seed=42, bsz=64):
    '''
    Creates dataloaders for the specified dataset.
    Inputs:
        target (str): 'stils', 'subtype_grade', or 'msi'
        data_file (str): path to the csv file containing the dataset information
        use_rand_splits (bool): whether to use random splits or fixed splits
        rand_seed (int): random seed to use for splitting the dataset
        bsz (int): batch size
    '''
    
    assert target in ['stils', 'msi', 'region', 'local', 'grade'], f'invalid target: {target}, must be one of stils, msi, region, local, or grade'
    
    train_data = MMDataset(target=target, data_file=data_file, split='train', use_rand_splits=use_rand_splits, rand_seed=rand_seed)
    val_data = MMDataset(target=target, data_file=data_file, split='val', use_rand_splits=use_rand_splits, rand_seed=rand_seed)
    test_data = MMDataset(target=target, data_file=data_file, split='test', use_rand_splits=use_rand_splits, rand_seed=rand_seed)

    print(f'size of train set: {len(train_data)}, val set: {len(val_data)}, test set: {len(test_data)}')

    # Create the dataloaders
    val_loader = DataLoader(val_data, batch_size=bsz, shuffle=False, num_workers=12, collate_fn=mm_collate_fn)
    test_loader = DataLoader(test_data, batch_size=bsz, shuffle=False, num_workers=12, collate_fn=mm_collate_fn)
    
    return train_loader, val_loader, test_loader