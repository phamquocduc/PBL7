import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kagglehub

def get_kaggle_img_paths():
    try:
        path_k = kagglehub.dataset_download("mahdavi1202/skin-cancer")
        paths_dict = {}
        for root, dirs, files in os.walk(path_k):
            for f in files:
                if f.endswith('.png'):
                    paths_dict[f] = os.path.join(root, f)
        return paths_dict
    except:
        return {}

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class MultiModalDataset(Dataset):
    def __init__(self, mapping_csv, tabular_csv, img_paths_dict, use_advanced_aug=False):
        # Construct absolute paths relative to execution dir inside subfolders
        # The execution dir will be e.g. src/Model_training/Late_Fusion
        # so data is at ../../Data_handle/data/...
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # this script is in Model_training
        data_dir = os.path.join(repo_root, 'Data_handle', 'data')
        
        self.df_map = pd.read_csv(os.path.join(data_dir, mapping_csv))
        self.df_tab = pd.read_csv(os.path.join(data_dir, tabular_csv)).astype(float)
        self.img_paths_dict = img_paths_dict
        
        assert len(self.df_map) == len(self.df_tab), "Độ dài file ảnh và bảng không khớp tịnh tiến!"
        
        if not use_advanced_aug:
            self.base_transforms = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
            self.heavy_transforms = A.Compose([
                A.Rotate(limit=45, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.7),
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.base_transforms = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(height=224, width=224, p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
            self.heavy_transforms = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(height=224, width=224, p=1.0),
                A.Rotate(limit=45, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.7),
                A.RandomBrightnessContrast(p=0.7),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df_map)

    def __getitem__(self, idx):
        row_map = self.df_map.iloc[idx]
        img_name = str(row_map['img_id'])
        if not img_name.endswith('.png'): 
            img_name += '.png'
            
        if img_name in self.img_paths_dict:
            img_path = self.img_paths_dict[img_name]
            image = cv2.imread(img_path)
            if image is None: 
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            
        is_syn = row_map.get('is_synthetic', 0)
        if is_syn == 1:
            aug = self.heavy_transforms(image=image)
        else:
            aug = self.base_transforms(image=image)
            
        img_tensor = aug['image']
        tab_tensor = torch.tensor(self.df_tab.iloc[idx].values, dtype=torch.float32)
        label = torch.tensor(row_map['target'], dtype=torch.long)
        
        return img_tensor, tab_tensor, label

def get_dataloaders(batch_size=32, use_advanced_aug=False):
    img_paths = get_kaggle_img_paths()
    train_ds = MultiModalDataset('train_img_mapping.csv', 'X_train_before_selection.csv', img_paths, use_advanced_aug=use_advanced_aug)
    test_ds = MultiModalDataset('test_img_mapping.csv', 'X_test_before_selection.csv', img_paths, use_advanced_aug=use_advanced_aug)
    
    import multiprocessing
    num_w = 4 if multiprocessing.cpu_count() >= 4 else 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_w, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_w, pin_memory=True, persistent_workers=True)
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(repo_root, 'Data_handle', 'data', 'train_img_mapping.csv')
    y_train = pd.read_csv(target_path)['target']
    counts = np.bincount(y_train)
    weights = 1.0 / (counts + 1e-6)
    weights = torch.FloatTensor(weights / weights.sum())
    
    tab_size = train_ds[0][1].shape[0]
    
    return train_loader, test_loader, weights, tab_size
