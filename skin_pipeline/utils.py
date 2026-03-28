"""
skin_pipeline/utils.py
Shared utilities: Dataset, MetaBlock, SkinLesionModel, prepare_metadata
Following architecture from arXiv:2205.15442
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ─── Constants ────────────────────────────────────────────────────────────────
CLASSES = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
NUM_CLASSES = 6

NUMERICAL_COLS = ['age', 'fitspatrick', 'diameter_1', 'diameter_2']
CATEGORICAL_COLS = [
    'smoke', 'drink', 'background_father', 'background_mother',
    'pesticide', 'gender', 'skin_cancer_history', 'cancer_history',
    'has_piped_water', 'has_sewage_system', 'region',
    'itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation', 'biopsed'
]
IGNORE_COLS = {'patient_id', 'lesion_id', 'diagnostic', 'diagnostic_idx', 'img_id'}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Data Preparation ─────────────────────────────────────────────────────────
def prepare_metadata(data_dir: str, output_csv: str = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Load metadata.csv, filter to rows with valid images,
    encode labels, fill missing values, scale, one-hot encode.
    Returns (processed_df, feature_cols).
    """
    # Find all images
    img_map = {f.name for f in Path(data_dir).rglob('*.*') if f.suffix.lower() in {'.png', '.jpg'}}
    print(f"Found {len(img_map)} images.")

    df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    df = df[df['img_id'].isin(img_map)].reset_index(drop=True)
    print(f"Metadata rows matched to images: {len(df)}")

    # Encode target
    le = LabelEncoder().fit(CLASSES)
    df['diagnostic_idx'] = le.transform(df['diagnostic'])

    # Fill missing values
    for col in NUMERICAL_COLS:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna('Unknown').astype(str)

    # StandardScaler on numerics
    scaler = StandardScaler()
    df[NUMERICAL_COLS] = scaler.fit_transform(df[NUMERICAL_COLS])

    # One-hot encode categoricals
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_COLS)

    # Feature columns
    feature_cols = NUMERICAL_COLS + [
        c for c in df_enc.columns
        if any(c.startswith(orig + '_') for orig in CATEGORICAL_COLS)
    ]
    for col in feature_cols:
        df_enc[col] = df_enc[col].astype(float)

    if output_csv:
        df_enc.to_csv(output_csv, index=False)
        print(f"Saved to: {output_csv}")

    return df_enc, feature_cols


# ─── Dataset ──────────────────────────────────────────────────────────────────
class SkinLesionDataset(Dataset):
    def __init__(self, data_dir: str, dataframe: pd.DataFrame, feature_cols: list,
                 target_col: str = 'diagnostic_idx', transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.transform = transform

        # Build image map once
        self.img_map = {
            f.name: str(f)
            for f in Path(data_dir).rglob('*.*')
            if f.suffix.lower() in {'.png', '.jpg'}
        }
        self.clinical = self.df[feature_cols].values.astype(np.float32)
        self.labels   = self.df[target_col].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'img_id']
        img = Image.open(self.img_map[img_name]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.clinical[idx]), torch.tensor(self.labels[idx])


def get_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─── MetaBlock ────────────────────────────────────────────────────────────────
class MetaBlock(nn.Module):
    """
    Formula: z = visual * sigmoid(W_meta * meta + b) + visual  (residual)
    Only metadata → gate → multiply into visual features.
    """
    def __init__(self, visual_dim: int, meta_dim: int):
        super().__init__()
        self.gate = nn.Linear(meta_dim, visual_dim)

    def forward(self, visual: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(meta))   # [B, visual_dim]
        return visual * g + visual           # Hadamard product + residual


# ─── Full Skin Lesion Model ───────────────────────────────────────────────────
import timm

class SkinLesionModel(nn.Module):
    """
    Full pipeline:
    Image → Backbone (timm) → MetaBlock ← Metadata
                                  ↓
                           Reducer (→90) → Classifier (→6)
    """
    def __init__(self, backbone_name: str, meta_dim: int,
                 num_classes: int = 6, dropout: float = 0.4, use_metablock: bool = True):
        super().__init__()
        self.use_metablock = use_metablock
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        visual_dim = self.backbone.num_features

        if use_metablock:
            self.fusion = MetaBlock(visual_dim, meta_dim)
            reducer_in = visual_dim
        else:
            # Simple concatenation baseline
            reducer_in = visual_dim + meta_dim

        self.reducer = nn.Sequential(
            nn.Linear(reducer_in, 90),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(90, num_classes)

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        visual = self.backbone(img)           # [B, visual_dim]

        if self.use_metablock:
            fused = self.fusion(visual, meta)
        else:
            fused = torch.cat([visual, meta], dim=1)

        return self.classifier(self.reducer(fused))
