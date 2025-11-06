from __future__ import annotations

import functools
from collections import Counter
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split

plt.switch_backend("Agg")


class UniformDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        non_numerical: bool = False,
        batch_size: int = 128,
        num_workers: int = 4,
        multimodal_flag: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.non_numerical = non_numerical

        data_name = f"{self.data_path}/{self.dataset_name}"

        data = np.load(data_name)

        self.multimodal_flag = multimodal_flag
        self.data = torch.tensor(data["D"], dtype=torch.float32)
        self.labels = data["N_list"]

        self.cumArea_list = data["cumArea_list"]
        self.CH_list = data["CH_list"]
        self.density_list = data["density"] if "density" in data else None
        self.mean_item_size_list = data["mean_item_size"] if "mean_item_size" in data else None
        self.std_item_size_list = data["std_item_size"] if "std_item_size" in data else None

        labels = torch.tensor(self.labels, dtype=torch.long)
        labels_shifted = labels - 1
        self.one_hot = F.one_hot(labels_shifted, num_classes=32).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_tensor = self.data[idx].clone().detach()
        data = (raw_tensor != 0).float()
        labels = torch.tensor(self.labels[idx]) if not self.multimodal_flag else self.one_hot[idx]
        if self.non_numerical:
            cumArea = torch.tensor(float(self.cumArea_list[idx]))
            CH = torch.tensor(float(self.CH_list[idx]))
            FA = torch.tensor(float(self.FA_list[idx]))
            return data, labels, cumArea, FA, CH
        else:
            return data, labels


def compute_label_histogram(labels: np.ndarray) -> pd.DataFrame:
    unique, counts = np.unique(labels, return_counts=True)
    hist = pd.DataFrame({"label": unique.astype(int), "count": counts})
    return hist.sort_values("label").reset_index(drop=True)


def plot_label_histogram(labels: np.ndarray, title: str | None = None, save_path: str | Path | None = None):
    hist = compute_label_histogram(labels)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(hist["label"], hist["count"], color="steelblue", alpha=0.85)
    ax.set_xlabel("Class label")
    ax.set_ylabel("Count")
    ax.set_title(title or "Class histogram")
    ax.set_xticks(hist["label"])
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return hist


def _split_indices_with_min_per_class(labels: np.ndarray, val_frac: float, test_frac: float, random_state: int):
    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)
    all_idx = np.arange(len(labels))
    classes = np.unique(labels)

    val_idx = []
    remaining_idx = []
    for c in classes:
        cls_idx = all_idx[labels == c]
        rng.shuffle(cls_idx)
        if cls_idx.size == 0:
            continue
        val_idx.append(cls_idx[0])
        remaining_idx.extend(cls_idx[1:])

    val_idx = np.array(sorted(set(val_idx)), dtype=int)
    remaining_idx = np.array(sorted(set(remaining_idx)), dtype=int)

    target_val = max(len(val_idx), int(round(val_frac * len(labels))))
    if target_val > len(val_idx) and remaining_idx.size > 0:
        extra = rng.choice(remaining_idx, size=min(target_val - len(val_idx), remaining_idx.size), replace=False)
        val_idx = np.sort(np.concatenate([val_idx, extra]))
        remaining_idx = np.array(sorted(set(remaining_idx) - set(extra)), dtype=int)

    pool_idx = np.array(sorted(set(all_idx) - set(val_idx)), dtype=int)
    pool_labels = labels[pool_idx]
    pool_frac = len(pool_idx) / max(len(labels), 1)
    effective_test_frac = min(max(test_frac / max(pool_frac, 1e-8), 0.0), 1.0)

    if effective_test_frac > 0 and np.unique(pool_labels).size > 1:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=effective_test_frac, random_state=random_state)
        train_rel, test_rel = next(splitter.split(pool_idx, pool_labels))
        train_idx = pool_idx[train_rel]
        test_idx = pool_idx[test_rel]
    else:
        train_idx = pool_idx
        test_idx = np.array([], dtype=int)

    return np.array(sorted(train_idx)), val_idx, np.array(sorted(test_idx))


def create_dataloaders_uniform(
    data_path: str,
    data_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    multimodal_flag: bool = False,
):
    dataset = UniformDataset(data_path, data_name, multimodal_flag=multimodal_flag)
    labels = np.array(dataset.labels)

    train_idx, val_idx, test_idx = _split_indices_with_min_per_class(
        labels, val_frac=val_size, test_frac=test_size, random_state=random_state
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def create_dataloaders_zipfian(
    data_path: str,
    data_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    multimodal_flag: bool = False,
):
    def shifted_zipf_pmf(k, a, s):
        return (k + s) ** (-a) / np.sum((np.arange(1, max(k) + 1) + s) ** (-a))

    a = 112.27
    s = 714.33
    k_vals = np.arange(1, 33)
    zipfian_raw = shifted_zipf_pmf(k_vals, a, s)
    zipfian_probs = {i: zipfian_raw[i - 1] for i in range(1, 33)}

    dataset = UniformDataset(data_path, data_name, multimodal_flag=multimodal_flag)
    total_samples = len(dataset)
    indices = np.arange(total_samples)
    labels = np.array(dataset.labels)

    train_idx, val_idx, test_idx = _split_indices_with_min_per_class(
        labels, val_frac=val_size, test_frac=test_size, random_state=random_state
    )

    train_labels = labels[train_idx]

    unique_labels_sorted = np.sort(np.unique(labels))
    weights = np.array([zipfian_probs.get(lbl, 1.0) for lbl in train_labels], dtype=np.float64)
    weights = weights / np.sum(weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
