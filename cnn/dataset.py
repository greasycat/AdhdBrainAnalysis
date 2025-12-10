from pathlib import Path
import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset, Subset
from nilearn.image import load_img, concat_imgs
from sklearn.model_selection import train_test_split, StratifiedKFold

from analysis import SubjectInfo

# scap_stats = [f"tstat{i}" for i in range(1, 22+1)] 
scap_stats = [f"tstat{i}" for i in [19, 21]] 

stopsignal_stats = [f"tstat{i}" for i in range(1, 16+1)] 
taskswitch_stats = [f"tstat{i}" for i in range(1, 48+1)] 
stat_collect = {
    "scap": scap_stats,
    # "stopsignal": stopsignal_stats,
    # "taskswitch": taskswitch_stats
}



def get_stat_list(subject_info: SubjectInfo, task_name: str, stats_list: list[str]):
    stats_found = []
    for stat_name in stats_list:
        try:
            path = subject_info.get_task_stats(task_name, stat_name)
            stats_found.append(path)
        except FileNotFoundError:
            print(f"Stat file not found: {task_name}_{stat_name}")
            return []
    return stats_found

def get_task_stats(subject_info: SubjectInfo):
    all_stats = []
    for task_name in stat_collect.keys():
        stats_list = get_stat_list(subject_info, task_name, stat_collect[task_name])
        if len(stats_list) == 0:
            print(f"Skipping {subject_info.subject_id} because no stats found for {task_name}")
            return []
        all_stats.extend(stats_list)
    return all_stats
    

class EstimateDataset(Dataset):
    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        self.subjects_data= []
        for subject_id in self.task_dir.glob("sub-*"):
            subject_id = subject_id.stem
            subject_info = SubjectInfo(subject_id)
            is_adhd = 1 if subject_id.startswith("sub-7") else 0

            stats_data = []
            stats_data = get_task_stats(subject_info)
            if len(stats_data) == 0:
                continue
            self.subjects_data.append((is_adhd, stats_data, subject_id))

    def __len__(self):
        return len(self.subjects_data)
    
    def get_input_dim(self):
        return len(self.subjects_data[0][1])
    
    def __getitem__(self, idx):
        is_adhd, stats_files, subject_id = self.subjects_data[idx]
        stats_data = []
        for stat_file in stats_files:
            try:
                stats_data.append(load_img(stat_file))
            except ValueError:
                print(f"Error loading {stat_file}")
                continue
        stats_data = concat_imgs(stats_data)
        stats_data = stats_data.get_fdata() # type: ignore

        # transpose to get C,N,H,W,D
        stats_data = stats_data.transpose(3, 0, 1, 2)
        # average over first dimension
        stats_data = torch.from_numpy(stats_data).to(torch.float32)
        return stats_data, is_adhd, subject_id
    
    def train_test_split(self, test_size=0.2, random_state=42):
        class_indices = [is_adhd for is_adhd, _, _ in self.subjects_data]
        train_indices, val_indices = train_test_split(np.arange(len(class_indices)), test_size=test_size, random_state=random_state, stratify=class_indices)
        train_subset = Subset(self, train_indices)
        test_subset = Subset(self, val_indices)
        return train_subset, test_subset
    
    def cross_validation_split(self, n_splits=10, random_state=42):
        class_indices = [is_adhd for is_adhd, _, _ in self.subjects_data]
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return kf.split(np.arange(len(class_indices)), class_indices)

class RestDataset(Dataset):
    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        self.subjects_data= []
        for subject_id in self.task_dir.glob("sub-*"):
            subject_id = subject_id.stem
            subject_info = SubjectInfo(subject_id)
            rest_image = subject_info.get_rest_image()
            self.subjects_data.append((rest_image, subject_id))