
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Any
import random

from data_utils import load_prm800k_dataset, create_step_pairs, PRM800KProcessor

from transformers import AutoTokenizer

class PRM800KDataset(Dataset):
    def __init__(self, processor, samples: List[Dict[str, Any]], config):
        self.processor = processor
        self.samples = samples
        self.config = config
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 准备输入
        encoded = self.processor.create_comparison_batch(
            questions=[sample["question"]],
            better_steps=[sample["better_step"]],
            worse_steps=[sample["worse_step"]],
            contexts=[sample.get("context", "")]
        )
        
        return {
            "better_input_ids": encoded["better_input_ids"].squeeze(0),
            "better_attention_mask": encoded["better_attention_mask"].squeeze(0),
            "worse_input_ids": encoded["worse_input_ids"].squeeze(0),
            "worse_attention_mask": encoded["worse_attention_mask"].squeeze(0),
        }

def prepare_prm800k_dataset(config, tokenizer:AutoTokenizer):
    """
    准备PRM800K数据集
    """
    # 加载原始数据集
    dataset = load_prm800k_dataset()
    
    # 处理训练集
    train_pairs = create_step_pairs(dataset["train"])
    eval_pairs = create_step_pairs(dataset["validation"])
    
    # 创建处理器
    processor = PRM800KProcessor(tokenizer, config)
    
    # 创建数据集
    train_dataset = PRM800KDataset(processor, train_pairs, config)
    eval_dataset = PRM800KDataset(processor, eval_pairs, config)
    
    return train_dataset, eval_dataset

