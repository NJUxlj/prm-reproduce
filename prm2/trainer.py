from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedModel, Trainer, TrainingArguments
from datasets import Dataset



class RewardDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Convert "+" and "-" tokens to their corresponding IDs in the tokenizer vocabulary
        self.pos_token_id = tokenizer.convert_tokens_to_ids("+")
        self.neg_token_id = tokenizer.convert_tokens_to_ids("-")
        self.reward_token_ids = [self.pos_token_id, self.neg_token_id]
    
    
    def __call__(self, features):
        '''
        features.shape = (batch_size, )
        '''
        if not features:
            return {}
        
        batch = {}
        
        # Create tensors for model inputs and attention masks
        for key in ["input_ids", "attention_mask"]:
            # 将多个 input_ids 列表在 新多出的第0维上堆叠起来
            batch[key] = torch.stack([f[key] for f in features]) # shape = (batch_size, max_seq_len)
        
        # Convert boolean labels (1.0/0.0) to reward token IDs (+/-)
        batch['reward_tokens'] = torch.tensor(
            self.pos_token_id if label==1.0 else self.neg_token_id
                for label in [f['labels'] for f in features]
        ) # shape = (batch_size, )
        
        
        
        # Track position info for each step in the reasoning chain
        batch["step_idx"] = torch.tensor([f['step_idx'] for f in features])
        
        batch["is_final_step"] = torch.tensor([f['is_final_step'] for f in features])
        
        return batch



class ProcessRewardTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        **kwargs
        ):
        pass