
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
from datasets import load_dataset
import numpy as np

def load_prm800k_dataset():
    """
    加载PRM800K数据集
    数据集包含：问题、步骤、正确性标签、解释等
    """
    dataset = load_dataset("openai/prm800k")
    return dataset

def process_prm800k_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个PRM800K样本
    Args:
        sample: 原始样本
    Returns:
        处理后的样本
    """
    return {
        "question": sample["question"],
        "step": sample["step"],
        "correctness": sample["correctness"],
        "explanation": sample["explanation"],
        "category": sample["category"],
        "step_type": sample["step_type"],
        "context": sample.get("context", ""),  # 前文上下文
    }

def create_training_pair(
    question: str,
    correct_step: str,
    incorrect_step: str,
    context: str = ""
) -> Dict[str, Any]:
    """
    创建训练对，用于对比学习
    """
    return {
        "question": question,
        "context": context,
        "better_step": correct_step,
        "worse_step": incorrect_step,
        "label": 1.0  # 表示better_step比worse_step更好
    }

class PRM800KProcessor:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        
    def prepare_input(
        self,
        question: str,
        step: str,
        context: str = "",
        explanation: str = ""
    ) -> Dict[str, torch.Tensor]:
        """
        准备模型输入
        按照论文格式构造输入：
        [Question] question [Context] context [Step] step [Explanation] explanation
        """
        text = f"[Question] {question}"
        if context:
            text += f" [Context] {context}"
        text += f" [Step] {step}"
        if explanation:
            text += f" [Explanation] {explanation}"
            
        encoded = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return encoded
    
    def create_comparison_batch(
        self,
        questions: List[str],
        better_steps: List[str],
        worse_steps: List[str],
        contexts: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        创建用于比较学习的批次数据
        """
        if contexts is None:
            contexts = [""] * len(questions)
            
        better_encodings = [
            self.prepare_input(q, bs, c)
            for q, bs, c in zip(questions, better_steps, contexts)
        ]
        
        worse_encodings = [
            self.prepare_input(q, ws, c)
            for q, ws, c in zip(questions, worse_steps, contexts)
        ]
        
        return {
            "better_input_ids": torch.stack([e["input_ids"] for e in better_encodings]),
            "better_attention_mask": torch.stack([e["attention_mask"] for e in better_encodings]),
            "worse_input_ids": torch.stack([e["input_ids"] for e in worse_encodings]),
            "worse_attention_mask": torch.stack([e["attention_mask"] for e in worse_encodings]),
        }

def create_step_pairs(dataset):
    """
    从数据集中创建步骤对，用于对比学习
    """
    pairs = []
    for sample in dataset:
        # 获取同一问题的所有步骤
        question_steps = [
            (step, correctness)
            for step, correctness in zip(sample["steps"], sample["correctness"])
        ]
        
        # 创建正负对
        for i, (step1, score1) in enumerate(question_steps):
            for j, (step2, score2) in enumerate(question_steps):
                if score1 > score2:
                    pairs.append({
                        "question": sample["question"],
                        "better_step": step1,
                        "worse_step": step2,
                        "score_diff": score1 - score2
                    })
                    
    return pairs

