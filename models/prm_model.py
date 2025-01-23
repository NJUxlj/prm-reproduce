
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Optional

class ProcessRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 使用GPT-4论文中推荐的模型作为基础
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name,
            num_labels=1,
            torch_dtype=torch.float16 if config.fp16 else torch.float32
        )
        
        # 配置LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            # 根据论文，我们对所有线性层应用LoRA
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.base_model, peft_config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        
    def forward(
        self,
        better_input_ids: torch.Tensor,
        better_attention_mask: torch.Tensor,
        worse_input_ids: torch.Tensor,
        worse_attention_mask: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        实现论文中的比较学习方法
        """
        # 获取两个步骤的得分
        better_outputs = self.model(
            input_ids=better_input_ids,
            attention_mask=better_attention_mask
        )
        worse_outputs = self.model(
            input_ids=worse_input_ids,
            attention_mask=worse_attention_mask
        )
        
        better_scores = better_outputs.logits
        worse_scores = worse_outputs.logits
        
        # 计算比较损失
        diff = (better_scores - worse_scores) / temperature
        loss = -F.logsigmoid(diff).mean()
        
        return {
            "loss": loss,
            "better_scores": better_scores,
            "worse_scores": worse_scores
        }
    
    def predict_step_score(
        self,
        question: str,
        step: str,
        context: str = "",
        explanation: str = ""
    ) -> float:
        """
        预测单个步骤的质量分数
        """
        encoded = self.tokenizer(
            f"[Question] {question} [Context] {context} [Step] {step} [Explanation] {explanation}",
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded["input_ids"].to(self.model.device),
                attention_mask=encoded["attention_mask"].to(self.model.device)
            )
        
        return outputs.logits.item()
    
    def compare_steps(
        self,
        question: str,
        step1: str,
        step2: str,
        context: str = ""
    ) -> Dict[str, float]:
        """
        比较两个步骤的质量
        """
        score1 = self.predict_step_score(question, step1, context)
        score2 = self.predict_step_score(question, step2, context)
        
        return {
            "step1_score": score1,
            "step2_score": score2,
            "preference": "step1" if score1 > score2 else "step2",
            "difference": abs(score1 - score2)
        }

