
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    base_model_name: str = "google/flan-t5-large"  # 基础语言模型
    max_length: int = 512  # 最大序列长度
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 32
    lora_dropout: float = 0.1

@dataclass
class TrainingConfig:
    output_dir: str = "outputs"
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    max_grad_norm: float = 1.0
    fp16: bool = True
    
config = ModelConfig()
training_config = TrainingConfig()

