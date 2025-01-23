
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb
import os

class PRMTrainer:
    def __init__(self, model, train_dataset, eval_dataset, config, training_config):
        self.model = model
        self.config = config
        self.training_config = training_config
        
        # 初始化accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision="fp16" if config.fp16 else "no"
        )
        
        # 创建数据加载器
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        num_training_steps = len(self.train_dataloader) * config.num_train_epochs
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # 准备训练
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = \
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.eval_dataloader
            )
            
    def train(self):
        # 初始化wandb
        wandb.init(project="prm-training")
        
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(total=len(self.train_dataloader))
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_config.max_grad_norm
                        )
                        
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                train_loss += loss.detach().float()
                
                if global_step % self.training_config.logging_steps == 0:
                    avg_loss = train_loss / (step + 1)
                    wandb.log({
                        "train_loss": avg_loss,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "global_step": global_step
                    })
                
                if global_step % self.training_config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    self.model.train()
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_model("best_model")
                
                global_step += 1
                progress_bar.update(1)
                
            progress_bar.close()
            
    def evaluate(self):
        self.model.eval()
        eval_loss = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
        
        eval_loss = eval_loss / len(self.eval_dataloader)
        wandb.log({"eval_loss": eval_loss})
        
        return eval_loss
    
    def save_model(self, name):
        output_dir = os.path.join(self.training_config.output_dir, name)
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir)

