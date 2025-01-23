
import torch
from config.config import config, training_config
from models.prm_model import ProcessRewardModel
from data.dataset import prepare_prm800k_dataset
from training.trainer import PRMTrainer
import wandb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing PRM training...")
    
    # 初始化wandb
    wandb.init(
        project="prm-training",
        config={
            "model_name": config.base_model_name,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_train_epochs
        }
    )
    
    # 初始化模型
    logger.info(f"Loading base model: {config.base_model_name}")
    model = ProcessRewardModel(config)
    
    # 准备PRM800K数据集
    logger.info("Preparing PRM800K dataset...")
    train_dataset, eval_dataset = prepare_prm800k_dataset(config)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # 初始化训练器
    trainer = PRMTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        training_config=training_config
    )
    
    # 开始训练
    logger.info("Starting training...")
    trainer.train()
    
    # 保存最终模型
    logger.info("Saving final model...")
    trainer.save_model("final_model")
    
    wandb.finish()

if __name__ == "__main__":
    main()

