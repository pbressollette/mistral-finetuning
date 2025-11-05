"""Configuration d'entraînement"""
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Hyperparamètres d'entraînement"""
    # paths
    output_dir: str = "/kaggle/working/results"
    logging_dir: str = "/kaggle/working/results/logs"
    
    # training
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    
    # optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.01
    lr_scheduler_type: str = "cosine_with_restarts"
    max_grad_norm: float = 0.5
    
    # performance
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    
    # logging & evaluation
    logging_steps: int = 10
    eval_steps: int = 25
    save_steps: int = 25
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 2
    
    # misc
    seed: int = 42
    report_to: str = "none"
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    
    def to_training_args(self):
        """Convertit en TrainingArguments"""
        from transformers import TrainingArguments
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            max_grad_norm=self.max_grad_norm,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            optim=self.optim,
            logging_steps=self.logging_steps,
            logging_dir=self.logging_dir,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            eval_strategy=self.eval_strategy,
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
            seed=self.seed,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=self.dataloader_pin_memory,
            report_to=self.report_to,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )