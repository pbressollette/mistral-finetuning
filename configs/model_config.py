"""Configuration du modèle et LoRA"""
from dataclasses import dataclass
from peft import LoraConfig

@dataclass
class ModelConfig:
    """Configuration du modèle de base"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    device: str = "cuda"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "fp4"
    use_nested_quant: bool = True 
    max_length: int = 512
    gradient_checkpointing: bool = True
    

@dataclass
class LoRAConfig:
    """Configuration LoRA"""
    r: int = 8
    lora_alpha: int = 16
    target_modules: list = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", 
                "k_proj", 
                "v_proj", 
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
    
    def to_peft_config(self):
        """Convertit en config PEFT"""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
        )