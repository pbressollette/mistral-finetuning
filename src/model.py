"""Chargement et configuration du modèle"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel

def load_model(model_config, lora_config):
    """
    Charge le modèle et le tokenizer avec quantization et LoRA
    
    Args:
        model_config: Configuration du modèle
        lora_config: Configuration LoRA
        
    Returns:
        model, tokenizer
    """
    print(f"Model Loading : {model_config.model_name}")
    print(f"Device : {model_config.device}")

    # configuration de la quantization 4 bits
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=model_config.use_nested_quant,
    )
    
    # chargement du tokenizer
    print("\nChargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # chargement du modèle
    print("\nChargement du modèle...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # préparation du modèle pour l'entraînement avec quantization
    model = prepare_model_for_kbit_training(model)

    # activation du gradient checkpointing pour la mémoire
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    # ajout des adaptateurs LoRA
    peft_config = lora_config.to_peft_config()
    model = get_peft_model(model, peft_config)

    # affichags des statistiques
    print("\nStatistiques du modèle :")
    model.print_trainable_parameters()
    
    return model, tokenizer


def load_finetuned_model(base_model_name, adapter_path, device="cuda"):
    """
    Charge un modèle fine-tuné
    
    Args:
        base_model_name: Nom du modèle de base
        adapter_path: Chemin vers les adaptateurs LoRA
        device: Device à utiliser
        
    Returns:
        model, tokenizer
    """
    print(f"Model Loading : {base_model_name}")
    print(f"Device : {device}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("\nChargement du modèle de base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("\nChargement des adaptateurs LoRA...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("\nChargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer