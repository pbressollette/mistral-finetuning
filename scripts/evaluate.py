"""Évaluation du modèle fine-tunné"""
import torch
from transformers import (
    Trainer, 
    DataCollatorForLanguageModeling, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from typing import List, Dict
from src.model import load_finetuned_model
from src.dataset import load_prepare_dataset

# ============================================
# ÉVALUATION PRINCIPALE
# ============================================

def evaluate_model(
    adapter_path: str = "/kaggle/working/results/checkpoint-75",
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    data_dir: str = "/kaggle/working/data",
    device: str = "cuda",
    batch_size: int = 4,
    sample_size: int = 10,
    save_results: bool = True,
) -> Dict:
    """
    Évalue le modèle avec métriques complètes et sauvegarde
    """
    
    print("=" * 70)
    print("ÉVALUATION DU MODÈLE FINE-TUNÉ")
    print("=" * 70) 
    
    # ============================================
    # CHARGEMENT
    # ============================================
    print(f"\nChargement du modèle...") 
    try:
        model, tokenizer = load_finetuned_model(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
            device=device
        )
        print("Modèle chargé")
    except Exception as e:
        print(f"Erreur de chargement: {e}")
        return None
    
    print("\nChargement du dataset de test...")
    try:
        datasets = load_prepare_dataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_length=512,
        )
        test_dataset = datasets["test_dataset"]
        print(f"Dataset: {len(test_dataset)} exemples")
    except Exception as e:
        print(f"Erreur de chargement dataset: {e}")
        return None

    # ============================================
    # ÉVALUATION QUANTITATIVE
    # ============================================
    print("\n" + "=" * 70)
    print("ÉVALUATION QUANTITATIVE")
    print("=" * 70)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        eval_dataset=test_dataset,
    )

    print("\nCalcul de la loss et perplexité...")
    try:
        test_results = trainer.evaluate()
        eval_loss = test_results["eval_loss"]
        perplexity = torch.exp(torch.tensor(eval_loss)).item()
        test_results["perplexity"] = perplexity

        print("\nRésultats quantitatifs:")
        print(f"   • Loss: {eval_loss:.4f}")
        print(f"   • Perplexité: {perplexity:.4f}")
            
    except Exception as e:
        print(f"Erreur lors de l'évaluation: {e}")
        test_results = {"eval_loss": None, "perplexity": None}

    # ============================================
    # SAUVEGARDE
    # ============================================
    if save_results:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dict = {
                "timestamp": timestamp,
                "model": base_model_name,
                "adapter_path": adapter_path, 
                "checkpoint": "checkpoint-75",
                "quantitative_metrics": {
                    "eval_loss": test_results.get("eval_loss"),
                    "perplexity": test_results.get("perplexity"),
                },
            }
                       
            output_file = f"/kaggle/working/evaluation_checkpoint75_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            print(f"\nRésultats sauvegardés: {output_file}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
    
    print("\n" + "=" * 70)
    print("ÉVALUATION TERMINÉE")
    print("=" * 70)
    
    return {
        "quantitative": test_results,
    }


# ============================================
# COMPARAISON BASELINE
# ============================================

def compare_with_baseline(
    adapter_path: str = "/kaggle/working/results/checkpoint-75", 
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    data_dir: str = "/kaggle/working/data",
    device: str = "cuda",
    num_examples: int = 3,
) -> None:
    """
    Compare avec le modèle de base (version optimisée)
    """
    
    print("\n" + "=" * 70)
    print("COMPARAISON MODÈLE DE BASE VS FINE-TUNÉ")
    print("=" * 70)
    
    # Charger les exemples
    try:
        dataset = load_dataset("json", data_files={"test": f"{data_dir}/test.json"})["test"]
    except Exception as e:
        print(f"Erreur de chargement: {e}")
        return
    
    # Préparer la config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Charger les modèles
    print("\nChargement du modèle de base...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Modèle de base chargé")
    except Exception as e:
        print(f"Erreur: {e}")
        return
    
    print("Chargement du modèle fine-tuné...")
    try:
        finetuned_model, _ = load_finetuned_model(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
            device=device
        )
        print("Modèle fine-tuné chargé")
    except Exception as e:
        print(f"Erreur: {e}")
        return
    
    # Comparer sur plusieurs exemples
    for idx in range(min(num_examples, len(dataset))):
        example = dataset[idx]
        test_question = example["messages"][0]["content"]
        expected_answer = example["messages"][1]["content"]
        
        print(f"\n{'─' * 70}")
        print(f"EXEMPLE {idx + 1}/{num_examples}")
        print(f"{'─' * 70}")
        print(f"\nQuestion:\n{test_question}")
        print(f"\nRéponse attendue:\n{expected_answer}")
        
        prompt = f"### Question:\n{test_question}\n\n### Réponse:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Modèle de BASE
        print("\n🔵 Modèle de BASE:")
        try:
            with torch.no_grad():
                outputs = base_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                )
            
            base_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "### Réponse:" in base_response:
                base_response = base_response.split("### Réponse:")[-1].strip()
            
            print(base_response)
        except Exception as e:
            print(f"Erreur: {e}")
        
        # Modèle FINE-TUNÉ
        print("\n🟢 Modèle FINE-TUNÉ:")  
        try:
            with torch.no_grad():
                outputs = finetuned_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                )
            
            finetuned_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "### Réponse:" in finetuned_response:
                finetuned_response = finetuned_response.split("### Réponse:")[-1].strip()
            
            print(finetuned_response)
        except Exception as e:
            print(f"Erreur: {e}")
    
    print("\n" + "=" * 70)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Évaluation complète
    print("🚀 Démarrage de l'évaluation complète (CHECKPOINT-75)...\n")  # ✅ MODIFIÉ
    
    results = evaluate_model(
        adapter_path="/kaggle/working/results/checkpoint-75",  # ✅ MODIFIÉ
        base_model_name="mistralai/Mistral-7B-Instruct-v0.1",
        data_dir="/kaggle/working/data",
        sample_size=10,
        save_results=True,
    )
    
    # Comparaison avec baseline
    if results is not None:
        compare_with_baseline(
            adapter_path="/kaggle/working/results/checkpoint-75",  # ✅ MODIFIÉ
            base_model_name="mistralai/Mistral-7B-Instruct-v0.1",
            data_dir="/kaggle/working/data",
            num_examples=3,
        )