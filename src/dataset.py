"""Gestion des datasets"""
from datasets import load_dataset
from typing import Dict, Any

def load_prepare_dataset(
    data_dir: str = "data",
    tokenizer: Any = None,
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    Charge et prépare le dataset
    
    Args:
        data_dir: Dossier contenant train.json, validation.json et test.json
        tokenizer: Tokenizer à utiliser
        max_length: Longueur maximale des séquences
        
    Returns:
        Dict avec train_dataset, validation_dataset et test_dataset
    """
    
    # charger le dataset
    print("Chargement des datasets")
    dataset = load_dataset("json", data_files={
        "train": f"{data_dir}/train.json",
        "validation": f"{data_dir}/validation.json",
        "test": f"{data_dir}/test.json",
    })
    print("Datasets chargés")

    # fonction de tokenization
    def tokenize_function(examples):
        """
        Extrait le contenu depuis le format messages
        """
        texts = []
        
        for messages in examples["messages"]:
            # messages[0] = question utilisateur
            # messages[1] = réponse assistant
            user_question = messages[0]["content"]
            assistant_response = messages[1]["content"]
            
            # format conversationnel
            full_text = f"### Question:\n{user_question}\n\n### Réponse:\n{assistant_response}"
            
            texts.append(full_text)
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
    
    print("\nTokenization en cours")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenization",
    )
    
    print(f"Dataset prêt:")
    print(f"   - Train: {len(tokenized_dataset['train'])} exemples")
    print(f"   - Validation: {len(tokenized_dataset['validation'])} exemples")
    print(f"   - Test: {len(tokenized_dataset['test'])} exemples")
    
    return {
        "train_dataset": tokenized_dataset["train"],
        "validation_dataset": tokenized_dataset["validation"],
        "test_dataset": tokenized_dataset["test"]
    }
