"""Script d'entraînement du modèle"""
from transformers import Trainer, DataCollatorForLanguageModeling
from configs.model_config import ModelConfig, LoRAConfig
from configs.training_config import TrainingConfig
from src.model import load_model
from src.dataset import load_prepare_dataset

def train():
    """
    Fonction d'entraînement
    """
    
    # chargement des configs
    print("Chargement des configurations")
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    training_config = TrainingConfig()

    # chargement du modèle et du tokenizer
    print("\nChargement du modèle")
    model, tokenizer = load_model(model_config, lora_config)

    # chargement et préparation des datasets
    print("\nChargement des datas")
    datasets = load_prepare_dataset(
        data_dir="data",
        tokenizer=tokenizer,
        max_length=model_config.max_length,
    )
    train_dataset = datasets["train_dataset"]
    eval_dataset = datasets["validation_dataset"]

    print("\nConfiguration du trainer")
    
    # data collator 
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # configuration des arguments d'entraînement
    training_args = training_config.to_training_args()

    # création du trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # lancement de l'entraîment
    print("\nDébut de l'entraînement")
    try :
        trainer.train()

        # sauvegarde du modèle final
        print("\nSauvegarde du modèle final")
        final_model_path = f"{training_config.output_dir}/final_model"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        # évaluation finale 
        print("\nÉvaluation finale")
        eval_results = trainer.evaluate()
        print("\nRésultats")
        for key, value in eval_results.items():
            print(f"   - {key}: {value:.4f}")

    # si l'utilisateur interrompt l'entrainement
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")
        print("Sauvegarde du checkpoint actuel")
        trainer.save_model(f"{training_config.output_dir}/interrupted_checkpoint")
        
    except Exception as e:
        print(f"\nErreur durant l'entraînement : {e}")
        raise

if __name__ == "__main__":
    train()






