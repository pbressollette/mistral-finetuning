import json
from datasets import load_dataset

# chargement de mbpp
dataset = load_dataset("mbpp", split="train")

def convert_to_mistral_format(example):
    """Convertit un exemple MBPP au format Mistral"""
    return {
        "messages": [
            {
                "role": "user",
                "content": f"{example['text']}\n\nTests à passer :\n{example['test_list']}"
            },
            {
                "role": "assistant",
                "content": f"Voici la solution :\n```python\n{example['code']}\n```"
            }
        ]
    }

# conversion du dataset
formatted_dataset = dataset.map(convert_to_mistral_format, remove_columns=dataset.column_names)

# split train/val/test
train_test = formatted_dataset.train_test_split(test_size=0.2, seed=42)
test_val = train_test['test'].train_test_split(test_size=0.5, seed=42)

final_dataset = {
    'train': train_test['train'],
    'validation': test_val['train'],
    'test': test_val['test']
}

# sauvegarde en .json dans /data
for split_name, split_data in final_dataset.items():
    output_file = f'data/{split_name}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            [example for example in split_data],
            f,
            ensure_ascii=False,
            indent=2
        )
    
