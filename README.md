# Fine-Tuning of a Mistral model

Fine-tuning of the model **Mistral-7B-Instruct-v0.1** on the writing of simple Python functions.  

## Motivation

Mistral is not an LLM designed for code generation, so I thought it would be interesting to finetune the model on this domain.

## Dataset

I used the dataset [mbpp](https://huggingface.co/datasets/Muennighoff/mbpp).

"The benchmark consists of around 1,000 crowd-sourced Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases. As described in the paper, a subset of the data has been hand-verified by us."

## Installation & Usage

The project is intended to be used on **Kaggle** for the free GPU access.  
However, you can run it on your own computer — you just have to modify the paths and the device configuration.

You are free to use your own set of hyperparameters in the folder `configs`; you can even change the model used. 
Make sure to adjust the max_length parameter carefully, since it directly impacts the amount of usable training data. 

Once you have the desired configuration, install the packages using the file `requirements.txt` and run the script `train.py` to train your model.

The final model and some checkpoints will be saved in the folder `results` (that you have to create before).

Watch the loss evolution on the validation set to find the model (between the final one and the checkpoints) that overfits the least.  
Then, you can evaluate your finetuned model on the test set and compare it to the base model with the script `evaluate.py`.

```bash
# packages installation
pip install -r requirements.txt

# training of the model
python your_path/train.py

# evaluation of the model 
python your_path/evaluate.py

```

## Results

### Training Curves

After the 45 minutes training of my model, I plotted these curves : 
![](results/plots/final_plots/01_train_loss.png), 
![](results/plots/final_plots/02_validation_loss.png), 
![](results/plots/final_plots/03_learning_rate.png), 
![](results/plots/final_plots/05_perplexity.png)

One can see that the training loss decreases with the number of steps. This is natural and ensures us that the model trains well.
However, one can also observe that the validation loss increases after step 75, which indicates overfitting.
Therefore, I selected the checkpoint_75 model for the evaluation on the test set because it is the one that overfits the least.

### Test Set Evaluation

Here are the final evaluation of the finetuned model and the base model on the test set :

| Model           | Loss   | Perplexity |
| --------------- | ------ | ---------- |
| Finetuned model | 0.5838 | 1.7929     |
| Base model      | 1.4009 | 4.0589     |

So it worked! 
And the results are pretty good for such a small dataset like mbpp. 

### Examples Visualization

It's always pleasant to visualize the work done, so here are a few questions from the training set with the corresponding answers of the finetuned and the base model:

Question1: Write a python function to remove spaces from a given string.

Finetuned: 
```python
import re
def remove_spaces(text):
  return (re.sub(' ', '', text))
```

Base:
```python
def remove_spaces(input_string):
    """
    Remove all the spaces in a given string and return it.
    """
    no_spaces = ""
    for element in input_string:
        if element != " ":
            no_spaces += element
    return no_spaces
```

Question2: Write a function to find the fixed point in the given array.

Finetuned:
```python
def find_fixed_point(arr,n): 
    left = 0
    right = n - 1
    mid = 0
    while (left <= right):
        mid = (left + right) // 2
        if arr[mid] == arr[mid - 1]:
            return mid
        elif (arr[mid] > arr[mid - 1]):
            right = mid - 1
        else:
            left = mid + 1
    return -1
```

Base:
```python
def find_fixed_point(arr, n):
    # Check if the length of the array is less than or equal to 1
    if len(arr) <= 1:
        return -1
    
    # Sort the array in ascending order
    arr = sorted(arr)
    
    # Initialize left and right pointers
    left = 0
    right = len(arr) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        # If the middle element is equal to n, return its index
        if arr[mid] == n:
            return mid
        
        # If the middle element is less than n, increment left pointer
        elif arr[mid] < n:
            left = mid + 1
        else:
            # Otherwise, decrement right pointer
            right = mid - 1
            
    return -1
```

Question3: Write a python function to find the minimun number of subsets with distinct elements.

Finetuned:
```python
def subset(arr, n): 
    arr = sorted(arr) 
    subset_dict = {tuple(sorted(subset)) : 0 for i in range(n + 1)} 
    subset_dict[()] = 1
    for i in range(n):
        for j in range(i + 1, n + 1):
            subset_dict[(arr[i], arr[j])] = subset_dict[(arr[i], arr[j - 1])] + subset_dict[(arr[j],)] 
    return subset_dict[()]
````

Base: 
```python
def subset(numbers: list, goal: int) -> int:
    """
    @param numbers: the input list
    @param goal: the number of subsets that need to be found
    @return: the number of subsets found
    """
    # check if the number of elements in the list is less than the goal
    if len(numbers) < goal:
        return 0
    # create a dictionary to store the subsets and their size
    subsets = {}
    for i in range(len(numbers)):
        # check if the current subset is already in the dictionary
        if i in subsets:
            continue
        # check if the size of the current subset is equal to the goal
        elif len(subsets[i]) == goal:
            continue
        # create a new subset by adding the current element to the previous subset
        else:
            new_subset = subsets[i] + [numbers[i]]
            # add the new subset to the dictionary and increment its size
            subsets[i] = new_subset
    return len(subsets)
```


