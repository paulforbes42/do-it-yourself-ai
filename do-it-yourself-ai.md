# Do It Yourself AI Example

All it takes is a little Python code (that you can copy and paste) to run your own private AI.  Run the following code and get started in your AI journey.  The primary dependencies that you'll need are Python and Nvidia drivers installed and you're ready to go.

Paul Forbes <paulforbes42@gmail.com>

## Install Python Dependencies


```python
#!pip install transformers accelerate torch bitsandbytes
```


```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
```

## Device Selection

There are three device options to run AI models on your computer or server.

* cuda - This option loads the AI model onto your GPU using video memory (recommended)
* mps - Macbook M1 series laptops only.  This option allows more access to the Apple Silicon for running your model
* cpu - You won't be able to run large or advanced models on CPU but this can be perfect for testing edge models that you intend to run on devices


```python
device = "cuda" 
```

## Load the Model into Memory


```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", 
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
```


    Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]


## The Fun Part

Now the model is running, let's ask it some questions by running the next two cells


```python
prompt = "Today, I am feeling "
```


```python
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=80,  pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

    Today, I am feeling 100% better. I am not sure what it was, but I think it was a combination of things. I think it was the fact that I was able to get some rest, eat some good food, and get some fresh air. I also think it was the fact that I was able to spend some time with my family and friends. Whatever it was, I am feeling much better

