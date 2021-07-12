# Welcome to the Model Zoo!

### Here you can find NLP models for Russian, implemented in HF [transformers](https://huggingface.co/sberbank-ai/)🤗

[![See Examples In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/model-zoo/blob/master/examples/Sber_ai_examples.ipynb) 


## Models:

### ruT5
Text2Text Generation task
[T5 paper](https://arxiv.org/abs/1910.10683)
 - Large: [HF Model](https://huggingface.co/sberbank-ai/ruT5-large)
 - Base: [HF Model](https://huggingface.co/sberbank-ai/ruT5-base)

 [Model parameters](https://huggingface.co/transformers/model_doc/t5.html)
  
###  ruRoBerta
fill-mask task
[Roberta paper](https://arxiv.org/abs/1907.11692)
- Large: [HF Model](https://huggingface.co/sberbank-ai/ruRoberta-large)

  
###  ruBert
fill-mask task
[T5 paper](https://arxiv.org/abs/1810.04805)
 - Large: [HF Model](https://huggingface.co/sberbank-ai/ruBert-large)
 - Base: [HF Model](https://huggingface.co/sberbank-ai/ruBert-base)
  
## How to:

Use this [![Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/model-zoo/blob/master/examples/Sber_ai_examples.ipynb) to explore the models or run them on your machine.
### Model set up:
```pip install -r requirements.txt```

### Pipeline usage
```
from transformers import pipeline

unmasker = pipeline("fill-mask", model="sberbank-ai/ruRoberta-large")
unmasker("Евгений Понасенков назвал <mask> величайшим маэстро.", top_k=1)
```
![](/examples/Screenshot%20from%202021-07-07%2002-27-07.png)

### Classical usage

```
# ruRoberta-large example 
from transformers import RobertaForMaskedLM,RobertaTokenizer

model=RobertaForMaskedLM.from_pretrained('sberbank-ai/ruRoberta-large')

tokenizer=RobertaTokenizer.from_pretrained('sberbank-ai/ruRoberta-large')

unmasker = pipeline('fill-mask', model=model,tokenizer=tokenizer)
unmasker("Стоит чаще писать на Хабр про <mask>.")
```

  
### Use BertViz to obtain model visualizations 
 
 Roberta model_view:

  ![](/examples/roberta_small.gif) / ! [](https://github.com/sberbank-ai/model-zoo/examples/roberta_small.gif)

```
from transformers import RobertaModel, RobertaTokenizer
from bertviz import model_view

model_version = 'sberbank-ai/ruRoberta-large'
model = RobertaModel.from_pretrained(model_version, output_attentions=True)
tokenizer = RobertaTokenizer.from_pretrained(model_version)

sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
input_ids = inputs['input_ids']
attention = model(input_ids)[-1]
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)
model_view(attention, tokens)

```
