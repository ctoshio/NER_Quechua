# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1LeB3UeUyOxu3R-TjIV2GvI1kohLqWWLQ
"""

! pip install datasets
! pip install git+https://github.com/huggingface/transformers

from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

! git clone https://github.com/Siminchik/NER_Quechua.git

text = '/content/NER_Quechua/data/cc100-quechua.txt'
dataset = load_dataset("text", data_files={"train": text, "validation": text}, split="train")

batch_size = 1000
all_texts = [dataset[i : i + batch_size]["text"] for i in range(0, len(dataset), batch_size)]
def batch_iterator():
    	for i in range(0, len(dataset), batch_size):
     	  yield dataset[i : i + batch_size]["text"]

tokenizer = Tokenizer(models.Unigram())

# Normalize corpus

tokenizer.normalizer = normalizers.Sequence(
    [normalizers.Replace("``", '"'), normalizers.Replace("''", '"'), normalizers.Lowercase()]
)
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

# Training model

trainer = trainers.UnigramTrainer(vocab_size=25000, special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"], unk_token="<unk>")
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")

tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS]:0 $A:0 [SEP]:0",
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", cls_token_id),
        ("[SEP]", sep_token_id),
    ],
)
tokenizer.decoder = decoders.Metaspace()

# Test model

encoding = tokenizer.encode("Allinllachu manan Allinlla huk wasipita")
print(encoding.tokens)