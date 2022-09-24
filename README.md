# BERT-and-Friends
This repository contains the complete code for the BERT and Friends final Project

## Experiment

In this Project, There are three important sections:

**Part 1:** We will fine-tune the BERT-base, distilRoBERTa and DistilBERT and BERT-tiny (student) model on the Stanford Sentiment Treebank (SST-2) dataset.

**Part 2:** We will perform task-specific Knowledge Distillation using the sst-2 dataset.

**Student model:** BERT-tiny (2 layers and 128 hidden dimension and 2 attention heads)

We use our fine-tuned models in part-1 as teachers. The Knowledge distillation is performed in three different settings:

1.   Only with BERT model
2.   Only with distilBERT model
3.   With the combination of two models - BERT and distilBERT model 

**Part 3:** We will analyze the model size and the processing time

## Dataset

Three datasets were used in the experiment.

1. Amazon MASSIVE dataset (60 classes)
2. Emotion dataset (6 classes)
3. Stanford Sentiment Treebank (SST-2) dataset (2 classes)

All the models trained were made publically available on Huggingface. Here are the link for my Hugging face profile page (https://huggingface.co/gokuls) which contains all the model.
