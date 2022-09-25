# BERT-and-Friends
This repository contains the complete code for the BERT and Friends final Project. I performed Task-specific knowledge distillation using 3 datasets and compared the performance of the student models with the Fine-tuned Teacher models. 

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

## Folder Structure

This Folder contains 7 Jupyter notebooks. They can be catagorized as follows

1. Amazon MASSIVE dataset
  * Part 1 code: Part_1_BERT_distilRoBERTa_and_DistilBERT_Massive_dataset_Finetuning.ipynb (https://colab.research.google.com/drive/1J9YUFZy9hMRe_M5zDJdlpCTJbKZNjq4M?usp=sharing)
  * Part 2 code: Part_2_BERT_and_DistilBERT_Massive_dataset_Knowledge_distillation.ipynb (https://colab.research.google.com/drive/1e-KhOiMeS_9EYmeIh_KnF-Yga-LHx1HV?usp=sharing)

2. Emotion dataset
  * Part 1 code: Part_1_BERT_distilRoBERTa_and_DistilBERT_Emotion_dataset_Finetuning.ipynb (https://colab.research.google.com/drive/1uV1DxrTkoy48zoHrdE478OsSYcs_FvXz?usp=sharing)
  * Part 2 code: Part_2_BERT_and_DistilBERT_emotion_dataset_Knowledge_distillation.ipynb (https://colab.research.google.com/drive/1Tq23K0BpKAGzZeOdI7u_A7BBJnJ5zflI?usp=sharing)

3. Stanford Sentiment Treebank (SST-2) dataset
  * Part 1 code: Part_1_BERT_distilRoBERTa_and_DistilBERT_SST2_dataset_Finetuning.ipynb (https://colab.research.google.com/drive/1SNtIWNmsf_uWKp-uKLYgPmAOZjkTefbU?usp=sharing)
  * Part 2 code: Part_2_BERT_and_DistilBERT_sst2_dataset_Knowledge_distillation.ipynb (https://colab.research.google.com/drive/1N-UdMWiEYexkMOxbL_XFi2RV4L-2RGct?usp=sharing)

**Part 3 code** (Part_3_Analyzing_the_model_size_and_processing_time.ipynb - **colab notebook link:** https://colab.research.google.com/drive/1RaKlccFbQHrTnIGVkzKRTMm-dH3OTogd?usp=sharing) is common for all the three datasets.

## Model cards

1. Amazon MASSIVE dataset
 * Fine-tuned models
   * BERT base: https://huggingface.co/gokuls/bert-base-Massive-intent
   * DistilRoBERTa: https://huggingface.co/gokuls/distilroberta-base-Massive-intent
   * DistilBERT: https://huggingface.co/gokuls/distilbert-base-Massive-intent
   * BERT-tiny (Student model): https://huggingface.co/gokuls/BERT-tiny-Massive-intent
 * Knowledge Distillation
   * BERT only: https://huggingface.co/gokuls/bert-tiny-Massive-intent-KD-BERT
   * DistilBERT only: https://huggingface.co/gokuls/bert-tiny-Massive-intent-KD-distilBERT
   * BERT and DistilBERT (Two Teachers): https://huggingface.co/gokuls/bert-tiny-Massive-intent-KD-BERT_and_distilBERT
2. Emotion dataset
 * Fine-tuned models
   * BERT base: https://huggingface.co/gokuls/bert-base-emotion-intent
   * DistilRoBERTa: https://huggingface.co/gokuls/distilroberta-emotion-intent
   * DistilBERT: https://huggingface.co/gokuls/distilbert-emotion-intent
   * BERT-tiny (Student model): https://huggingface.co/gokuls/BERT-tiny-emotion-intent
 * Knowledge Distillation
   * BERT only: https://huggingface.co/gokuls/bert-tiny-emotion-KD-BERT
   * DistilBERT only: https://huggingface.co/gokuls/bert-tiny-emotion-KD-distilBERT
   * BERT and DistilBERT (Two Teachers): https://huggingface.co/gokuls/bert-tiny-emotion-KD-BERT_and_distilBERT
3. Stanford Sentiment Treebank (SST-2) dataset
 * Fine-tuned models
   * BERT base: https://huggingface.co/gokuls/bert-base-sst2
   * DistilRoBERTa: https://huggingface.co/gokuls/distilroberta-sst2
   * DistilBERT: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
   * BERT-tiny (Student model): https://huggingface.co/gokuls/BERT-tiny-sst2
 * Knowledge Distillation
   * BERT only: https://huggingface.co/gokuls/bert-tiny-sst2-KD-BERT
   * DistilBERT only: https://huggingface.co/gokuls/bert-tiny-sst2-KD-distilBERT
   * BERT and DistilBERT (Two Teachers): https://huggingface.co/gokuls/bert-tiny-sst2-KD-BERT_and_distilBERT
