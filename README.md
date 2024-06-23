---
license: apache-2.0
base_model: t5-base
tags:
- generated_from_trainer
model-index:
- name: finnetuned-tf-base-model
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# finnetuned-tf-base-model

This model is a fine-tuned version of [t5-base](https://huggingface.co/t5-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7519

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| No log        | 1.0   | 17   | 1.1922          |
| No log        | 2.0   | 34   | 1.0339          |
| No log        | 3.0   | 51   | 0.9496          |
| No log        | 4.0   | 68   | 0.8874          |
| No log        | 5.0   | 85   | 0.8411          |
| No log        | 6.0   | 102  | 0.8031          |
| No log        | 7.0   | 119  | 0.7797          |
| No log        | 8.0   | 136  | 0.7644          |
| No log        | 9.0   | 153  | 0.7550          |
| No log        | 10.0  | 170  | 0.7519          |


### Framework versions

- Transformers 4.41.2
- Pytorch 2.3.0+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1
