---
library_name: transformers
base_model: ./models/-google-flan-t5-base-2025-05-28_07-17-31
tags:
- generated_from_trainer
datasets:
- ./generated_data/RECOMP_tuning_with_no_Judul_nTeks
model-index:
- name: -.-models--google-flan-t5-base-2025-05-28_07-17-31-2025-05-28_10-26-57
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# -.-models--google-flan-t5-base-2025-05-28_07-17-31-2025-05-28_10-26-57

This model is a fine-tuned version of [./models/-google-flan-t5-base-2025-05-28_07-17-31](https://huggingface.co/./models/-google-flan-t5-base-2025-05-28_07-17-31) on the ./generated_data/RECOMP_tuning_with_no_Judul_nTeks dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 4
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.51.3
- Pytorch 2.5.1+cu124
- Datasets 2.19.1
- Tokenizers 0.21.1
