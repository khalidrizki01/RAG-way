---
library_name: transformers
base_model: khalidrizki/T5base-RECOMP-seededDataset-withSelection
tags:
- generated_from_trainer
datasets:
- khalidrizki/RECOMP-tuning
model-index:
- name: -khalidrizki-T5base-RECOMP-seededDataset-withSelection-2025-06-11_12-41-43
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# -khalidrizki-T5base-RECOMP-seededDataset-withSelection-2025-06-11_12-41-43

This model is a fine-tuned version of [khalidrizki/T5base-RECOMP-seededDataset-withSelection](https://huggingface.co/khalidrizki/T5base-RECOMP-seededDataset-withSelection) on the khalidrizki/RECOMP-tuning dataset.

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
- eval_batch_size: 32
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.51.0
- Pytorch 2.5.1
- Datasets 2.19.1
- Tokenizers 0.21.1
