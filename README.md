# ArmSpeechTT
AI speech to text model finetuned on the Armenian language

## Model description
This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the common_voice_16_1 dataset. It is finetuned for the Armenian language. I call my model whisper-small-hy-AM.

It achieves the following results on the evaluation set:
- Loss: 0.2853
- Wer: 38.1160

## Training and evaluation data

Trained on mozilla common voice version 16.1, currently have plans to continue training, and also merge an extra 10 hours of data from google/fleurs and possibly google/xtreme_s.
Currently has a lower word error rate than even openai's whisper large v2 and v3, which have about a 48% word error rate, meanwhile my model has a 38% word error rate, 10% lower than whisper's and after some training it will be even lower. 

## Training procedure
Trained on google colab for 4000 steps.
### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 4000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer     |
|:-------------:|:-----:|:----:|:---------------:|:-------:|
| 0.0989        | 2.48  | 1000 | 0.1948          | 41.5758 |
| 0.03          | 4.95  | 2000 | 0.2165          | 39.1251 |
| 0.0016        | 7.43  | 3000 | 0.2659          | 38.4089 |
| 0.0005        | 9.9   | 4000 | 0.2853          | 38.1160 |


### Framework versions

- Transformers 4.37.2
- Pytorch 2.1.0+cu121
- Datasets 2.16.1
- Tokenizers 0.15.1
