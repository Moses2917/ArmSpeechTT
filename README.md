# ArmSpeechTT

ArmSpeechTT is an AI model designed for speech-to-text conversion specifically tailored for the Armenian language. Leveraging the power of fine-tuning, this model, named whisper-small-hy-AM, is based on [openai/whisper-small](https://huggingface.co/openai/whisper-small) and trained on the common_voice_16_1 dataset.

## Model Performance

The model demonstrates robust performance, as indicated by the following metrics on the evaluation set:
- Loss: 0.2853
- Word Error Rate (WER): 38.1160

## Training Data and Future Enhancements

The training data consists of Mozilla Common Voice version 16.1. Plans for future improvements include continuing the training process and integrating an additional 10 hours of data from datasets such as google/fleurs and possibly google/xtreme_s. Despite its current performance, efforts are underway to further reduce the WER.

## Training Details

The model was trained on Google Colab with the following hyperparameters:

- Learning Rate: 1e-05
- Train Batch Size: 16
- Evaluation Batch Size: 8
- Seed: 42
- Optimizer: Adam with betas=(0.9, 0.999) and epsilon=1e-08
- Learning Rate Scheduler Type: Linear
- Learning Rate Scheduler Warmup Steps: 500
- Training Steps: 4000
- Mixed Precision Training: Native AMP

### Training Results

| Training Loss | Epoch | Step | Validation Loss | WER     |
|:-------------:|:-----:|:----:|:---------------:|:-------:|
| 0.0989        | 2.48  | 1000 | 0.1948          | 41.5758 |
| 0.03          | 4.95  | 2000 | 0.2165          | 39.1251 |
| 0.0016        | 7.43  | 3000 | 0.2659          | 38.4089 |
| 0.0005        | 9.9   | 4000 | 0.2853          | 38.1160 |

## Framework Versions

- Transformers 4.37.2
- Pytorch 2.1.0+cu121
- Datasets 2.16.1
- Tokenizers 0.15.1

## Running the Webcam Demo

To run the webcam demo, follow these steps:
1. Ensure you have Python installed on your system.
2. **Clone the Repository**: 
   Clone the ArmSpeechTT repository to your local machine using the following command and then cd into ArmSpeechTT:

   ```bash
   git clone https://github.com/Moses2917/ArmSpeechTT.git
   cd ArmSpeechTT
   
3. Install the required dependencies listed in `reqs.txt` using the command:
   ```
   pip install -r reqs.txt
   ```
4. Run the `webcamcaption.py` script by executing the following command:
   ```
   python webcamcaption.py
   ```
5. The script will automatically download the model and then the webcam demo will start, capturing audio input and transcribing it into text in real-time and displaying the captions on the camera.

Feel free to explore and experiment with the webcam demo to experience the capabilities of the ArmSpeechTT model firsthand!
