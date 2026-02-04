# new_one

This repository contains a small example project with an end-to-end PyTorch CNN for image classification (ImageFolder format).

## Quick start

1. Install dependencies:

   pip install -r requirements.txt

2. Train (example):

   from cnn_end_to_end import train_cnn
   train_cnn("data/train", "data/val", epochs=10, batch_size=64, checkpoint_path="best.pth")

3. Inference (example):

   from cnn_end_to_end import load_model_for_inference
   model, classes = load_model_for_inference("best.pth")
