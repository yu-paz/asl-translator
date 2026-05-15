# ASL Translator (Static-Sign Classifier)

A simple computer-vision project that classifies four American Sign Language
hand signs from a single image: **"I love you"**, **"yes"**, **"stop"**, and
**"hello"**. Built with PyTorch and transfer learning on MobileNetV2.

## Project Structure

```
asl-translator/
├── src/
│   ├── train.py          # Train the model
│   ├── predict.py        # Classify a single image
│   └── webcam_demo.py    # Live webcam demo (final-deliverable bonus)
├── dataset/              # YOUR photos go here (one folder per class)
│   ├── i_love_you/
│   ├── yes/
│   ├── stop/
│   └── hello/
├── models/               # Trained model checkpoint(s) — created by train.py
├── figures/              # Plots — created by train.py
├── logs/                 # Per-epoch metrics CSV — created by train.py
├── requirements.txt
└── README.md
```


## Method 

I used MobileNetV2 pretrained on ImageNet as a fixed feature extractor and
train a small classifier head (Dropout + Linear) on top for our 4 classes.
This is the standard *transfer learning as feature extraction* approach: it
needs much less data than training from scratch, which is essential for a
custom-collected dataset of ~30 images per class.

