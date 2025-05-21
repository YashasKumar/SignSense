# SignSense (MS-ASL)

This repository contains a preprocessing and real-time inference pipeline for training and recognizing sign language gestures using the **MS-ASL** dataset. It leverages **MediaPipe Holistic** for extracting 3D keypoints of hands and pose, performs advanced preprocessing including pairwise distance computation, and integrates with deep learning models for sign classification. Additionally, it demonstrates the use of **transformers (LLaMA 2)** for natural language sentence formation from unordered words, showcasing potential NLP integration.

## Features

- Frame-wise keypoint extraction using MediaPipe Holistic (Pose, Left & Right Hands)
- Normalization and pairwise Euclidean distance computation of keypoints
- Fixed-length frame sampling (default: 16 frames per video segment)
- Real-time webcam capture and inference with OpenCV
- Fine-tuned Keras model for sign classification using frames and spatial features
- Example of sentence formation using LLaMA 2 causal language model from unordered word lists
- Various data augmentations applied during training for robustness (flip, rotation, brightness, blur, noise, contrast, zoom)

## Dataset: MS-ASL

MS-ASL is a large-scale American Sign Language dataset containing over 200 signs performed by multiple signers. This project uses a pre-filtered subset trimmed to single-word clips with manual annotations for precise training.

## Pipeline Overview

### Preprocessing and Feature Extraction

- Capture or load video frames
- Extract 3D keypoints of hands and pose landmarks with MediaPipe Holistic
- Normalize and compute pairwise distances to capture spatial relationships
- Interpolate frames and keypoints to fixed length (16 frames)
- Prepare data inputs for training or real-time inference

### Model Inference

- Load a fine-tuned Keras model for sign classification
- Predict sign classes from processed RGB frames and keypoint distances
- Apply confidence threshold to filter predictions
- Display live predictions during webcam capture

### Transformer Sentence Formation (Example)

- Load LLaMA 2 causal language model and tokenizer using Hugging Face `transformers`
- Generate grammatically correct sentences from unordered input words
- Demonstrates natural language processing capabilities linked to sign recognition

## Usage

### Real-time Sign Language Recognition

```bash
python try.py
```

### Sentence Formation with Transformers
``` bash
python working.py
```

## Future Work
- Transformer-based sequence-to-sequence models for direct video-to-text sign translation
- Full sentence-level sign language recognition with temporal segmentation
- GPU-accelerated keypoint extraction (e.g., MMPose) for faster real-time inference
- Advanced NLP integration for contextual sign language interpretation
