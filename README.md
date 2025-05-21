# SignSense (MS-ASL)

This repository contains a preprocessing pipeline for training a sign language recognition model using the **MS-ASL** dataset. It leverages **MediaPipe Holistic** for keypoint extraction and applies various **data augmentation techniques** to improve model robustness and generalization.

## Features

- Frame-wise keypoint extraction using MediaPipe Holistic (Pose, Left & Right Hand)
- Normalization and pairwise distance computation of keypoints
- 10+ types of video-level data augmentations:
  - Horizontal Flip
  - Rotation (90° and random angles)
  - Brightness Adjustment
  - Gaussian Blur
  - Gaussian Noise
  - Saturation Boost
  - Contrast Adjustment
  - Zoom (Scaling)
- Fixed-length frame sampling (default: 16 frames per video)
- Integration with custom MS-ASL annotations
- Returns frame–keypoint–label triplets for model training

---

## Dataset: MS-ASL

The MS-ASL dataset is a large-scale American Sign Language dataset with over 200 words and multiple signers. We use a **pre-filtered subset** of the dataset where videos are trimmed to a single word and manually annotated.
