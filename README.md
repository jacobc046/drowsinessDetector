# Drowsiness Detection System

## Overview
The Drowsiness Detection System is a real-time computer vision application that monitors facial features to determine if a user is alert or drowsy. Utilizing deep learning techniques, the system analyzes eye movements and facial landmarks to identify signs of drowsiness, making it useful for applications such as driver safety and workplace monitoring.

## Features
- **Real-Time Detection:** Continuously monitors facial features to assess user alertness.
- **Deep Learning Integration:** Uses PyTorch-based neural networks for accurate classification.
- **Facial Landmark Detection:** Implements OpenCV for precise feature extraction.
- **User Alerts:** Provides visual warnings on screen when drowsiness is detected.

## Technologies Used
- **Python** – Core programming language for development.
- **PyTorch** – Deep learning framework for training and implementing neural networks.
- **OpenCV** – Computer vision library for real-time image processing and facial landmark detection.
- **Numpy** – For numerical computations and data processing.

## Installation
### Prerequisites
Ensure you have Python installed (version 3.7 or higher). Then, install the necessary dependencies:
```bash
pip install opencv-python numpy scipy torch torchvision
```

### Clone the Repository
```bash
git clone https://github.com/jacobc046/drowsinessDetector
cd drowsinessDetector
```

## Usage
1. Run the application:
```bash
python drowsinessDetector.py
```
2. The system will access your webcam and begin real-time monitoring.
3. If drowsiness is detected, an on-screen alert will be displayed.

## How It Works
1. Captures real-time video using OpenCV.
2. Detects facial landmarks to identify eyes and face.
3. Uses a deep learning model to analyze eye openness and facial movements.
4. Classifies the user's state as **Alert** or **Drowsy**.

## Acknowledgments
- OpenCV and PyTorch for providing powerful computer vision and deep learning tools.
- Research in drowsiness detection for inspiring the project.
- Kagglehub for providing free access to large datasets of various eye open-or-closed states used for training. 
