# ASL Alphabet Computer Vision 🤟

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) 
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-📊-brightgreen.svg)](https://streamlit.io/)

A deployable **machine learning + computer vision project** that interprets the **ASL alphabet** from live video input and displays the predicted letter in real time.

![demo](assets/demo.gif)

## ✨ Features
- 📊 **Deep Learning Model** trained on ASL alphabet dataset (MobileNetV2 backbone)  
- 🖐️ **Hand Tracking** with [MediaPipe](https://mediapipe.dev/)  
- 🎥 **Real-time Inference** via OpenCV + optional Streamlit UI  
- 💾 **TFLite Export** for lightweight deployment 

## 🧱 Tech Stack
- Python • TensorFlow/Keras • MediaPipe • OpenCV • Streamlit

## 🗂️ Data
- Project was trained using [ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)
- dataset has about 7500 images per label
- Labels include A-Z, nothing, delete, and space.

## 📊 Results
- Achieved **91%** on held-out validation set
- Stable **hand detection** using MediaPipe landmarks  
- Correctly labeled ASL alphabet signs with confidence overlay

## 🧠 How It Works (overview)
- **MediaPipe** locates the hand and creates a bounding box
- Cropped hand images are **preprocessed** (224×224, normalized)
- A **CNN classifier** predicts the ASL letter (A–Z)

## 🛣️ Roadmap
- Export TFLite and measure CPU latency
- Improve noisy background robustness
- Improve low-light robustness
- Expand to dynamic signs (words)

---
Created by **Nathaniel Grant**
