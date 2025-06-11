# Image Captioning System
This project implements an image captioning system using a Vision Transformer (ViT) encoder and a GPT-2 decoder, built with PyTorch, with an interactive web interface powered by Streamlit.

![Project screenshot](/ics.png)

## Introduction:
The Image Captioning System automatically generates descriptive captions for images. It achieves this by combining a powerful pre-trained Vision Transformer to understand visual content with a sophisticated GPT-2 language model to generate coherent and contextually relevant text. This system is designed for ease of use, allowing users to upload images and instantly receive generated captions through a user-friendly Streamlit application.

## Features:
**Vision Transformer (ViT) Encoder:** Extracts rich visual features from images using a pre-trained ViT model.

**GPT-2 Decoder:** Generates human-like captions by leveraging the language understanding capabilities of a pre-trained GPT-2 model, conditioned on the extracted image features.

**Transfer Learning:** Utilizes pre-trained models (ViT and GPT-2) and fine-tunes them for the specific task of image captioning, ensuring high performance with less training data.

**Interactive Streamlit Interface:** Provides an intuitive web application for users to upload images and view generated captions.

## Installation:
Follow these steps to set up the project locally:
#### Clone the repository:
```
git clone https://github.com/RounakS33/Image_Captioning.git
cd image_captioning_system
```

#### Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage:
#### Run the Streamlit app:
```
streamlit run app.py
```

## Technologies Used:
Deep Learning, PyTorch, Hugging Face Transformers, Natural Language Processing (NLP), Computer Vision, Python, Streamlit
