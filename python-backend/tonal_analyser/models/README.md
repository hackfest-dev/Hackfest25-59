# Emotion Recognition Models

This directory contains pre-trained models for emotion recognition from audio.

## Expected Models

- `emotion_model.pt`: PyTorch model for emotion classification

## Model Format

The models should be PyTorch models that take MFCCs as input and output emotion class probabilities.

## Training Your Own Models

To train your own emotion recognition model, you can use datasets like:

- RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
- TESS: Toronto Emotional Speech Set
- EmoDB: Berlin Database of Emotional Speech

A sample training script will be provided in a future update.

## Using Pre-trained Models

If you don't have a pre-trained model, the system will fall back to a dummy model that returns random emotion probabilities. This is useful for testing the system, but not for actual emotion recognition.

To use a pre-trained model, place the model file in this directory and ensure it has the correct name as specified in the configuration.
