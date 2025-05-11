# Emotion-Based Video Recommendation System

## Overview

This application recommends videos based on the user's detected emotions using a combination of:

1. Traditional mapping-based recommendations
2. Neural network-based recommendations

## System Components

- **Emotion Detection**: Uses facial analysis to detect the user's current emotion
- **Video Recommendation**: Suggests YouTube videos based on detected emotions
- **Neural Network**: Uses deep learning to personalize recommendations based on historical data
- **User Insights**: Analytics dashboard to visualize emotional patterns and video preferences

## Setup and Installation

1. Ensure all dependencies are installed:

   ```
   pip install -r requirements.txt
   ```

   Or run the provided installation scripts:

   ```
   Install_LLM_Dependencies.bat  # For local recommendation explanations
   ```

2. Set up your environment variables in `config.py`:

   - YouTube API Key
   - (No need for Gemini API Key - we now use free local LLM)

3. Run the application:
   ```
   streamlit run app.py
   ```
   or use the `Start_Application.bat` file

## Neural Network Training

Unlike traditional recommendations, the neural network model needs to be trained separately:

### Training Requirements

- At least 10 video interactions in the database
- TensorFlow 2.x and related dependencies installed

### Training Process

1. Close the main application if it's running
2. Run the training script:
   ```
   python train_neural_model.py
   ```
   or use the `Train_Model.bat` file
3. Restart the main application after training is complete

### Training Parameters

- Default: 100 epochs, batch size of 8
- You can adjust these in the training script:
  ```
  python train_neural_model.py --epochs 200 --batch-size 16
  ```

## Using the Application

1. **Main Interface**: View emotion detection and video recommendations
2. **User Insights**: Analyze your emotional patterns and video preferences
3. **Admin Panel**: View system status and run maintenance operations

## How the Neural Network Works

The neural network takes two inputs:

1. The detected emotion (one-hot encoded)
2. The time of day (hour, normalized)

It then predicts which video categories are most likely to interest the user based on historical data.

This allows for personalization beyond simple emotion-to-category mapping, taking into account:

- Individual user preferences
- Time-of-day patterns
- Subtle emotional variations

## Troubleshooting

If the neural network recommendations are not appearing:

1. Check if you have enough interaction data (at least 10 video clicks)
2. Verify the model was trained successfully (check the admin panel)
3. Try running the training script again with more epochs
