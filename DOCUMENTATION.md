# Emotion-Based Video Recommendation System Documentation

**Date:** May 11, 2025  
**Version:** 1.0  
**Author:** Project Team

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Key Components](#key-components)
   - [Emotion Analysis](#emotion-analysis)
   - [Video Recommendation](#video-recommendation)
   - [Recommendation Explanation](#recommendation-explanation)
   - [Neural Network Integration](#neural-network-integration)
   - [Machine Learning Integration](#machine-learning-integration)
   - [User Interface](#user-interface)
4. [Implementation Details](#implementation-details)
   - [Local LLM Integration](#local-llm-integration)
   - [Face Detection & Authentication](#face-detection--authentication)
   - [Neural Network Training](#neural-network-training)
   - [Data Management](#data-management)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Troubleshooting](#troubleshooting)
8. [Future Enhancements](#future-enhancements)

## Project Overview

The Emotion-Based Video Recommendation System is an innovative application that detects users' facial emotions in real-time and provides personalized YouTube video recommendations based on their emotional state. The system utilizes a combination of computer vision for emotion detection, API integration for content retrieval, and both traditional mapping and neural network-based approaches for personalized recommendations.

The system also features a local language model (LLM) for generating natural language explanations about why specific videos are recommended, eliminating dependency on paid external API services.

## System Architecture

The application follows a modular architecture with the following main components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Emotion        │────▶│  Recommendation │────▶│  User           │
│  Detection      │     │  Engine         │     │  Interface      │
│                 │     │                 │     │                 │
└─────────────────┘     └───────┬─────────┘     └─────────────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
          ┌──────────▼─────────┐ ┌───────▼──────────┐
          │                    │ │                  │
          │  YouTube API       │ │  Local LLM       │
          │  Integration       │ │  Explainer       │
          │                    │ │                  │
          └────────────────────┘ └──────────────────┘
```

The system interacts with external services (YouTube API) while maintaining user privacy by processing emotions and generating explanations locally.

## Key Components

### Emotion Analysis

**File:** `emotion_analysis.py`

- Uses face detection and recognition to identify users
- Analyzes facial expressions to determine emotional states
- Supports seven basic emotions: happy, sad, angry, fear, surprise, disgust, and neutral
- Maintains a database of known users for personalization

### Video Recommendation

**Files:** `youtube_recommender.py`, `recommendation_engine.py`

- Integrates with YouTube API to fetch relevant videos
- Maps detected emotions to appropriate content categories and keywords
- Supports location-aware recommendations for regional content
- Implements caching to improve performance and reduce API calls

### Recommendation Explanation

**Files:** `local_explainer.py`, `gemini_explainer.py`

- Generates natural language explanations for video recommendations
- Uses local LLM (TinyLlama) to create personalized explanations
- Provides fallback template-based explanations when LLM is unavailable
- Caches explanations to improve response time

### Neural Network Integration

**Files:** `nn_recommender.py`, `train_neural_model.py`

- Deep learning model that predicts video categories based on emotions and time
- Trains on historical user interaction data
- Provides personalized recommendations beyond simple emotion-to-category mapping
- Improves over time as more user data becomes available

### Machine Learning Integration

**File:** `ml_recommender.py`

- Uses traditional machine learning to rank recommendations
- Builds user profiles based on interaction history
- Adapts to individual preference patterns
- Works in conjunction with neural recommendations

### User Interface

**File:** `app.py`

- Streamlit-based web interface
- Four main tabs: Face Detection, Video Recommendations, User Insights, and Admin Panel
- Real-time webcam integration for emotion detection
- Interactive recommendation display with explanations

## Implementation Details

### Local LLM Integration

The system uses a lightweight local language model to generate natural language explanations for video recommendations:

- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~1.1GB)
- **Benefits**:
  - No API costs
  - Works offline
  - Better privacy (all processing stays on device)
  - Custom fallback templates for each emotion type
- **Caching**: The model is downloaded only once and saved in the `models/tinyllama_cache` folder, then reused for subsequent runs
- **Dependencies**: transformers, torch, accelerate

### Face Detection & Authentication

- Uses `face_recognition` library for detection and recognition
- Stores known faces with encodings for authentication
- Provides registration interface for new users
- Implements privacy-focused emotion detection

### Neural Network Training

The neural network recommendation system:

- Uses TensorFlow for deep learning
- Requires a minimum of 10 video interactions to start training
- Features a multi-layer architecture with regularization to prevent overfitting
- Takes both emotion and time of day as input features
- Outputs video category predictions with confidence scores

### Data Management

- SQLite database (`user_data.db`) for storing:
  - User emotion logs
  - Video interaction data
  - Click tracking
  - User profiles
- Implements schema validation and automatic table creation

## Installation & Setup

1. **Prerequisites**:

   - Python 3.8+ with pip
   - Webcam for emotion detection
   - Internet connection for YouTube API

2. **Installation**:

   ```
   pip install -r requirements.txt
   ```

   For local LLM integration:

   ```
   Install_LLM_Dependencies.bat
   ```

3. **Configuration**:

   - Set up your YouTube API key in `config.py`
   - No external API key required for explanations (uses local LLM)

4. **Running the Application**:
   ```
   streamlit run app.py
   ```
   or use the `Start_Application.bat` file

## Usage Guide

### Face Detection

1. Click "Start Face Detection" to begin emotion analysis
2. Look at the camera for the set duration
3. The system will detect your face and emotion
4. Unidentified faces can be named and saved

### Video Recommendations

1. Select a user from the dropdown menu
2. Click "Get Recommendations"
3. Browse personalized video recommendations based on your emotional state
4. Click on videos to watch on YouTube (interactions are tracked for better recommendations)

### User Insights

1. Authenticate with face recognition
2. View emotion trends, video watching patterns, and personalized recommendations
3. Explore analytics about your emotional journey and content preferences

### Admin Panel

1. View system status including neural network availability
2. Train the neural network model with existing interaction data
3. Manage application settings

## Troubleshooting

### Common Issues

1. **No emotion detected**:

   - Ensure proper lighting
   - Position your face clearly in the webcam view
   - Try adjusting the detection duration

2. **Neural network not providing recommendations**:

   - Check if you have at least 10 video interactions in the database
   - Verify that the model was trained correctly
   - Try running the training script again

3. **Local LLM not working**:
   - Ensure dependencies are installed correctly
   - Check if model download was completed successfully
   - Verify that the `models/tinyllama_cache` directory exists

### Error Messages

- "No known faces found": Register your face first through the Face Detection tab
- "Insufficient training data": Interact with more videos to build up training data
- "Could not load local LLM": Run `Install_LLM_Dependencies.bat` and restart

## Future Enhancements

1. **Multi-emotion detection**: Recognize mixed emotions and their intensities
2. **Collaborative filtering**: Incorporate recommendations based on similar users
3. **Video content analysis**: Analyze video content directly for better matching
4. **Offline mode**: Full functionality without internet connection
5. **Mobile application**: Port the system to mobile platforms
6. **Custom content sources**: Integrate with additional video platforms beyond YouTube

---

This documentation provides a comprehensive overview of the Emotion-Based Video Recommendation System. For specific technical details, refer to the codebase and individual module documentation.
