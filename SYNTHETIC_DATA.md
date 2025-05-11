# Synthetic Data Documentation

This document explains the synthetic data generation for the Emotion-Based Video Recommendation system that uses a local LLM (Hugging Face's TinyLlama) for explanations.

## Purpose

The neural network model requires training data to learn patterns between user emotions and video preferences. The synthetic data simulates user interactions that would normally be collected over time through actual system usage.

## Data Tables

The synthetic data populates the following tables in the SQLite database (`user_data.db`):

1. **`emotion_logs`** - Records of user emotions detected

   - `name`: User name
   - `emotion`: Detected emotion (happy, sad, angry, fear, surprise, neutral, disgust)
   - `input`: User text input associated with the emotion
   - `timestamp`: Unix timestamp of when the emotion was detected
   - `city`, `country`: User location

2. **`video_clicks`** - Records of users clicking on recommended videos

   - `name`: User name
   - `video_id`: YouTube video ID
   - `video_title`: Title of the video
   - `video_category`: YouTube category ID
   - `emotion`: User's emotion when clicking the video
   - `timestamp`: Unix timestamp of the click
   - `city`, `country`: User location

3. **`video_interactions`** - Higher-level data about video genre interactions
   - `name`: User name
   - `video_genre`: Genre/category of the video
   - `emotion`: User's emotion when interacting with the video
   - `timestamp`: Unix timestamp of the interaction
   - `city`, `country`: User location

## Data Generation

The synthetic data generation:

1. Creates a set of random synthetic users
2. Generates emotion logs for these users over the past 30 days
3. Creates video click events that occur shortly after emotion detection
4. Creates video interaction records that associate emotions with video genres
5. Uses reasonable probability distributions to ensure the data is realistic

## Data Patterns

The synthetic data incorporates several important patterns:

1. Users tend to watch videos that match their emotional state (80% probability)
2. The video clicks and interactions occur after emotion detection (realistic time flow)
3. Location data is mostly consistent for the same user session
4. Video categories align with the emotion-to-video mapping defined in the system

## Using the Synthetic Data

To generate the synthetic data:

1. Run the `Generate_Synthetic_Data.bat` script or execute `python generate_synthetic_data.py`
2. Verify that the `user_data.db` file has been created or updated
3. Check database content with SQL queries if needed

To train the neural network model using this data:

1. Run the `Train_Model.bat` script or execute `python train_neural_model.py`
2. The trained model will be saved in the `models` directory
3. Restart the application to use the trained model for recommendations

## Configuration

You can modify the amount of synthetic data by changing these constants in `generate_synthetic_data.py`:

```python
NUM_USERS = 5  # Number of synthetic users
NUM_EMOTION_LOGS = 50  # Number of emotion logs to generate
NUM_VIDEO_CLICKS = 20  # Number of video clicks to generate
NUM_VIDEO_INTERACTIONS = 30  # Number of video interactions to generate
```

## Data Quality

The synthetic data is designed to train the neural network with sensible patterns. However, real user data will always be superior for personalization. The synthetic data should be considered a starting point to enable initial model training.
