# Emotion-Aware Intelligent Video Recommendation Framework

An advanced video recommendation system that detects user emotions through facial analysis and provides personalized content recommendations based on emotional state.

## Overview

This project implements an intelligent recommendation system that combines computer vision for emotion detection with machine learning for personalized content recommendations. The system uses facial emotion recognition to detect the user's current emotional state and then suggests YouTube videos tailored to that emotion.

## Key Features

- **Real-time Emotion Detection**: Recognizes seven emotions (happy, sad, angry, fear, surprise, disgust, neutral) using computer vision
- **Personalized Recommendations**: Maps emotions to appropriate content types using both rule-based and neural network approaches
- **Local Explanation Model**: Uses a lightweight local language model to explain recommendations without external API calls
- **Neural Network Integration**: Learns from user interactions to improve recommendations over time
- **Enhanced Recommendation Features**:
  - Time-of-day awareness
  - Collaborative filtering
  - Content diversity optimization
  - User context awareness

## Getting Started

### Prerequisites

- Python 3.8+
- Webcam for emotion detection
- An internet connection for video recommendations

### Installation

1. Clone this repository
2. Create and activate a virtual environment:

```powershell
python -m venv MP
.\MP\Scripts\Activate.ps1
```

3. Install the required packages:

```powershell
pip install -r requirements.txt
```

4. Install the local language model dependencies:

```powershell
.\Install_LLM_Dependencies.bat
```

### Running the Application

Start the application using:

```powershell
.\Start_Application.ps1
```

Or create a desktop shortcut:

```powershell
.\Create_Desktop_Shortcut.ps1
```

## System Architecture

The system consists of the following main components:

1. **Emotion Detection Module** (`emotion_analysis.py`): Handles face detection, recognition, and emotion classification
2. **Recommendation Engine** (`recommendation_engine.py`): Maps emotions to appropriate video content
3. **Neural Network** (`nn_recommender.py`): Makes personalized category predictions based on user's emotion and time of day
4. **Local Explanation System** (`local_explainer.py`): Generates natural language explanations for recommendations

## Documentation

Detailed documentation can be found in:

- `DOCUMENTATION.md`: System overview and architecture
- `USAGE.md`: User guide and features
- `ENHANCEMENT_DOCUMENTATION.md`: Technical details of enhanced recommendation features
- `LOCAL_LLM_README.md`: Local language model integration details

## Utilities

- `Generate_Synthetic_Data.bat`: Creates synthetic data for initial neural network training
- `Train_Model.bat`: Trains the neural network model with existing data
- `Clear_Database.bat`: Resets the database (use with caution)
- `Test_Enhancements.bat`: Tests the enhanced recommendation features

## License

This project was developed for academic purposes.

## Acknowledgments

Special thanks to all contributors and to the open-source libraries that made this project possible.
