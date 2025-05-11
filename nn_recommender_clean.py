"""
Neural network-based recommendation system for emotion-based video recommendations.
Uses pre-trained models to predict video categories.
"""

import os
import numpy as np
import tensorflow as tf
import joblib
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NNRecommender")

# Configure tensorflow to be less verbose
tf.get_logger().setLevel("ERROR")

class NNRecommender:
    """Neural network-based recommendation system for video recommendations."""
    
    def __init__(self, model_dir="models"):
        """Initialize the neural network recommender."""
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "emotion_video_nn.h5")
        self.encoders_path = os.path.join(model_dir, "emotion_video_encoders.pkl")
        self.model = None
        self.emotion_encoder = None
        self.category_encoder = None
        self.feature_scaler = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to load existing model and encoders
        self._load_model_and_encoders()
        
        if self.is_ready():
            logger.info("Neural network recommendation system is ready")
        else:
            logger.warning("Neural network model or encoders not found. Please run train_neural_model.py first.")
    
    def _load_model_and_encoders(self):
        """Load pre-trained model and encoders if they exist."""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            else:
                logger.info(f"No pre-trained model found at {self.model_path}")
                
            if os.path.exists(self.encoders_path):
                encoders_dict = joblib.load(self.encoders_path)
                self.emotion_encoder = encoders_dict.get("emotion_encoder")
                self.category_encoder = encoders_dict.get("category_encoder")
                self.feature_scaler = encoders_dict.get("feature_scaler")
                logger.info(f"Loaded encoders from {self.encoders_path}")
            else:
                logger.info(f"No encoders found at {self.encoders_path}")
        except Exception as e:
            logger.error(f"Error loading model or encoders: {e}")
    
    def predict_categories(self, emotion, hour=None):
        """Predict video categories based on emotion and time of day."""
        if not self.is_ready():
            logger.warning("Model or encoders not loaded, cannot make predictions")
            return []
            
        if hour is None:
            hour = datetime.now().hour
            
        try:
            # Encode emotion
            emotion_encoded = self.emotion_encoder.transform([[emotion]])
            
            # Scale hour
            hour_scaled = self.feature_scaler.transform([[float(hour)]])
            
            # Combine features
            features = np.hstack((emotion_encoded, hour_scaled))
            
            # Make prediction
            predictions = self.model.predict(features, verbose=0)[0]
            
            # Convert to category names
            categories = self.category_encoder.categories_[0]
            
            # Create ranked list of (category, score)
            results = [(cat, float(score)) for cat, score in zip(categories, predictions)]
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return []
    
    def is_ready(self):
        """Check if the model and encoders are loaded and ready to use."""
        return (self.model is not None and 
                self.emotion_encoder is not None and 
                self.category_encoder is not None and 
                self.feature_scaler is not None)
