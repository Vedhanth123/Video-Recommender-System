"""
Neural network-based recommendation system for the Emotion-Based Video Recommendation application.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, regularizers
import tensorflow as tf
import sqlite3
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

from config import DATABASE_PATH

class NeuralRecommender:
    """Neural network-based recommendation system for video recommendations."""
    
    def __init__(self, model_dir='models'):
        """
        Initialize the neural recommendation system.
        
        Args:
            model_dir (str): Directory to store model files and encoders
        """
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'emotion_video_nn.h5')
        self.encoders_path = os.path.join(model_dir, 'emotion_video_encoders.pkl')
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Neural network model
        self.model = None
        
        # Encoders and scalers
        self.emotion_encoder = None
        self.category_encoder = None
        self.feature_scaler = None
        
        # Load model and encoders if they exist
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model and encoders or create new ones."""
        try:
            if os.path.exists(self.model_path):
                self.model = models.load_model(self.model_path)
                
                # Load encoders and scaler
                if os.path.exists(self.encoders_path):
                    encoders = joblib.load(self.encoders_path)
                    self.emotion_encoder = encoders.get('emotion_encoder')
                    self.category_encoder = encoders.get('category_encoder')
                    self.feature_scaler = encoders.get('feature_scaler')
                    
                    logging.info("Successfully loaded model and encoders")
                    return True
            
            # If model or encoders don't exist, create new ones
            logging.info("Creating new model and encoders")
            return False
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False

    def _create_model(self, input_dim, output_dim):
        """
        Create a neural network model for emotion-video recommendation.
        
        Args:
            input_dim (int): Input dimension size (emotions + user features)
            output_dim (int): Output dimension size (video categories)
            
        Returns:
            keras.Model: The compiled neural network model
        """
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,),
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(output_dim, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_training_data(self):
        """
        Prepare training data from the database.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) for model training
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(DATABASE_PATH)
            
            # Get emotion logs
            emotion_data = pd.read_sql_query("SELECT * FROM emotion_logs", conn)
            
            # Get video clicks
            try:
                video_data = pd.read_sql_query("SELECT * FROM video_clicks", conn)
            except:
                conn.close()
                logging.error("No video_clicks table found. Cannot train model without video interactions.")
                return None, None, None, None
                
            conn.close()
            
            # Check if we have sufficient data
            if len(video_data) < 10:
                logging.warning(f"Insufficient training data: only {len(video_data)} video interactions.")
                return None, None, None, None
            
            # Extract features
            features = []
            categories = []
            
            for _, click in video_data.iterrows():
                user_name = click['name']
                emotion = click['emotion']
                category = click['video_category']
                
                # Get user's recent emotions (last 5)
                user_emotions = emotion_data[emotion_data['name'] == user_name].sort_values(
                    by='timestamp', ascending=False
                ).head(5)['emotion'].tolist()
                
                # Pad with current emotion if not enough history
                while len(user_emotions) < 5:
                    user_emotions.append(emotion)
                
                # Hour of day as a feature (0-23)
                hour = datetime.fromtimestamp(click['timestamp']).hour
                
                # Create feature vector: [emotion, time_of_day]
                feature = [emotion, hour]
                features.append(feature)
                categories.append(category)
            
            # Convert to numpy arrays
            features = np.array(features)
            categories = np.array(categories)
            
            # Create and fit encoders if they don't exist
            if self.emotion_encoder is None:
                self.emotion_encoder = OneHotEncoder(sparse_output=False)
                emotion_encoded = self.emotion_encoder.fit_transform(features[:, 0].reshape(-1, 1))
            else:
                emotion_encoded = self.emotion_encoder.transform(features[:, 0].reshape(-1, 1))
                
            if self.category_encoder is None:
                self.category_encoder = OneHotEncoder(sparse_output=False)
                categories_encoded = self.category_encoder.fit_transform(categories.reshape(-1, 1))
            else:
                categories_encoded = self.category_encoder.transform(categories.reshape(-1, 1))
                
            # Prepare numerical features
            numerical_features = features[:, 1].reshape(-1, 1).astype(float)  # hour
            
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
                numerical_features_scaled = self.feature_scaler.fit_transform(numerical_features)
            else:
                numerical_features_scaled = self.feature_scaler.transform(numerical_features)
                
            # Combine all features
            X = np.hstack((emotion_encoded, numerical_features_scaled))
            y = categories_encoded
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error preparing training data: {e}")
            return None, None, None, None
    
    def train_model(self, epochs=50, batch_size=8, verbose=1):
        """
        Train the neural network model using the database data.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level for training
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Prepare training data
            X_train, X_test, y_train, y_test = self._prepare_training_data()
            
            if X_train is None:
                return False
                
            # Create model if it doesn't exist
            if self.model is None:
                input_dim = X_train.shape[1]
                output_dim = y_train.shape[1]
                self.model = self._create_model(input_dim, output_dim)
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose
            )
            
            # Save model and encoders
            self.model.save(self.model_path)
            
            encoders = {
                'emotion_encoder': self.emotion_encoder,
                'category_encoder': self.category_encoder,
                'feature_scaler': self.feature_scaler
            }
            joblib.dump(encoders, self.encoders_path)
            
            # Log training results
            val_acc = history.history['val_accuracy'][-1]
            logging.info(f"Model trained successfully. Validation accuracy: {val_acc:.4f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return False
    
    def recommend_categories(self, emotion, time_of_day=None):
        """
        Get recommended video categories based on emotion and time of day.
        
        Args:
            emotion (str): The detected emotion
            time_of_day (int, optional): Hour of the day (0-23). Defaults to current hour.
            
        Returns:
            list: Ranked list of (category, score) tuples
        """
        if self.model is None or self.emotion_encoder is None or self.category_encoder is None:
            logging.warning("Model or encoders not loaded. Cannot make recommendations.")
            return []
            
        try:
            # Use current hour if not provided
            if time_of_day is None:
                time_of_day = datetime.now().hour
                
            # Encode emotion
            try:
                emotion_encoded = self.emotion_encoder.transform([[emotion]])
            except:
                logging.error(f"Unknown emotion: {emotion}")
                return []
                
            # Scale time feature
            time_scaled = self.feature_scaler.transform([[float(time_of_day)]])
            
            # Combine features
            features = np.hstack((emotion_encoded, time_scaled))
            
            # Get predictions
            predictions = self.model.predict(features, verbose=0)[0]
            
            # Map predictions to categories
            categories = self.category_encoder.categories_[0]
            
            # Create ranked list of (category, score) tuples
            recommendations = [(categories[i], float(predictions[i])) 
                              for i in range(len(categories))]
            
            # Sort by score in descending order
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error making recommendations: {e}")
            return []
