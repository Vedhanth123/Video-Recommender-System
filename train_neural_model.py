"""
Standalone script to train the neural network model for emotion-based video recommendations.
This script trains the model outside of the Streamlit app for better performance and separation of concerns.

Key benefits of this standalone training approach:
1. Prevents resource-intensive training from affecting the Streamlit app's performance
2. Allows for more complex and time-consuming training procedures
3. Creates a clear separation between model training and model inference
4. Enables batch processing of historical data for better model quality
5. Makes it easier to experiment with different model architectures and hyperparameters

The trained model is saved to the 'models' directory and automatically loaded by the main application.
"""

import os
import numpy as np
import pandas as pd
import argparse
import logging
import tensorflow as tf
import sqlite3
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('NNTrainer')

# Default paths
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.db")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_video_nn.h5")
ENCODERS_PATH = os.path.join(MODEL_DIR, "emotion_video_encoders.pkl")

def build_model(input_dim, output_dim):
    """
    Build a neural network model for emotion-video recommendation.
    
    Args:
        input_dim: Input dimension size (emotions + user features)
        output_dim: Output dimension size (video categories)
        
    Returns:
        A compiled Keras model
    """
    from tensorflow.keras import layers, models, regularizers
    
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,),
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(output_dim, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Built model with input dim {input_dim} and output dim {output_dim}")
    return model

def prepare_training_data(db_path):
    """
    Prepare emotion and video click data for training.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, encoders_dict) or (None, None, None, None, None) if data is insufficient
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        
        # Get video clicks
        try:
            video_data = pd.read_sql_query("SELECT * FROM video_clicks", conn)
            logger.info(f"Loaded {len(video_data)} video click records from database")
        except Exception as e:
            logger.error(f"Failed to load video_clicks table: {e}")
            conn.close()
            return None, None, None, None, None
            
        # Get emotion logs
        emotion_data = pd.read_sql_query("SELECT * FROM emotion_logs", conn)
        logger.info(f"Loaded {len(emotion_data)} emotion log records from database")
        conn.close()
        
        if len(video_data) < 10:
            logger.warning(f"Insufficient training data: only {len(video_data)} video interactions")
            return None, None, None, None, None
            
        # Prepare features and labels
        features = []
        labels = []
        
        for _, row in video_data.iterrows():
            # Extract features: emotion and hour of day
            emotion = row['emotion']
            timestamp = row['timestamp']
            hour = datetime.fromtimestamp(timestamp).hour
            
            # Extract label: video category
            category = row['video_category']
            
            features.append([emotion, hour])
            labels.append(category)
            
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels).reshape(-1, 1)
        
        # Create encoders
        emotion_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        emotions_encoded = emotion_encoder.fit_transform(features[:, 0].reshape(-1, 1))
        
        category_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categories_encoded = category_encoder.fit_transform(labels)
        
        # Create feature scaler for numerical features (hour)
        numerical_features = features[:, 1].reshape(-1, 1).astype(float)
        feature_scaler = StandardScaler()
        numerical_features_scaled = feature_scaler.fit_transform(numerical_features)
            
        # Combine features
        X = np.hstack((emotions_encoded, numerical_features_scaled))
        y = categories_encoded
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create encoders dictionary for saving
        encoders_dict = {
            'emotion_encoder': emotion_encoder,
            'category_encoder': category_encoder,
            'feature_scaler': feature_scaler
        }
        
        return X_train, X_test, y_train, y_test, encoders_dict
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        return None, None, None, None, None

def train_model(db_path, model_path, encoders_path, epochs=50, batch_size=16, verbose=1):
    """
    Train the neural network model and save it to disk.
    
    Args:
        db_path: Path to the SQLite database
        model_path: Path to save the trained model
        encoders_path: Path to save the encoders
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Verbosity level
        
    Returns:
        True if training was successful, False otherwise
    """
    # Make sure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Prepare training data
    X_train, X_test, y_train, y_test, encoders_dict = prepare_training_data(db_path)
    
    if X_train is None:
        return False
    
    # Build the model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = build_model(input_dim, output_dim)
    
    try:
        # Train the model
        logger.info(f"Starting model training with {epochs} epochs and batch size {batch_size}")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=verbose
        )
        
        # Evaluate the model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {test_acc:.4f}")
        
        # Save model and encoders
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        joblib.dump(encoders_dict, encoders_path)
        logger.info(f"Encoders saved to {encoders_path}")
        
        # Plot and save training history
        plot_training_history(history, os.path.join(os.path.dirname(model_path), "training_history.png"))
        
        return True
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False

def plot_training_history(history, save_path=None):
    """
    Plot the training history and optionally save to file.
    
    Args:
        history: The history object returned by model.fit()
        save_path: Path to save the plot, if None the plot is displayed
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Train neural network model for emotion-based video recommendations')
    
    parser.add_argument('--db', type=str, default=DATABASE_PATH,
                        help='Path to the SQLite database')
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR,
                        help='Directory to save model files')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0, 1, or 2)')
    
    args = parser.parse_args()
    
    # Set model paths
    model_path = os.path.join(args.model_dir, "emotion_video_nn.h5")
    encoders_path = os.path.join(args.model_dir, "emotion_video_encoders.pkl")
    
    # Print training configuration
    logger.info("Training configuration:")
    logger.info(f"  Database: {args.db}")
    logger.info(f"  Model output: {model_path}")
    logger.info(f"  Encoders output: {encoders_path}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    
    # Start training
    success = train_model(
        args.db,
        model_path,
        encoders_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose
    )
    
    if success:
        logger.info("Model training completed successfully")
    else:
        logger.error("Model training failed")

if __name__ == "__main__":
    main()
