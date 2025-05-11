"""
Database utilities module for the Emotion-Based Video Recommendation system.
Handles all database operations, including logging emotions and video interactions.
"""

import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

from config import DATABASE_PATH

def get_geolocation():
    """
    Get geolocation information for the current user.
    
    In a production application, this would use IP geolocation or user input.
    For simplicity, this function returns default values.
    
    Returns:
        tuple: A tuple containing (city, country)
    """
    # In a real app, this would use IP geolocation or user input
    # For simplicity, we'll return default values
    return "Unknown City", "Unknown Country"

def ensure_emotion_logs_schema():
    """
    Ensure the emotion_logs table has the correct schema.
    If the table exists but has an outdated schema, it will be updated.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='emotion_logs';")
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Check the schema of the existing table
            cursor.execute("PRAGMA table_info(emotion_logs);")
            columns = [row[1] for row in cursor.fetchall()]

            if 'input' not in columns:
                # If the 'input' column is missing, update the schema
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotion_logs_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    emotion TEXT,
                    input TEXT,
                    timestamp INTEGER,
                    city TEXT,
                    country TEXT
                );
                ''')

                # Copy data from the old table to the new table
                cursor.execute('''
                INSERT INTO emotion_logs_new (id, name, emotion, timestamp, city, country)
                SELECT id, name, emotion, timestamp, city, country FROM emotion_logs;
                ''')

                # Drop the old table and rename the new table
                cursor.execute("DROP TABLE emotion_logs;")
                cursor.execute("ALTER TABLE emotion_logs_new RENAME TO emotion_logs;")

        else:
            # Create the table if it does not exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                emotion TEXT,
                input TEXT,
                timestamp INTEGER,
                city TEXT,
                country TEXT
            );
            ''')

        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error ensuring emotion_logs schema: {e}")

def log_emotion_data(name, emotion, input_text, timestamp, city, country):
    """
    Log emotion data to the emotion_logs table.
    
    Args:
        name (str): The name of the user.
        emotion (str): The detected emotion.
        input_text (str): Any text input provided by the user.
        timestamp (int): The Unix timestamp when the emotion was detected.
        city (str): The city location of the user.
        country (str): The country location of the user.
        
    Returns:
        bool: True if logging was successful, False otherwise.
    """
    try:
        # First, ensure the table has the correct schema
        ensure_emotion_logs_schema()
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Insert data into emotion_logs table
        cursor.execute('''
        INSERT INTO emotion_logs (name, emotion, input, timestamp, city, country)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, emotion, input_text, timestamp, city, country))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def log_video_interaction(name, video_genre, emotion, timestamp, city, country):
    """
    Log video interaction data to the video_interactions table.
    
    Args:
        name (str): The name of the user.
        video_genre (str): The genre or category of the video.
        emotion (str): The emotion of the user when watching the video.
        timestamp (int): The Unix timestamp when the interaction occurred.
        city (str): The city location of the user.
        country (str): The country location of the user.
        
    Returns:
        bool: True if logging was successful, False otherwise.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Create video_interactions table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            video_genre TEXT,
            emotion TEXT,
            timestamp INTEGER,
            city TEXT,
            country TEXT
        )
        ''')

        # Insert data into video_interactions table
        cursor.execute('''
        INSERT INTO video_interactions (name, video_genre, emotion, timestamp, city, country)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, video_genre, emotion, timestamp, city, country))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def log_video_click(name, video_id, video_title, video_category, emotion, timestamp, city, country):
    """
    Log when a user clicks on a recommended video.
    
    Args:
        name (str): The name of the user.
        video_id (str): The YouTube video ID.
        video_title (str): The title of the video.
        video_category (str): The category or genre of the video.
        emotion (str): The emotion of the user when clicking the video.
        timestamp (int): The Unix timestamp when the click occurred.
        city (str): The city location of the user.
        country (str): The country location of the user.
        
    Returns:
        bool: True if logging was successful, False otherwise.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Create video_clicks table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_clicks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            video_id TEXT,
            video_title TEXT,
            video_category TEXT,
            emotion TEXT,
            timestamp INTEGER,
            city TEXT,
            country TEXT
        )
        ''')

        # Insert data into video_clicks table
        cursor.execute('''
        INSERT INTO video_clicks (name, video_id, video_title, video_category, emotion, timestamp, city, country)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, video_id, video_title, video_category, emotion, timestamp, city, country))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error logging video click: {e}")
        return False

def train_video_recommendation_model():
    """
    Train a machine learning model using data from both emotion logs and video interactions.
    
    Returns:
        model: A trained RandomForestClassifier model if successful, None otherwise.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)

        # Load data from emotion_logs table
        emotion_data = pd.read_sql_query('SELECT * FROM emotion_logs', conn)

        # Load data from video_interactions table
        video_data = pd.read_sql_query('SELECT * FROM video_interactions', conn)

        conn.close()

        # Merge the two datasets on common columns (e.g., name, emotion, timestamp)
        merged_data = pd.merge(emotion_data, video_data, on=['name', 'emotion', 'timestamp', 'city', 'country'], how='inner')

        # Preprocess data for training
        merged_data['features'] = merged_data['emotion'] + ' ' + merged_data['video_genre'] + ' ' + merged_data['city'] + ' ' + merged_data['country']
        X = TfidfVectorizer().fit_transform(merged_data['features'])
        y = merged_data['video_genre']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")

        return model
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None
