"""
ML-based recommendation engine for the Emotion-Based Video Recommendation system.
Uses historical user interaction data to provide personalized recommendations.
"""

import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DATABASE_PATH

class MLRecommendationEngine:
    """
    Machine learning based recommendation engine that uses historical user interactions
    to provide personalized video recommendations.
    """
    
    def __init__(self, database_path=DATABASE_PATH):
        """
        Initialize the ML recommendation engine.
        
        Args:
            database_path (str): Path to the SQLite database
        """
        self.database_path = database_path
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.user_profiles = {}
        
    def _load_user_data(self, user_name):
        """
        Load historical data for a user from the database.
        
        Args:
            user_name (str): Name of the user
            
        Returns:
            pandas.DataFrame: DataFrame containing user's interaction data
        """
        import sqlite3
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Check if the video_clicks table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='video_clicks';")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # If table doesn't exist, return empty DataFrames
            conn.close()
            return pd.DataFrame(), pd.DataFrame()
        
        # Load video clicks data
        query = f"""
        SELECT * FROM video_clicks 
        WHERE name = '{user_name}'
        ORDER BY timestamp DESC
        """
        
        try:
            clicks_data = pd.read_sql_query(query, conn)
        except sqlite3.OperationalError:
            clicks_data = pd.DataFrame()
        
        # Load emotion logs data
        query = f"""
        SELECT * FROM emotion_logs 
        WHERE name = '{user_name}'
        ORDER BY timestamp DESC
        """
        
        try:
            emotion_data = pd.read_sql_query(query, conn)
        except sqlite3.OperationalError:
            emotion_data = pd.DataFrame()
        
        conn.close()
        
        return clicks_data, emotion_data
        
    def build_user_profile(self, user_name):
        """
        Build a profile for a user based on their historical interactions.
        
        Args:
            user_name (str): Name of the user
            
        Returns:
            dict: User profile with feature vectors
        """
        clicks_data, emotion_data = self._load_user_data(user_name)
        
        if clicks_data.empty:
            # If no click data, we can't build a profile
            return None
        
        # Extract features from click data
        # Combine text features for vectorization
        clicks_data['text_features'] = (
            clicks_data['video_title'] + ' ' + 
            clicks_data['video_category'] + ' ' + 
            clicks_data['emotion']
        )
        
        # Create a matrix of TF-IDF features
        if len(clicks_data) > 0:
            try:
                # Make sure the vectorizer is initialized
                if not hasattr(self, 'vectorizer') or self.vectorizer is None:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    self.vectorizer = TfidfVectorizer(stop_words='english')
                
                text_features = self.vectorizer.fit_transform(clicks_data['text_features'])
                
                # Create a weighted average based on recency (more recent = higher weight)
                # Convert timestamps to weights
                max_timestamp = clicks_data['timestamp'].max()
                min_timestamp = clicks_data['timestamp'].min()
                range_timestamp = max(1, max_timestamp - min_timestamp)  # Avoid division by zero
                
                weights = clicks_data['timestamp'].apply(
                    lambda x: 0.5 + 0.5 * (x - min_timestamp) / range_timestamp
                )
                
                # Apply weights to features
                weighted_features = []
                for i, weight in enumerate(weights):
                    weighted_features.append(text_features[i].toarray() * weight)
                
                # Combine into a user profile vector (average of weighted features)
                profile_vector = np.mean(weighted_features, axis=0)
                
                # Store user profile
                self.user_profiles[user_name] = {
                    'profile_vector': profile_vector,
                    'frequent_emotions': clicks_data['emotion'].value_counts().to_dict(),
                    'last_updated': time.time()
                }
                
                return self.user_profiles[user_name]
            except Exception as e:
                print(f"Error building user profile: {e}")
                return None
        else:
            return None
    
    def get_video_features(self, video):
        """
        Extract features from a video for recommendation matching.
        
        Args:
            video (dict): Dictionary containing video metadata
            
        Returns:
            numpy.ndarray: Feature vector for the video
        """
        # Make sure the vectorizer is initialized
        if not hasattr(self, 'vectorizer') or self.vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words='english')
            # If vectorizer not fit, we can't extract features
            return np.zeros((1, 1))  # Return a dummy vector
        
        try:
            # Combine text features
            text_features = f"{video['title']} {video['channel']} {video['description']}"
            
            # Vectorize
            feature_vector = self.vectorizer.transform([text_features]).toarray()
            
            return feature_vector
        except Exception as e:
            print(f"Error extracting video features: {e}")
            return np.zeros((1, 1))  # Return a dummy vector
        
    def rank_recommendations(self, user_name, videos, current_emotion=None):
        """
        Rank videos based on user profile and current emotion.
        
        Args:
            user_name (str): Name of the user
            videos (list): List of video dictionaries to rank
            current_emotion (str, optional): User's current emotion
            
        Returns:
            list: Ranked list of videos or None if ranking isn't possible
        """
        # Make sure we have a user profile
        if user_name not in self.user_profiles:
            profile = self.build_user_profile(user_name)
            if not profile:
                # If we can't build a profile, return None to indicate
                # that ML ranking isn't possible yet
                return None
        else:
            profile = self.user_profiles[user_name]
        
        try:
            # Extract user profile vector
            user_vector = profile['profile_vector']
            
            # Initialize vectorizer if needed
            if not hasattr(self, 'vectorizer') or self.vectorizer is None:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = TfidfVectorizer(stop_words='english')
                # We need some data to fit the vectorizer
                return None
            
            # Calculate similarity scores for each video
            scored_videos = []
            for video in videos:
                # Get video features
                video_features = self.get_video_features(video)
                
                # Calculate cosine similarity between user profile and video
                similarity = cosine_similarity(user_vector.reshape(1, -1), video_features)[0][0]
                
                # Apply emotion boost if current emotion matches frequently clicked emotions
                emotion_boost = 1.0
                if current_emotion and 'frequent_emotions' in profile:
                    frequent_emotions = profile['frequent_emotions']
                    if current_emotion in frequent_emotions:
                        # Boost based on frequency of this emotion in user history
                        total = sum(frequent_emotions.values())
                        emotion_boost = 1.0 + (frequent_emotions[current_emotion] / total)
                
                # Calculate final score
                final_score = similarity * emotion_boost
                
                # Store video with score
                scored_videos.append((video, final_score))
            
            # Sort by score (descending)
            scored_videos.sort(key=lambda x: x[1], reverse=True)
            
            # Extract just the videos in ranked order
            ranked_videos = [video for video, _ in scored_videos]
            
            return ranked_videos
        except Exception as e:
            print(f"Error ranking recommendations: {e}")
            return None
