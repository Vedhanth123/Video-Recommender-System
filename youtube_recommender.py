"""
YouTube Recommender module for the Emotion-Based Video Recommendation system.
Handles interactions with the YouTube API to find videos based on emotion-based queries.
"""

import streamlit as st
from googleapiclient.discovery import build
from config import YOUTUBE_API_KEY, MAX_RESULTS

class YouTubeRecommender:
    """Class for handling YouTube API interactions and video recommendations."""
    
    def __init__(self, api_key):
        """
        Initialize the YouTube recommender with the API key.
        
        Args:
            api_key (str): YouTube API key for authentication
        """
        self.api_key = api_key
        try:
            self.youtube = build('youtube', 'v3', developerKey=api_key)
        except Exception as e:
            st.error(f"Error initializing YouTube API: {e}")
            self.youtube = None
            
    def validate_api(self):
        """
        Check if the YouTube API key is valid.
        
        Returns:
            bool: True if the API key is valid, False otherwise
        """
        if not self.youtube:
            return False
            
        try:
            # Make a simple API call to test the key
            self.youtube.videos().list(part='snippet', id='dQw4w9WgXcQ').execute()
            return True
        except Exception as e:
            st.error(f"YouTube API key validation failed: {e}")
            return False
        
    def search_videos(self, query, category_id=None, max_results=MAX_RESULTS):
        """
        Search for videos based on query and optional category.
        
        Args:
            query (str): Search query for finding videos
            category_id (str, optional): YouTube video category ID
            max_results (int, optional): Maximum number of results to return
            
        Returns:
            list: List of video dictionaries with metadata
        """
        if not self.youtube:
            st.error("YouTube API not available")
            return []
            
        try:
            search_params = {
                'q': query,
                'type': 'video',
                'part': 'snippet',
                'maxResults': max_results,
                'videoEmbeddable': 'true',
                'videoSyndicated': 'true',
                'videoDuration': 'medium'  # Medium length videos (4-20 minutes)
            }
            
            if category_id:
                search_params['videoCategoryId'] = category_id
                
            search_response = self.youtube.search().list(**search_params).execute()
            
            videos = []
            for item in search_response.get('items', []):
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                description = item['snippet']['description']
                thumbnail = item['snippet']['thumbnails']['medium']['url']
                channel = item['snippet']['channelTitle']
                
                # Get video statistics
                video_response = self.youtube.videos().list(
                    part='statistics,contentDetails',
                    id=video_id
                ).execute()
                
                if video_response['items']:
                    stats = video_response['items'][0]['statistics']
                    duration = video_response['items'][0]['contentDetails']['duration']
                    
                    videos.append({
                        'id': video_id,
                        'title': title,
                        'description': description,
                        'thumbnail': thumbnail,
                        'channel': channel,
                        'views': int(stats.get('viewCount', 0)),
                        'likes': int(stats.get('likeCount', 0)),
                        'duration': self._parse_duration(duration)
                    })
                    
            # Sort by views (popularity)
            videos.sort(key=lambda x: x['views'], reverse=True)
            return videos
            
        except Exception as e:
            st.error(f"Error searching YouTube: {e}")
            return []
            
    def _parse_duration(self, duration_str):
        """
        Convert ISO 8601 duration to human-readable format.
        
        Args:
            duration_str (str): ISO 8601 duration string from YouTube API
            
        Returns:
            str: Human-readable duration string (e.g., "3:45")
        """
        # Simple parsing of PT#M#S format
        minutes = 0
        seconds = 0
        
        if 'M' in duration_str:
            minutes_part = duration_str.split('M')[0]
            minutes = int(minutes_part.split('PT')[-1])
            
        if 'S' in duration_str:
            seconds_part = duration_str.split('S')[0]
            if 'M' in seconds_part:
                seconds = int(seconds_part.split('M')[-1])
            else:
                seconds = int(seconds_part.split('PT')[-1])
                
        return f"{minutes}:{seconds:02d}"


class EmotionVideoRecommender:
    """Class for using ML to recommend videos based on emotions and user history."""
    
    def __init__(self, database_path):
        """
        Initialize the emotion-based video recommender.
        
        Args:
            database_path (str): Path to the SQLite database
        """
        self.database_path = database_path
        self.model = None
        self.vectorizer = None
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()

    def load_data(self):
        """
        Load user interaction data from the database.
        
        Returns:
            pandas.DataFrame: DataFrame containing emotion logs
        """
        import sqlite3
        import pandas as pd
        conn = sqlite3.connect(self.database_path)
        query = "SELECT * FROM emotion_logs"
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data

    def preprocess_data(self, data):
        """
        Preprocess data for training.
        
        Args:
            data (pandas.DataFrame): DataFrame containing emotion logs
            
        Returns:
            tuple: (features, labels) preprocessed for ML training
        """
        data['text_features'] = data['emotion'] + ' ' + data['city'] + ' ' + data['country']
        X = self.vectorizer.fit_transform(data['text_features'])
        y = data['emotion']
        return X, y

    def train_model(self):
        """
        Train a machine learning model for emotion-based recommendations.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        data = self.load_data()
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")

    def recommend_videos(self, emotion, city, country):
        """
        Recommend videos based on the user's emotion and location.
        
        Args:
            emotion (str): User's detected emotion
            city (str): User's city location
            country (str): User's country location
            
        Returns:
            list: List of recommended videos
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.model:
            raise ValueError("Model is not trained. Please train the model first.")

        input_features = self.vectorizer.transform([f"{emotion} {city} {country}"])
        predicted_emotion = self.model.predict(input_features)[0]

        # Use the predicted emotion to fetch video recommendations
        youtube_recommender = YouTubeRecommender(YOUTUBE_API_KEY)
        videos = youtube_recommender.search_videos(query=predicted_emotion, max_results=10)
        return videos
