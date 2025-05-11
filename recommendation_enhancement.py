"""
Recommendation Enhancement module for the Emotion-Based Video Recommendation system.
Provides advanced features to strengthen the recommendation logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
import json
import random

from config import DATABASE_PATH
from emotion_constants import EMOTION_VIDEO_MAPPING

class RecommendationEnhancer:
    """
    Class to enhance the recommendation engine with advanced features:
    1. Time-of-day awareness
    2. Collaborative filtering
    3. Content diversity scoring
    4. User context awareness
    5. Feedback-based learning
    """
    
    def __init__(self, database_path=DATABASE_PATH):
        """Initialize the recommendation enhancer."""
        self.database_path = database_path
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Cache for similarity calculations
        self.similarity_cache = {}
        # Time segments for time-of-day awareness
        self.time_segments = {
            "morning": (5, 11),    # 5 AM - 11:59 AM
            "afternoon": (12, 17), # 12 PM - 5:59 PM
            "evening": (18, 21),   # 6 PM - 9:59 PM
            "night": (22, 4)       # 10 PM - 4:59 AM
        }
    
    def time_of_day_score(self, video, current_hour=None):
        """
        Score videos based on time of day appropriateness.
        
        Args:
            video (dict): Video information
            current_hour (int, optional): Current hour (24h format). Defaults to current time.
            
        Returns:
            float: Time-of-day score between 0.0 and 1.0
        """
        if current_hour is None:
            current_hour = datetime.now().hour
        
        # Determine current time segment
        current_segment = "afternoon"  # Default
        for segment, (start, end) in self.time_segments.items():
            if start <= end:
                if start <= current_hour <= end:
                    current_segment = segment
                    break
            else:  # Handles night case where start > end (e.g., 22-4)
                if current_hour >= start or current_hour <= end:
                    current_segment = segment
                    break
        
        # Keywords that work well for different times of day
        time_keywords = {
            "morning": ["energizing", "motivational", "fresh", "start", "morning"],
            "afternoon": ["relaxing", "inspiring", "educational", "informative"],
            "evening": ["entertaining", "comedy", "relaxing", "wind down", "recap"],
            "night": ["calming", "sleep", "peaceful", "ambient", "meditation"]
        }
        
        # Check if video title or description contains relevant keywords
        score = 0.1  # Base score
        
        # Check title and description for time-specific keywords
        video_text = (video.get('title', '') + ' ' + video.get('description', '')).lower()
        
        for keyword in time_keywords[current_segment]:
            if keyword.lower() in video_text:
                score += 0.2
                
        # Cap at 1.0
        return min(score, 1.0)
    
    def collaborative_filtering(self, user_name, videos, top_n=5):
        """
        Apply collaborative filtering to find similar users and their liked videos.
        
        Args:
            user_name (str): Current user name
            videos (list): List of candidate videos
            top_n (int): Number of similar users to consider
            
        Returns:
            list: Videos reordered based on collaborative filtering
        """
        # Connect to database
        conn = sqlite3.connect(self.database_path)
        
        # Get current user's video interactions
        query = f"""
        SELECT video_id, video_title, video_category, emotion
        FROM video_clicks 
        WHERE name = '{user_name}'
        ORDER BY timestamp DESC
        """
        try:
            current_user_data = pd.read_sql_query(query, conn)
        except Exception:
            conn.close()
            return videos  # Return original videos if error
            
        if current_user_data.empty:
            conn.close()
            return videos  # Not enough data for collaborative filtering
        
        # Find other users with similar tastes
        query = """
        SELECT DISTINCT name FROM video_clicks 
        WHERE name != ?
        """
        try:
            all_users = pd.read_sql_query(query, conn, params=(user_name,))
        except Exception:
            conn.close()
            return videos
            
        if all_users.empty:
            conn.close()
            return videos
            
        # Calculate user similarity based on video interaction overlap
        user_similarities = {}
        for other_user in all_users['name'].unique():
            query = f"""
            SELECT video_id, video_title, video_category, emotion
            FROM video_clicks 
            WHERE name = '{other_user}'
            """
            other_user_data = pd.read_sql_query(query, conn)
            
            # Calculate overlap in video_ids
            current_user_videos = set(current_user_data['video_id'])
            other_user_videos = set(other_user_data['video_id'])
            
            # Jaccard similarity for video overlap
            intersection = len(current_user_videos.intersection(other_user_videos))
            union = len(current_user_videos.union(other_user_videos))
            
            if union > 0:
                similarity = intersection / union
                # Also consider emotion agreement
                emotion_agreement = 0
                for video_id in current_user_videos.intersection(other_user_videos):
                    current_emotion = current_user_data[current_user_data['video_id'] == video_id]['emotion'].iloc[0]
                    other_emotion = other_user_data[other_user_data['video_id'] == video_id]['emotion'].iloc[0]
                    if current_emotion == other_emotion:
                        emotion_agreement += 1
                
                if intersection > 0:
                    emotion_agreement = emotion_agreement / intersection
                    similarity = similarity * 0.7 + emotion_agreement * 0.3
                
                user_similarities[other_user] = similarity
        
        # Get top N similar users
        similar_users = sorted(user_similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # If we have similar users, get their liked videos
        if similar_users:
            # Collect videos liked by similar users
            similar_user_videos = {}
            for similar_user, similarity in similar_users:
                query = f"""
                SELECT video_id, COUNT(*) as click_count
                FROM video_clicks 
                WHERE name = '{similar_user}'
                GROUP BY video_id
                ORDER BY click_count DESC
                LIMIT 20
                """
                user_vids = pd.read_sql_query(query, conn)
                
                for _, row in user_vids.iterrows():
                    video_id = row['video_id']
                    count = row['click_count']
                    if video_id not in similar_user_videos:
                        similar_user_videos[video_id] = 0
                    similar_user_videos[video_id] += count * similarity
        
            conn.close()
            
            # Boost scores for videos that appear in similar users' history
            for i, video in enumerate(videos):
                if video['id'] in similar_user_videos:
                    # Mark this video as collaborative recommendation
                    video['collab_score'] = similar_user_videos[video['id']]
            
            # Sort videos based on a weighted score
            videos.sort(key=lambda x: x.get('collab_score', 0), reverse=True)
            
            # Return a mix of collaborative and original recommendations
            if len(videos) >= 6:
                # Take top 3 collaborative videos and mix with others
                collab_videos = videos[:3]
                other_videos = videos[3:]
                random.shuffle(other_videos)
                return collab_videos + other_videos
        
        return videos
    
    def ensure_content_diversity(self, videos, min_categories=3):
        """
        Ensure content diversity by including videos from different categories.
        
        Args:
            videos (list): List of video recommendations
            min_categories (int): Minimum number of different categories to include
            
        Returns:
            list: Reordered videos for better diversity
        """
        if not videos or len(videos) < min_categories:
            return videos
            
        # Extract categories
        categories = {}
        for video in videos:
            category = video.get('nn_category', video.get('video_category', 'Unknown'))
            if category not in categories:
                categories[category] = []
            categories[category].append(video)
        
        # If we already have enough diversity, return original list
        if len(categories) >= min_categories:
            return videos
            
        # Otherwise, try to ensure diversity by picking from each category
        result = []
        # Round-robin selection from categories
        while result_len := len(result) < len(videos):
            for category in list(categories.keys()):
                if categories[category]:
                    result.append(categories[category].pop(0))
                if len(result) >= len(videos):
                    break
                    
            # If we've taken all videos from all categories, break
            if result_len == len(result):
                break
                
        return result
        
    def get_user_context(self, user_name):
        """
        Get contextual information about the user to improve recommendations.
        
        Args:
            user_name (str): Name of the user
            
        Returns:
            dict: Context information
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        context = {
            'preferred_time': None,
            'favorite_categories': [],
            'emotional_patterns': {},
            'watch_duration_pattern': None
        }
        
        # Get preferred time of day
        try:
            cursor.execute("""
            SELECT strftime('%H', datetime(timestamp, 'unixepoch', 'localtime')) as hour,
                   COUNT(*) as count
            FROM video_clicks
            WHERE name = ?
            GROUP BY hour
            ORDER BY count DESC
            LIMIT 1
            """, (user_name,))
            result = cursor.fetchone()
            if result:
                hour, _ = result
                hour = int(hour)
                if 5 <= hour < 12:
                    context['preferred_time'] = 'morning'
                elif 12 <= hour < 18:
                    context['preferred_time'] = 'afternoon'
                elif 18 <= hour < 22:
                    context['preferred_time'] = 'evening'
                else:
                    context['preferred_time'] = 'night'
        except Exception:
            pass
            
        # Get favorite categories
        try:
            cursor.execute("""
            SELECT video_category, COUNT(*) as count
            FROM video_clicks
            WHERE name = ?
            GROUP BY video_category
            ORDER BY count DESC
            LIMIT 3
            """, (user_name,))
            results = cursor.fetchall()
            context['favorite_categories'] = [category for category, _ in results]
        except Exception:
            pass
            
        # Get emotional patterns
        try:
            cursor.execute("""
            SELECT emotion, COUNT(*) as count
            FROM emotion_logs
            WHERE name = ?
            GROUP BY emotion
            ORDER BY count DESC
            """, (user_name,))
            results = cursor.fetchall()
            total = sum(count for _, count in results)
            if total > 0:
                context['emotional_patterns'] = {
                    emotion: count / total for emotion, count in results
                }
        except Exception:
            pass
        
        conn.close()
        return context
        
    def rerank_recommendations(self, user_name, videos, current_emotion=None, current_hour=None):
        """
        Rerank recommendations using all enhancer features.
        
        Args:
            user_name (str): Name of the user
            videos (list): List of video recommendations
            current_emotion (str, optional): Current emotion
            current_hour (int, optional): Current hour
            
        Returns:
            list: Reranked video recommendations
        """
        if not videos:
            return videos
            
        # Apply collaborative filtering
        videos = self.collaborative_filtering(user_name, videos)
        
        # Apply time of day scoring
        for video in videos:
            video['time_score'] = self.time_of_day_score(video, current_hour)
            
        # Get user context
        context = self.get_user_context(user_name)
        
        # Calculate final score for each video
        for video in videos:
            base_score = 1.0
            
            # Neural network score if available
            nn_confidence = video.get('nn_confidence', 0.5)
            
            # Time score
            time_score = video.get('time_score', 0.5)
            
            # Collaborative score
            collab_score = min(video.get('collab_score', 0) / 5.0, 1.0)
            
            # Category preference score
            category_score = 0.5
            video_category = video.get('nn_category', video.get('video_category', 'Unknown'))
            if video_category in context['favorite_categories']:
                position = context['favorite_categories'].index(video_category)
                category_score = 1.0 - (position * 0.2)  # 1.0, 0.8, 0.6 for top 3
                
            # Final score calculation (weighted average)
            video['final_score'] = (
                nn_confidence * 0.35 +
                time_score * 0.15 +
                collab_score * 0.25 +
                category_score * 0.25
            )
        
        # Sort by final score
        videos.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Ensure diversity in top results
        top_videos = videos[:6]
        rest_videos = videos[6:]
        
        diverse_top = self.ensure_content_diversity(top_videos)
        
        return diverse_top + rest_videos
