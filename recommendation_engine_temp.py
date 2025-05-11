"""
Recommendation Engine module for the Emotion-Based Video Recommendation system.
Combines emotion data with video content preferences to provide personalized recommendations.
"""

import random
from datetime import datetime, timedelta
import streamlit as st

from database_utils import get_geolocation
from emotion_constants import EMOTION_VIDEO_MAPPING
from config import YOUTUBE_API_KEY

class RecommendationEngine:
    """
    Class for generating personalized recommendations based on user emotions.
    Combines YouTube recommendations with Gemini explanations.
    """
    
    def __init__(self, youtube_recommender, gemini_explainer):
        """
        Initialize the recommendation engine with recommender components.
        
        Args:
            youtube_recommender: An instance of YouTubeRecommender
            gemini_explainer: An instance of GeminiExplainer
        """
        self.youtube = youtube_recommender
        self.gemini = gemini_explainer
        self.cached_recommendations = {}
    
    def get_recommendations_for_user(self, user_name, emotion_data):
        """
        Generate personalized video recommendations based on emotion data.
        
        Args:
            user_name (str): Name of the user
            emotion_data (dict): Emotion data for the user
            
        Returns:
            list: List of personalized video recommendations with explanations
        """
        # Check cache first (for returning users)
        if user_name in self.cached_recommendations:
            cache_time, recommendations = self.cached_recommendations[user_name]
            # Cache valid for 1 hour
            if datetime.now() - cache_time < timedelta(hours=1):
                return recommendations
        
        if not emotion_data:
            st.warning(f"No emotion data available for {user_name}")
            return []
            
        # Extract data
        dominant_emotion = emotion_data['emotion'].lower()
        current_hour = datetime.now().hour
        
        # First, try to get neural network recommendations if model is ready
        use_neural_network = False
        if 'neural_recommender' in st.session_state:
            neural_model = st.session_state.neural_recommender
            if neural_model.is_ready():
                use_neural_network = True
                
        if use_neural_network:
            # Get recommendations from neural network
            category_predictions = neural_model.predict_categories(dominant_emotion, current_hour)
            
            # If we have useful predictions
            if category_predictions and len(category_predictions) > 0:
                st.sidebar.success("🧠 Using neural network recommendations")
                
                all_recommendations = []
                # Get top 3 predicted categories with their scores
                top_categories = category_predictions[:3]
                
                # Display the top predicted categories
                st.sidebar.markdown("**Top predicted categories:**")
                for category, score in top_categories:
                    st.sidebar.markdown(f"- {category}: {score:.2f}")
                    
                st.sidebar.markdown("---")
                
                # Get geolocation for regional content
                city, country = get_geolocation()
                
                # Get emotion settings for keywords
                emotion_settings = EMOTION_VIDEO_MAPPING.get(dominant_emotion, EMOTION_VIDEO_MAPPING["neutral"])
                
                # For each predicted category, get videos
                for category_name, confidence in top_categories:
                    # Only use categories with reasonable confidence
                    if confidence > 0.1:
                        # Try each keyword 
                        for keyword in emotion_settings['keywords']:
                            full_query = f"{keyword} {category_name}"
                            
                            # Sometimes add location-specific content
                            if country and country != "Unknown Country" and random.random() > 0.5:
                                full_query += f" {country}"
                                
                            # Search videos
                            videos = self.youtube.search_videos(
                                query=full_query,
                                max_results=5
                            )
                            
                            # Add category name to each video
                            for video in videos:
                                video['nn_category'] = category_name
                                video['nn_confidence'] = float(confidence)
                                video['is_neural_rec'] = True
                                
                            # Add to recommendations
                            all_recommendations.extend(videos)
                
                # If we have recommendations from NN, use them
                if all_recommendations:
                    # Deduplicate by video ID
                    unique_videos = {}
                    for video in all_recommendations:
                        if video['id'] not in unique_videos:
                            unique_videos[video['id']] = video
                            
                    # Get a list of videos, limited to 10
                    recommendations = list(unique_videos.values())[:10]
                    
                    # Add explanations
                    recommendations_with_explanations = self._add_explanations(
                        recommendations, dominant_emotion, user_name
                    )
                    
                    # Cache recommendations
                    self.cached_recommendations[user_name] = (datetime.now(), recommendations_with_explanations)
                    return recommendations_with_explanations
        
        # Fall back to regular recommendations if neural network didn't produce results
        st.sidebar.info("Using traditional recommendation model")
        
        # Get settings for the emotion
        emotion_settings = EMOTION_VIDEO_MAPPING.get(dominant_emotion, EMOTION_VIDEO_MAPPING["neutral"])
        
        # Get geolocation for regional content
        city, country = get_geolocation()
        
        # Get a mix of videos from all keywords
        all_videos = []
        
        for category in emotion_settings['categories']:
            # For each keyword
            for keyword in emotion_settings['keywords']:
                full_query = f"{keyword} {category}"
                
                # Sometimes add location-specific content
                if country and country != "Unknown Country" and random.random() > 0.7:
                    full_query += f" {country}"
                
                videos = self.youtube.search_videos(
                    query=full_query, 
                    max_results=3
                )
                
                all_videos.extend(videos)
        
        # Shuffle and limit
        random.shuffle(all_videos)
        recommendations = all_videos[:10]
        
        # Add explanations
        recommendations_with_explanations = self._add_explanations(
            recommendations, dominant_emotion, user_name
        )
        
        # Cache recommendations
        self.cached_recommendations[user_name] = (datetime.now(), recommendations_with_explanations)
        
        return recommendations_with_explanations
    
    def _add_explanations(self, videos, emotion, user_name):
        """
        Add personalized explanations to the recommended videos.
        
        Args:
            videos (list): List of video recommendations
            emotion (str): User's current dominant emotion
            user_name (str): Name of the user
            
        Returns:
            list: Videos with personalized explanations
        """
        # For each video, get an explanation
        for video in videos:
            # Check if it's a neural recommendation
            if 'is_neural_rec' in video and video['is_neural_rec']:
                # Special explanation for neural recommendations
                category = video.get('nn_category', 'this category')
                confidence = video.get('nn_confidence', 0) * 100  # Convert to percentage
                
                explanation = self.gemini.explain_recommendation(
                    video_title=video['title'],
                    video_category=category,
                    user_emotion=emotion,
                    user_name=user_name,
                    neural_rec=True,
                    confidence=confidence
                )
            else:
                # Regular explanation
                explanation = self.gemini.explain_recommendation(
                    video_title=video['title'],
                    video_category=video['category_name'],
                    user_emotion=emotion,
                    user_name=user_name
                )
                
            video['explanation'] = explanation
            
        return videos
