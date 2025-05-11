"""
Gemini Explainer module for the Emotion-Based Video Recommendation system.
Uses Google's Gemini AI to generate personalized explanations for video recommendations.
"""

import streamlit as st
import google.generativeai as genai
from emotion_constants import EMOTION_DESCRIPTIONS
from config import GEMINI_API_KEY

class GeminiExplainer:
    """Class for generating personalized recommendation explanations using Gemini AI."""
    
    def __init__(self, api_key):
        """
        Initialize the Gemini explainer with the API key.
        
        Args:
            api_key (str): Google Gemini API key
        """
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            self.available = True
        except Exception as e:
            st.error(f"Error initializing Gemini: {e}")
            self.available = False
        
        # Cache for explanations to avoid duplicate API calls
        self.explanation_cache = {}
            
    def explain_recommendation(self, video, user_name, emotion):
        """
        Generate explanation for why a video is recommended based on user emotion.
        
        Args:
            video (dict): Dictionary containing video metadata
            user_name (str): Name of the user
            emotion (str): Detected emotion of the user
            
        Returns:
            str: Personalized explanation for the video recommendation
        """
        # Create cache key based on video ID and emotion
        cache_key = f"{video['id']}_{emotion}"
        
        # Return cached explanation if available
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        if not self.available:
            return "Video recommended based on your current emotion."
            
        try:
            # Create prompt for Gemini
            prompt = f"""
            You are an AI assistant helping explain video recommendations to users.
            
            USER: {user_name}
            DETECTED EMOTION: {emotion}
            EMOTION CONTEXT: {EMOTION_DESCRIPTIONS.get(emotion, "You might enjoy content tailored to your current mood.")}
            
            VIDEO DETAILS:
            Title: {video['title']}
            Channel: {video['channel']}
            Description: {video['description'][:200]}...
            
            Explain in 1-2 brief, personalized sentences why this video might benefit the user given their current emotional state.
            Keep your explanation friendly, supportive, and under 30 words.
            Start directly with the explanation without phrases like "This video is recommended because".
            """
            
            # Generate explanation from Gemini
            response = self.model.generate_content(prompt)
            explanation = response.text.strip()
            
            # Clean up explanation
            explanation = explanation.replace('"', '')
            if len(explanation) > 100:
                explanation = explanation[:97] + "..."
                
            # Cache the explanation
            self.explanation_cache[cache_key] = explanation
            
            return explanation
            
        except Exception as e:
            print(f"Gemini error: {e}")
            return "Video recommended based on your current emotion."
