"""
Local LLM Explainer module for the Emotion-Based Video Recommendation system.
Uses a local transformer model to generate personalized explanations for video recommendations.
"""

import streamlit as st
import os
import time
from emotion_constants import EMOTION_DESCRIPTIONS

# Import required libraries only when needed to avoid loading them during import
def load_transformer_libraries():
    """Load transformer libraries only when needed"""
    try:
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        return True
    except ImportError:
        st.warning("Transformer libraries not found. Installing required packages...")
        return False

class LocalExplainer:
    """Class for generating personalized recommendation explanations using a local LLM."""
    
    def __init__(self):
        """Initialize the local explainer with a small transformer model."""
        self.model = None
        self.tokenizer = None
        self.available = False
        self.explanation_cache = {}
        self.explanation_templates = {
            'happy': [
                "This uplifting content matches your positive mood and might boost your happiness even more.",
                "Since you're feeling happy, you might enjoy this cheerful content.",
                "Perfect choice for your current joyful state!",
                "This aligns well with your happy mood right now."
            ],
            'sad': [
                "This content might help lift your spirits when you're feeling down.",
                "When you're feeling sad, this type of video can provide comfort.",
                "This may offer a pleasant distraction from feeling blue.",
                "A good match for when you need something uplifting."
            ],
            'angry': [
                "This could help you channel your energy into something positive.",
                "When feeling frustrated, this content might provide a helpful perspective.",
                "A calming choice that might help balance your current mood.",
                "This might help shift your focus to something more enjoyable."
            ],
            'fear': [
                "This reassuring content might help ease your worries.",
                "When feeling anxious, this type of video can provide comfort.",
                "Something engaging to help distract from stressful thoughts.",
                "A good choice to help you feel more secure and relaxed."
            ],
            'surprise': [
                "This fascinating content complements your curious mood.",
                "Since you're feeling intrigued, you might enjoy this interesting topic.",
                "Perfect for your current state of wonder and curiosity!",
                "This matches well with your open and receptive mindset."
            ],
            'disgust': [
                "This pleasant content offers a refreshing change of perspective.",
                "When you need a palate cleanser, this type of video works well.",
                "Something positive to help shift your mood in a better direction.",
                "A good choice to help replace negative feelings with positive ones."
            ],
            'neutral': [
                "This engaging content might pique your interest.",
                "A good match for your current receptive state of mind.",
                "This balanced content aligns with your current neutral mood.",
                "Something interesting that might enhance your viewing experience."
            ]
        }
        
        # Try to load the model (will do lazy loading)
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the local LLM model if needed."""
        if load_transformer_libraries():
            try:
                import torch
                from transformers import pipeline
                import os
                
                # First check if we can use a GPU
                device = 0 if torch.cuda.is_available() else -1
                
                # Define a local cache directory to store model
                cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tinyllama_cache")
                
                # Make sure the directory exists
                os.makedirs(cache_dir, exist_ok=True)
                
                # Check if model files are in the cache directory
                model_dir_path = os.path.join(cache_dir, "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")
                has_cached_model = os.path.exists(model_dir_path) and any(
                    os.path.exists(os.path.join(root, f))
                    for root, _, files in os.walk(model_dir_path)
                    for f in files
                    if f.endswith('.safetensors') or f.endswith('.bin')
                )
                
                # Display appropriate message based on whether model is already cached
                if has_cached_model:
                    st.info("Using locally cached LLM model...")
                else:
                    st.info("Downloading local LLM model (only needed once)...")
                    st.warning("This may take a few minutes on the first run.")
                
                # Use a small model that can run on CPU if needed
                # TinyLlama is only 1.1GB and can run on modest hardware
                self.generator = pipeline(
                    'text-generation',
                    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                    device=device,
                    cache_dir=cache_dir
                )
                
                self.available = True
                st.success("Local LLM model loaded successfully!")
                
                # Display cache location
                st.info(f"Model is cached at: {cache_dir}")
                
                # Indicate whether GPU acceleration is being used
                if torch.cuda.is_available():
                    st.success(f"Using GPU acceleration: {torch.cuda.get_device_name(0)}")
                else:
                    st.info("Running on CPU (no GPU acceleration available)")
                    
            except Exception as e:
                st.error(f"Could not load local LLM: {e}")
                self.available = False
        else:
            self.available = False
            
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
            
        try:
            # First try using the transformer model if available
            if self.available:
                import torch
                
                # Create prompt for the model
                prompt = f"""<|im_start|>system
You are an AI assistant helping explain video recommendations to users.
<|im_end|>
<|im_start|>user
Write a brief explanation of why this video might benefit a user who is feeling {emotion}.

Video Title: {video['title']}
Video Channel: {video['channel']}

Make the explanation personal, supportive, and under 20 words.
<|im_end|>
<|im_start|>assistant
"""
                
                # Generate explanation from local model
                start_time = time.time()
                result = self.generator(
                    prompt,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    num_return_sequences=1
                )
                generation_time = time.time() - start_time
                
                # Extract the response
                explanation = result[0]['generated_text'].split("<|im_start|>assistant")[-1].strip()
                explanation = explanation.replace("<|im_end|>", "").strip()
                
                # Clean up explanation
                if len(explanation) > 100:
                    explanation = explanation[:97] + "..."
                    
                if "<|" in explanation:  # If format markers remain, clean them
                    explanation = explanation.split("<|")[0].strip()
                
                # Cache the explanation
                self.explanation_cache[cache_key] = explanation
                
                return explanation
            
            # Fall back to template-based explanations
            else:
                return self.get_template_based_explanation(video, user_name, emotion)
                
        except Exception as e:
            print(f"Local LLM error: {e}")
            return self.get_template_based_explanation(video, user_name, emotion)
    
    def get_template_based_explanation(self, video, user_name, emotion):
        """Generate an explanation using templates when LLM is not available."""
        import random
        
        # Get templates for this emotion, or use neutral if not found
        templates = self.explanation_templates.get(emotion, self.explanation_templates['neutral'])
        
        # Select a random template
        explanation = random.choice(templates)
        
        # Add video context sometimes
        if random.random() > 0.5:
            category_phrases = {
                'Music': ['music', 'song', 'melody', 'tune', 'beat'],
                'Sports': ['sports content', 'athletic performance', 'game highlights', 'sports clip'],
                'Gaming': ['gaming content', 'gameplay', 'gaming stream', 'gaming video'],
                'Education': ['educational content', 'informative video', 'learning material', 'tutorial'],
                'Comedy': ['comedy', 'humorous video', 'funny content', 'comedic clip']
            }
            # Get phrase for category if available
            category = video.get('category', '')
            phrases = category_phrases.get(category, [category.lower() + ' content'])
            phrase = random.choice(phrases)
            
            # Add phrase to explanation 50% of the time
            if random.random() > 0.5:
                explanation = f"This {phrase} " + explanation.lower()
        
        return explanation
