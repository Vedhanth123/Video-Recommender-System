# Local LLM Integration for Video Recommendations

This update replaces the paid Gemini API with a free local LLM solution for generating personalized video recommendation explanations.

## What's Changed

- **Removed Gemini API dependency**: We've eliminated the need to pay for the Gemini API service
- **Added local transformer model**: Using the TinyLlama-1.1B model for generating explanations
- **Implemented fallback mechanisms**: Added template-based responses when the model isn't available
- **Performance optimizations**: Cached explanations to improve response time

## How It Works

1. When first run, the system will download a small but effective language model (~1.1GB)
   - The model is downloaded only once and saved in the `models/tinyllama_cache` folder
   - On subsequent runs, it will use the locally saved model without downloading again
2. This model runs entirely on your local machine with no API costs
3. The model generates personalized explanations for video recommendations based on user emotions
4. If the model can't be loaded, we fall back to well-crafted template responses

## Setup Instructions

1. Run `Install_LLM_Dependencies.bat` to install the required packages
2. Start the application normally with `Start_Application.bat`
3. The first run might take a minute as the model is downloaded

## System Requirements

- **RAM**: At least 4GB of free RAM for optimal performance
- **Disk Space**: ~1.2GB for the model files
- **GPU**: Optional, but will significantly improve performance if available

## Benefits

- **Cost Savings**: Completely eliminates API costs
- **Privacy**: All processing happens locally, no data sent to external services
- **Offline Use**: Works without an internet connection (after initial model download)
- **Customizable**: Can be easily extended with different local models
