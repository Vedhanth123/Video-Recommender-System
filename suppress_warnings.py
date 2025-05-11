"""
Suppress TensorFlow and other warnings during application startup.
Import this at the beginning of your main application file.
"""

import os
import warnings
import logging

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Suppress Google API discovery warnings
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
