"""
Launcher script for the Emotion-Based Video Recommendation System.
This script provides an elegant way to start the application with appropriate settings.
"""

import os
import sys
import subprocess
import time

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    """Print a stylish banner for the application."""
    banner = f"""
{Colors.BLUE}{Colors.BOLD}╔════════════════════════════════════════════════════════════╗
║                                                            ║
║  {Colors.GREEN}Emotion-Based Video Recommendation System{Colors.BLUE}                 ║
║                                                            ║
║  {Colors.YELLOW}© 2025 - Academic Project{Colors.BLUE}                                ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)

def check_environment():
    """Check if the environment is properly set up."""
    print(f"{Colors.BLUE}[*]{Colors.ENDC} Checking environment...")
    
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print(f"{Colors.YELLOW}[!]{Colors.ENDC} Not running in a virtual environment.")
        print(f"{Colors.YELLOW}[!]{Colors.ENDC} Attempting to activate the virtual environment...")
        
        # Try to activate the environment
        venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MP")
        if os.path.exists(venv_path):
            activate_script = os.path.join(venv_path, "Scripts", "Activate.ps1")
            if os.path.exists(activate_script):
                print(f"{Colors.GREEN}[+]{Colors.ENDC} Found virtual environment at: {venv_path}")
                
                # We can't directly activate in this process, so we'll just inform the user
                print(f"\n{Colors.YELLOW}Please activate the virtual environment before running this script:{Colors.ENDC}")
                print(f"    cd {os.path.dirname(os.path.abspath(__file__))}")
                print(f"    .\\MP\\Scripts\\Activate.ps1")
                print(f"    python launcher.py")
                return False
            else:
                print(f"{Colors.RED}[-]{Colors.ENDC} Cannot find activation script: {activate_script}")
                return False
        else:
            print(f"{Colors.RED}[-]{Colors.ENDC} Cannot find virtual environment at: {venv_path}")
            return False
    else:
        print(f"{Colors.GREEN}[+]{Colors.ENDC} Running in a virtual environment: {sys.prefix}")
    
    # Check for required packages
    try:
        import streamlit
        import cv2
        import face_recognition
        import numpy
        import tensorflow
        print(f"{Colors.GREEN}[+]{Colors.ENDC} All required packages are available.")
        return True
    except ImportError as e:
        print(f"{Colors.RED}[-]{Colors.ENDC} Missing package: {e}")
        print(f"{Colors.YELLOW}[!]{Colors.ENDC} Please install the required packages:")
        print("    pip install -r requirements.txt")
        return False

def launch_application():
    """Launch the Streamlit application with optimized settings."""
    print(f"\n{Colors.BLUE}[*]{Colors.ENDC} Setting up application environment...")
    
    # Set environment variables to suppress warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations
    
    # Set Streamlit configuration for a cleaner appearance
    print(f"{Colors.BLUE}[*]{Colors.ENDC} Configuring Streamlit settings...")
    
    print(f"{Colors.GREEN}[+]{Colors.ENDC} Starting Emotion-Based Video Recommendation System...")
    print(f"{Colors.YELLOW}[!]{Colors.ENDC} Press Ctrl+C in this terminal to stop the application")
    print(f"\n{Colors.BLUE}{Colors.BOLD}Application is launching...{Colors.ENDC}")
    
    # Launch Streamlit with optimized settings
    cmd = [
        "streamlit", "run", "app.py",
        "--server.headless", "true",  # Headless mode for cleaner startup
        "--browser.serverAddress", "localhost",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light"
    ]
    
    # Wait a moment for the user to read the message
    time.sleep(1.5)
    
    # Start Streamlit
    return subprocess.call(cmd)

def main():
    """Main entry point for the launcher."""
    print_banner()
    
    if check_environment():
        exit_code = launch_application()
        
        if exit_code == 0:
            print(f"\n{Colors.GREEN}[+]{Colors.ENDC} Application closed successfully.")
        else:
            print(f"\n{Colors.RED}[-]{Colors.ENDC} Application exited with code: {exit_code}")
            
    print(f"\n{Colors.BLUE}Thank you for using Emotion-Based Video Recommendation System!{Colors.ENDC}")

if __name__ == "__main__":
    main()
