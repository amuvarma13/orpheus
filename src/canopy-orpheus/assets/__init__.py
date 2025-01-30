import os.path

# Get the directory containing this __init__.py file
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to speech.wav
SPEECH_WAV_PATH = os.path.join(ASSETS_DIR, 'speech.wav')

# Make these available when importing from assets
__all__ = ['SPEECH_WAV_PATH', 'ASSETS_DIR']