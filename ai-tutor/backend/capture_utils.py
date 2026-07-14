import cv2
import os

def get_haarcascade_path():
    """Get the path to Haar cascade files"""
    possible_paths = [
        cv2.data.haarcascades,
        'haarcascade_frontalface_default.xml',
        os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
        elif os.path.exists(path) and path.endswith('.xml'):
            return os.path.dirname(path)
    
    # Fallback to current directory
    return ""

# Note: capture_multiple_frames_for_registration and capture_face_for_registration
# have been removed — they relied on cv2.VideoCapture(0) which is not available
# on headless cloud servers. Webcam input is now handled in the browser via
# getUserMedia() and frames are sent to the /recognize API endpoint.