import cv2
import numpy as np
from fer import FER

# Initialize detector. Using default Haar Cascade for speed.
detector = FER()

def detect_emotion(image_bytes: bytes):
    # Convert incoming bytes to OpenCV image format
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return "error", 0.0
        
    # top_emotion returns a tuple: ('happy', 0.99)
    result = detector.top_emotion(img)
    
    if result and result[0] is not None:
        emotion, confidence = result
        return emotion, confidence
        
    return "no face detected", 0.0
