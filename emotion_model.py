import cv2
import numpy as np
try:
    from fer import FER
except ImportError:
    from fer.fer import FER

detector = FER(mtcnn=True)

def detect_emotion(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return "error", 0.0
        
    # cv2.imwrite("debug_frame.jpg", img) 
    # print("Debug: Saved incoming frame to debug_frame.jpg")
        
    result = detector.top_emotion(img)
    
    if result and result[0] is not None:
        emotion, confidence = result
        return emotion, confidence
        
    return "no face detected", 0.0
