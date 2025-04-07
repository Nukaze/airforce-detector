import numpy as np
import cv2
from typing import Tuple, List
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

def get_aircraft_class_names():
    """Return list of aircraft class names"""
    class_names = [
        "A10", "A400M", "AG600", "AH64", "An124", "An22", "An225", "An72", 
        "AV8B", "B1", "B2", "B21", "B52", "Be200", "C130", "C17", "C2", 
        "C390", "C5", "CH47", "CL415", "E2", "E7", "EF2000", "F117", 
        "F14", "F15", "F16", "F18", "F22", "F35", "F4", "H6", "J10", 
        "J20", "JAS39", "JF17", "JH7", "Ka27", "Ka52", "KC135", "KF21", 
        "KJ600", "Mi24", "Mi26", "Mi28", "Mig29", "Mig31", "Mirage2000", 
        "MQ9", "P3", "Rafale", "RQ4", "SR71", "Su24", "Su25", "Su34", 
        "Su57", "TB001", "TB2", "Tornado", "Tu160", "Tu22M", "Tu95", 
        "U2", "UH60", "US2", "V22", "Vulcan", "WZ7", "XB70", "Y20", 
        "YF23", "Z19"
    ]
    return class_names

class BaseAircraftDetector:
    def __init__(self, model_params: dict = None):
        """
        Initialize the base aircraft detector
        
        Args:
            model_params: Dictionary of LightGBM parameters
        """
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = get_aircraft_class_names()
        self.model_params = model_params or {
            'objective': 'multiclass',
            'num_class': len(self.class_names),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Feature vector
        """
        # Resize image to standard size
        image = cv2.resize(image, (64, 64))
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features
        hog = cv2.HOGDescriptor()
        hog_features = hog.compute(gray)
        
        # Extract color histogram features
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Combine features
        features = np.concatenate([hog_features.flatten(), hist])
        
        return features

    def save_model(self, model_path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save_model(model_path)

    def load_model(self, model_path: str):
        """Load a trained model"""
        self.model = lgb.Booster(model_file=model_path) 