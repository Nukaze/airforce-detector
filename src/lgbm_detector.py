import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import os
from typing import Tuple, List

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

class AircraftLGBMDetector:
    def __init__(self, model_params: dict = None):
        """
        Initialize the LGBM-based aircraft detector
        
        Args:
            model_params: Dictionary of LightGBM parameters
        """
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = get_aircraft_class_names()
        self.model_params = model_params or {
            'objective': 'multiclass',
            'num_class': len(self.class_names),  # Number of aircraft classes
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

    def prepare_data(self, image_paths: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from image paths and labels
        
        Args:
            image_paths: List of paths to images
            labels: List of corresponding labels
            
        Returns:
            Tuple of (features, labels)
        """
        features = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                feature_vector = self.extract_features(img)
                features.append(feature_vector)
        
        return np.array(features), np.array(labels)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train the LGBM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create training dataset
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        
        # Create validation dataset if provided
        valid_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            valid_data = lgb.Dataset(X_val_scaled, label=y_val)
        
        # Train model
        self.model = lgb.train(
            self.model_params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data] + ([valid_data] if valid_data else []),
            early_stopping_rounds=10,
            verbose_eval=False
        )

    def predict(self, image: np.ndarray) -> Tuple[int, float, str]:
        """
        Predict aircraft type in an image
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (predicted class index, confidence score, class name)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Extract features
        features = self.extract_features(image)
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(features_scaled)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        class_name = self.class_names[predicted_class]
        
        return predicted_class, confidence, class_name

    def save_model(self, model_path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save_model(model_path)

    def load_model(self, model_path: str):
        """Load a trained model"""
        self.model = lgb.Booster(model_file=model_path)

def main():
    # Example usage
    detector = AircraftLGBMDetector()
    
    # Example data preparation (you'll need to implement this based on your data)
    # image_paths = [...]  # List of paths to your images
    # labels = [...]       # List of corresponding labels (0 for non-aircraft, 1 for aircraft)
    
    # X_train, X_val, y_train, y_val = train_test_split(
    #     image_paths, labels, test_size=0.2, random_state=42
    # )
    
    # # Prepare data
    # X_train_features, y_train = detector.prepare_data(X_train, y_train)
    # X_val_features, y_val = detector.prepare_data(X_val, y_val)
    
    # # Train model
    # detector.train(X_train_features, y_train, X_val_features, y_val)
    
    # # Save model
    # detector.save_model('aircraft_detector_lgbm.txt')

if __name__ == "__main__":
    main() 