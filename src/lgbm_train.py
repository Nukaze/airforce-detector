import numpy as np
import cv2
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from .lgbm_utils import BaseAircraftDetector

class AircraftLGBMTrainer(BaseAircraftDetector):
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

def main():
    # Example usage
    trainer = AircraftLGBMTrainer()
    
    # Example data preparation (you'll need to implement this based on your data)
    # image_paths = [...]  # List of paths to your images
    # labels = [...]       # List of corresponding labels
    
    # X_train, X_val, y_train, y_val = train_test_split(
    #     image_paths, labels, test_size=0.2, random_state=42
    # )
    
    # # Prepare data
    # X_train_features, y_train = trainer.prepare_data(X_train, y_train)
    # X_val_features, y_val = trainer.prepare_data(X_val, y_val)
    
    # # Train model
    # trainer.train(X_train_features, y_train, X_val_features, y_val)
    
    # # Save model
    # trainer.save_model('aircraft_detector_lgbm.txt')

if __name__ == "__main__":
    main() 