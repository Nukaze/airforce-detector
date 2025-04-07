import numpy as np
import cv2
from typing import Tuple, List, Dict
from .lgbm_utils import BaseAircraftDetector

class AircraftLGBMInference(BaseAircraftDetector):
    def predict(self, image: np.ndarray, top_k: int = 5) -> Tuple[int, float, str, List[Tuple[int, float, str]]]:
        """
        Predict aircraft type in an image
        
        Args:
            image: Input image
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (predicted class index, confidence score, class name, top-k predictions)
        """
        if self.model is None:
            raise ValueError("Model not loaded! Please load a trained model first.")
        
        # Extract features
        features = self.extract_features(image)
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(features_scaled)
        
        # Get top-k predictions
        top_k_indices = np.argsort(prediction[0])[::-1][:top_k]
        top_k_predictions = []
        
        for idx in top_k_indices:
            confidence = prediction[0][idx]
            class_name = self.class_names[idx]
            top_k_predictions.append((idx, confidence, class_name))
        
        # Get the best prediction
        predicted_class, confidence, class_name = top_k_predictions[0]
        
        return predicted_class, confidence, class_name, top_k_predictions

    def predict_batch(self, images: List[np.ndarray], top_k: int = 5) -> List[Tuple[int, float, str, List[Tuple[int, float, str]]]]:
        """
        Predict aircraft types for a batch of images
        
        Args:
            images: List of input images
            top_k: Number of top predictions to return for each image
            
        Returns:
            List of predictions for each image
        """
        return [self.predict(img, top_k) for img in images]

def main():
    # Example usage
    detector = AircraftLGBMInference()
    
    # Load trained model
    # detector.load_model('aircraft_detector_lgbm.txt')
    
    # Example prediction
    # image = cv2.imread('path_to_image.jpg')
    # predicted_class, confidence, class_name, top_k_predictions = detector.predict(image)
    
    # Print results
    # print(f"Predicted aircraft: {class_name}")
    # print(f"Confidence: {confidence:.2%}")
    # print("\nTop-k predictions:")
    # for idx, conf, name in top_k_predictions:
    #     print(f"{name}: {conf:.2%}")

if __name__ == "__main__":
    main() 