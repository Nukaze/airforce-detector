import cv2
import numpy as np
from lgbm_inference import AircraftLGBMInference
import os

def main():
    # Initialize detector
    detector = AircraftLGBMInference()
    
    # Load trained model
    model_path = 'aircraft_detector_lgbm.txt'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please train the model first using train_lgbm.py")
        return
    
    detector.load_model(model_path)
    
    # Test image path
    test_image_path = "test_image.jpg"  # Replace with your test image path
    
    if not os.path.exists(test_image_path):
        print(f"Error: Test image {test_image_path} not found!")
        return
    
    # Read and predict
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Error: Could not read image {test_image_path}")
        return
    
    # Get prediction
    predicted_class, confidence, class_name, top_k_predictions = detector.predict(image)
    
    # Print results
    print(f"\nPredicted aircraft: {class_name}")
    print(f"Confidence: {confidence:.2%}")
    print("\nTop-5 predictions:")
    for idx, conf, name in top_k_predictions:
        print(f"{name}: {conf:.2%}")

if __name__ == "__main__":
    main() 