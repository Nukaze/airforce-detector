import os
import numpy as np
import cv2
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from lgbm_train import AircraftLGBMTrainer

def get_image_paths_and_labels(dataset_dir: str) -> Tuple[List[str], List[int]]:
    """
    Get image paths and corresponding labels from the dataset directory
    
    Args:
        dataset_dir: Path to the dataset directory
        
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image_paths.append(img_path)
            labels.append(class_idx)
            
    return image_paths, labels

def main():
    # Path to your dataset
    dataset_dir = "../data/aircraft_dataset/dataset"
    
    # Get image paths and labels
    print("Loading dataset...")
    image_paths, labels = get_image_paths_and_labels(dataset_dir)
    print(f"Found {len(image_paths)} images")
    
    # Split into train and validation sets
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize trainer
    trainer = AircraftLGBMTrainer()
    
    # Prepare training data
    print("Preparing training data...")
    X_train_features, y_train = trainer.prepare_data(X_train_paths, y_train)
    X_val_features, y_val = trainer.prepare_data(X_val_paths, y_val)
    
    # Train model
    print("Training model...")
    trainer.train(X_train_features, y_train, X_val_features, y_val)
    
    # Save model
    print("Saving model...")
    trainer.save_model('aircraft_detector_lgbm.txt')
    print("Training complete!")

if __name__ == "__main__":
    main() 