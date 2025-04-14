import os
import numpy as np
import cv2
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lgbm_utils import BaseAircraftDetector, get_aircraft_class_names

# Check GPU availability
print("\nChecking GPU availability for LightGBM...")
try:
    # Try to create a simple model with GPU
    params = {'device': 'gpu', 'objective': 'binary'}
    lgb.Dataset(np.random.rand(10, 10), label=np.random.randint(2, size=10))
    lgb.train(params, lgb.Dataset(np.random.rand(10, 10), label=np.random.randint(2, size=10)), num_boost_round=1)
    gpu_available = True
    print("GPU available: Yes")
    print("GPU acceleration will be used for training.")
except Exception as e:
    gpu_available = False
    print("GPU available: No")
    print("Warning: GPU acceleration is not available. Training will use CPU.")
    print("To enable GPU support, make sure you have:")
    print("1. NVIDIA GPU with CUDA support")
    print("2. CUDA toolkit installed")
    print("3. LightGBM installed with GPU support: pip install lightgbm --install-option=--gpu")
    print(f"Error details: {str(e)}")

class AircraftLGBMTrainer(BaseAircraftDetector):
    def __init__(self):
        super().__init__()
        # Enable GPU acceleration if available
        if gpu_available:
            self.model_params.update({
                'device': 'gpu',  # Use GPU
                'gpu_platform_id': 0,  # Use first GPU platform
                'gpu_device_id': 0,    # Use first GPU device
                'gpu_use_dp': True,    # Use double precision
                'max_bin': 63,         # GPU-specific parameter
                'gpu_predictor': True, # Use GPU for prediction
            })
            print("\nGPU parameters configured:")
            for param, value in self.model_params.items():
                if param.startswith('gpu_'):
                    print(f"  {param}: {value}")
        else:
            print("\nUsing CPU parameters for training")

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
        valid_labels = []
        corrupted_count = 0
        
        print(f"Processing {len(image_paths)} images...")
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            try:
                img = cv2.imread(img_path)
                if img is not None and img.size > 0:
                    # Resize image to standard size
                    img = cv2.resize(img, (32, 32))
                    feature_vector = self.extract_features(img)
                    features.append(feature_vector)
                    valid_labels.append(label)
                else:
                    corrupted_count += 1
                    print(f"\nWarning: Could not read image {img_path}")
            except Exception as e:
                corrupted_count += 1
                print(f"\nError processing {img_path}: {str(e)}")
            
            print(f"Processed {i + 1}/{len(image_paths)} images (skipped {corrupted_count} corrupted)", end="\r")
        
        print(f"\nSuccessfully processed {len(features)} images, skipped {corrupted_count} corrupted images")
        
        if len(features) == 0:
            raise ValueError("No valid images could be processed")
            
        return np.array(features), np.array(valid_labels)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              num_boost_round: int = 5, verbose: int = -1):
        """
        Train the LGBM model with GPU acceleration if available
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_boost_round: Number of boosting rounds
            verbose: Verbosity level
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
        
        # Train model with GPU if available
        print(f"\nTraining model with {'GPU' if gpu_available else 'CPU'} acceleration...")
        self.model = lgb.train(
            self.model_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data] + ([valid_data] if valid_data else []),
            # early_stopping_rounds=10,
            # verbose_eval=verbose
        )

    def save_model(self, model_path: str):
        """
        Save the trained model to a file
        
        Args:
            model_path: Path to save the model
        """
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model in binary format
        model_path = os.path.join('models', model_path)
        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")

def load_individual_csv_data(dataset_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load data where each image has its own CSV file with the same base name
    
    Args:
        dataset_dir: Directory containing images and CSV files
        
    Returns:
        Tuple of (image_paths, labels)
    """
    # Get class names and create mapping
    class_names = get_aircraft_class_names()
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    count = 0
    total_files = len(os.listdir(dataset_dir))
    
    print(f"Scanning {total_files} files in dataset directory...")
    # Get all image files
    for img_name in os.listdir(dataset_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_name)[0]
            csv_name = f"{base_name}.csv"
            csv_path = os.path.join(dataset_dir, csv_name)
            
            print(f"Processing {count + 1} of {total_files} ({count / total_files * 100:.2f}%)", end="\r")
            count += 1
            
            if os.path.exists(csv_path):
                try:
                    # Read the CSV file
                    df = pd.read_csv(csv_path)
                    if 'class' in df.columns:
                        class_name = df['class'].iloc[0]  # Get the class from the first row
                        if class_name in class_to_idx:
                            img_path = os.path.join(dataset_dir, img_name)
                            image_paths.append(img_path)
                            labels.append(class_to_idx[class_name])
                        else:
                            print(f"\nWarning: Unknown class '{class_name}' in {csv_name}")
                    else:
                        print(f"\nWarning: No 'class' column in {csv_name}")
                except Exception as e:
                    print(f"\nError reading {csv_name}: {str(e)}")
            else:
                print(f"\nWarning: No CSV file found for {img_name}")
    
    return image_paths, labels

def main():
    root = os.getcwd()
    # Path to your dataset
    dataset_dir = os.path.join(root, "data", "aircraft_dataset", "dataset")
    
    # Get image paths and labels
    print("Loading dataset...")
    print(f"Dataset directory: {dataset_dir}")
    
    image_paths, labels = load_individual_csv_data(dataset_dir)
    print(f"\nFound {len(image_paths)} images with valid CSV annotations")
    
    if len(image_paths) == 0:
        print("Error: No valid images found in the dataset directory!")
        return
    
    # Split into train and validation sets
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize trainer
    trainer = AircraftLGBMTrainer()
    
    try:
        # Prepare training data
        print("\nPreparing training data...")
        X_train_features, y_train = trainer.prepare_data(X_train_paths, y_train)
        X_val_features, y_val = trainer.prepare_data(X_val_paths, y_val)
        
        # Train model
        trainer.train(
            X_train_features, 
            y_train, 
            X_val_features, 
            y_val,
            num_boost_round=100,
            verbose=0,
        )
        
        # Save model
        print("\nSaving model...")
        trainer.save_model('aircraft_detector.lgb')
        print("Training complete!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        return

if __name__ == "__main__":
    main() 