import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.model_training import HarassmentDetector

def main():
    print("Social Media Harassment Detection Model Training")
    print("=" * 50)
    
    # Preprocess data
    print("\n1. Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_preprocess_data()
    
    print("\n2. Preparing features...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_features(df)
    
    print(f"   Training set size: {len(X_train)}")
    print(f"   Test set size: {len(X_test)}")
    
    # Train models
    print("\n3. Training models...")
    detector = HarassmentDetector()
    results, X_test_vec, y_test = detector.train_models(X_train, X_test, y_train, y_test)
    
    # Save the best model
    print("\n4. Saving the best model...")
    detector.save_model()
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()