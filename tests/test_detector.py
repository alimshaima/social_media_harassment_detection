import sys
import os

# Add parent directory and src folder to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from src.prediction import HarassmentPredictor

def test_predictor():
    """Test the harassment predictor with sample texts"""
    print("Loading predictor...")
    predictor = HarassmentPredictor()
    
    test_texts = [
        "You're so stupid and worthless",
        "I love this beautiful day",
        "Go die you pathetic loser",
        "Great job on the project!",
        "hi",
        "hiiiiiiiii",
        "hw r u",
        "kill yourself"
    ]
    
    print("\nTesting Harassment Detector")
    print("=" * 50)
    
    for text in test_texts:
        result = predictor.predict(text)
        status = "🚨 HARASSMENT" if result['is_harassment'] else "✅ SAFE"
        print(f"{status}: '{text}'")
        print(f"   Method: {result['detection_method']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print()

if __name__ == "__main__":
    test_predictor()
