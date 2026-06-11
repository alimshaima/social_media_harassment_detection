import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_thresholds():
    """Test different thresholds with the current model"""
    
    # Check if model exists
    model_path = 'models/harassment_model.pkl'
    vectorizer_path = 'models/vectorizer.pkl'
    
    if not os.path.exists(model_path):
        print("❌ No model found. Please run train_model.py first!")
        return
    
    # Load model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    test_texts = [
        ('hello how are you', False),
        ('hi', False),
        ('good job', False),
        ('you are stupid', True),
        ('go die loser', True),
        ('kill yourself', True),
    ]
    
    print("🔧 Testing Different Thresholds:")
    print("=" * 60)
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75]:
        print(f"\n📏 Threshold: {threshold}")
        print("-" * 40)
        
        correct = 0
        for text, should_be_harassment in test_texts:
            vec = vectorizer.transform([text])
            proba = model.predict_proba(vec)[0]
            harassment_prob = proba[1]
            
            is_harassment = harassment_prob > threshold
            is_correct = (is_harassment == should_be_harassment)
            
            if is_correct:
                correct += 1
            
            status = '✅' if is_correct else '❌'
            print(f"{status} '{text:20}' -> prob: {harassment_prob:.2f}")
        
        accuracy = correct / len(test_texts)
        print(f"   Accuracy: {accuracy:.1%}")

if __name__ == "__main__":
    test_thresholds()