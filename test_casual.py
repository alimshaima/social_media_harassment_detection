import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.prediction import HarassmentPredictor

def main():
    predictor = HarassmentPredictor()
    
    tests = [
        ("bro", False, "casual greeting"),
        ("hi", False, "friendly hello"),
        ("hiiiiiiiii", False, "friendly hello with many i's"),
        ("hey there", False, "casual greeting"),
        ("hello friend", False, "friendly greeting"),
        ("hw r u", False, "how are you"),
        ("how are you", False, "how are you"),
        ("good", False, "positive response"),
        ("you bitch", True, "explicit harassment"),
        ("kill yourself", True, "self-harm encouragement"),
        ("you are stupid", True, "direct insult"),
        ("fuck you", True, "profanity"),
    ]
    
    print("\n🔍 Testing Casual Language Detection")
    print("=" * 50)
    
    all_correct = True
    for text, expected, reason in tests:
        result = predictor.predict(text)
        actual = result['is_harassment']
        correct = (actual == expected)
        
        if not correct:
            all_correct = False
        
        status = "✅ PASS" if correct else "❌ FAIL"
        expected_str = "HARASS" if expected else "SAFE"
        actual_str = "HARASS" if actual else "SAFE"
        
        print(f"{status} '{text:20}' -> {actual_str:6} (expected: {expected_str:6}) | {reason}")
    
    if all_correct:
        print("\n🎉 SUCCESS: All casual language correctly identified!")
    else:
        print("\n⚠️ Some casual language was misclassified")

if __name__ == "__main__":
    main()