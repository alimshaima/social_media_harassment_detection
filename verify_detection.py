import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.prediction import HarassmentPredictor

def main():
    predictor = HarassmentPredictor()
    
    tests = [
        ("bitch", True, "profanity"),
        ("fuck you", True, "directed profanity"),
        ("kill yourself", True, "self-harm"),
        ("you are stupid", True, "direct insult"),
        ("hi", False, "greeting"),
        ("hello", False, "greeting"),
        ("how are you", False, "question"),
        ("i feel stupid", False, "self-directed"),
    ]
    
    print("\n🔍 Testing Harassment Detection")
    print("=" * 50)
    
    correct_count = 0
    for text, expected, category in tests:
        result = predictor.predict(text)
        actual = result['is_harassment']
        correct = (actual == expected)
        
        if correct:
            correct_count += 1
            print(f"✅ PASS: '{text}' -> {'HARASS' if actual else 'SAFE'} ({category})")
        else:
            print(f"❌ FAIL: '{text}' -> {'HARASS' if actual else 'SAFE'} (should be {'HARASS' if expected else 'SAFE'})")
    
    print("\n" + "=" * 50)
    print(f"Results: {correct_count}/{len(tests)} correct")
    
    if correct_count == len(tests):
        print("🎉 SUCCESS: All detections are correct!")
    else:
        print("⚠️ Some detections are incorrect")

if __name__ == "__main__":
    main()