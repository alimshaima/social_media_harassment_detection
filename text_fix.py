import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.prediction import HarassmentPredictor

def main():
    predictor = HarassmentPredictor()
    
    test_cases = [
        ("hi", False),
        ("hiiiiiiiii", False),
        ("hw r u", False),
        ("how are you", False),
        ("good morning", False),
        ("kill yourself", True),
        ("you are stupid", True),
        ("fuck you", True),
        ("you bitch", True),
        ("bitch", True),
        ("nobody likes you", True),
        ("you should die", True),
    ]
    
    print("\n" + "="*60)
    print("TESTING HARASSMENT DETECTOR")
    print("="*60)
    
    all_passed = True
    for text, expected in test_cases:
        result = predictor.predict(text)
        actual = result['is_harassment']
        passed = (actual == expected)
        
        if not passed:
            all_passed = False
        
        status = "✅" if passed else "❌"
        expected_str = "HARASS" if expected else "SAFE"
        actual_str = "HARASS" if actual else "SAFE"
        
        print(f"{status} '{text:20}' -> {actual_str:6} (expected: {expected_str:6}) | {result['detection_method']}")
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️ Some tests failed - check patterns")
    print("="*60)

if __name__ == "__main__":
    main()