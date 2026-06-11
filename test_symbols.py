import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.prediction import HarassmentPredictor

def main():
    predictor = HarassmentPredictor()
    
    print("\n" + "="*70)
    print("🔍 TESTING SYMBOL AND LEETSPEAK DETECTION")
    print("="*70)
    
    test_cases = [
        # Should be detected as HARASSMENT
        ("f*ck", True, "fuck with asterisk"),
        ("f@ck", True, "fuck with @"),
        ("f u c k", True, "spaced out"),
        ("b!tch", True, "bitch with !"),
        ("wh0re", True, "whore with 0"),
        ("sl*t", True, "slut with *"),
        ("c*nt", True, "cunt with *"),
        ("k!ll yourself", True, "kill with !"),
        ("kys", True, "kill yourself abbreviation"),
        ("you are st*pid", True, "stupid with *"),
        ("you b1tch", True, "bitch with 1"),
        ("@sshole", True, "asshole with @"),
        ("d!ck", True, "dick with !"),
        
        # Should be SAFE
        ("hi", False, "normal greeting"),
        ("how are you", False, "normal question"),
        ("good morning", False, "greeting"),
        ("i feel stupid", False, "self-directed"),
        ("this game is killing me", False, "hyperbole"),
        ("you are crazy good", False, "compliment"),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected, description in test_cases:
        result = predictor.predict(text)
        actual = result['is_harassment']
        is_correct = (actual == expected)
        
        if is_correct:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
        
        expected_str = "HARASS" if expected else "SAFE"
        actual_str = "HARASS" if actual else "SAFE"
        
        print(f"{status} '{text:25}' -> {actual_str:6} (expected: {expected_str:6}) | {description}")
        if actual and 'profanity_words' in result:
            print(f"   └─ Profanity: {result['profanity_words']}")
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Symbol/leetspeak detection is working!")
    else:
        print("⚠️ Some tests failed - check patterns")
    print("="*70)

if __name__ == "__main__":
    main()
