import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script"""
    print("\n" + "="*60)
    print(f"Running: {script_name}")
    print("="*60)
    result = subprocess.run([sys.executable, script_name])
    return result.returncode == 0

def main():
    print("\n" + "🎯"*30)
    print("HARASSMENT DETECTOR - COMPLETE TEST SUITE")
    print("🎯"*30)
    
    # First train the model if needed
    if not os.path.exists('models/harassment_model.pkl'):
        print("\n⚠️ No model found. Training first...")
        if not run_script("train_model.py"):
            print("❌ Model training failed!")
            return
    
    # Run all tests
    tests = ["test_fix.py", "test_casual.py", "verify_detection.py"]
    
    for test in tests:
        if os.path.exists(test):
            run_script(test)
        else:
            print(f"⚠️ {test} not found")
    
    print("\n" + "🎉"*30)
    print("✅ All tests completed!")
    print("🎉"*30)
    print("\n📝 Next steps:")
    print("   1. Run 'python app.py' to start the web server")
    print("   2. Open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()