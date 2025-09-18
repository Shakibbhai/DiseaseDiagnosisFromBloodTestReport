import google.generativeai as genai
import sys

# Configure API
GEMINI_API_KEY = "AIzaSyAnJqwNBngAizC5ibZQVE2aMoWq3TS3bek"
genai.configure(api_key=GEMINI_API_KEY)

def test_api():
    try:
        # List available models
        print("Available models:")
        for m in genai.list_models():
            print(f"- {m.name}")
        
        # Try to initialize the flash model
        model = genai.GenerativeModel("gemini-1.5-flash")
        print("\nModel initialized successfully!")
        
        return True
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if test_api():
        print("\nAPI test passed successfully!")
    else:
        print("\nAPI test failed. Please check your API key and quota.", file=sys.stderr)
