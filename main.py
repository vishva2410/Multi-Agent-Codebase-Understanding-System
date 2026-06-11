# main.py
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print("Codebase AI starting...")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: No API key found. Check your .env file.")
        return

    print(f"API Key loaded: {api_key[:8]}...")
    print("All good. Ready to build.")

if __name__ == "__main__":
    main()