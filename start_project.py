import os
import sys
import subprocess
import time

def main():
    print("=" * 60)
    print(" SmartKitchen AI X - Master Setup & Run Script")
    print("=" * 60)
    
    # 1. Get the current directory and the backend directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    
    if not os.path.exists(backend_dir):
        print(f"❌ Error: Could not find the backend folder at {backend_dir}")
        return

    # 2. Change the working directory to the backend folder
    os.chdir(backend_dir)
    sys.path.insert(0, backend_dir)
    print(f"✅ Changed directory to: {backend_dir}")
    print()

    # 3. Install requirements automatically
    print("📦 Step 1: Installing required packages (this might take a minute)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return
    print()

    # 4. Generate data and train ML models
    print("⏳ Step 2: Generating data and training ML models...")
    try:
        # Run setup_ml.py programmatically
        subprocess.run([sys.executable, "setup_ml.py"], check=True)
        print("✅ ML Setup Complete!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during ML setup: {e}")
        return
    except FileNotFoundError:
        print("❌ Error: python command not found.")
        return
        
    print()

    # 4. Start the server
    print("⏳ Step 2: Starting the backend server...")
    print("🌐 The API will be available at: http://localhost:8000")
    print("🚀 You can now double-click your index.html file to open the website!")
    print("-" * 60)
    
    try:
        # Run Uvicorn
        subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()
