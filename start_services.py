import subprocess
import sys
import time
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_SCRIPT = os.path.join(BASE_DIR, "app.py")
WORKER_SCRIPT = os.path.join(BASE_DIR, "automation_service.py")

def start_system():
    print("\n" + "="*50)
    print("🚀 STARTING AI MICROSERVICES ECOSYSTEM")
    print("="*50 + "\n")

    processes = []

    try:
        # 1. Start Flask API
        print(f"   [1/2] Launching Flask API (Port 5001)...")
        api_process = subprocess.Popen([sys.executable, API_SCRIPT])
        processes.append(api_process)
        
        time.sleep(2) 

        # 2. Start Automation Worker
        print(f"   [2/2] Launching Automation Worker...")
        worker_process = subprocess.Popen([sys.executable, WORKER_SCRIPT])
        processes.append(worker_process)

        print("\n✅ SYSTEM IS LIVE!")
        print("   - API: http://localhost:5001/get_all_trends")
        print("   - Worker: Running in background")
        print("\nPress Ctrl+C to stop all services.\n")

        while True:
            time.sleep(1)
            if api_process.poll() is not None:
                print("\n❌ Flask API stopped!")
                break
            if worker_process.poll() is not None:
                print("\n❌ Worker stopped!")
                break

    except KeyboardInterrupt:
        print("\n\n🛑 STOPPING SYSTEM...")
    
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                p.wait()
        print("✓ All services stopped.")

if __name__ == "__main__":
    start_system()