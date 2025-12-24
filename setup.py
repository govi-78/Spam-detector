"""
Setup script for Spam Detection System
Automates the installation and setup process
"""
import os
import sys
import subprocess

def print_header(message):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {message}")
    print("="*60 + "\n")

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"➜ {description}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}: {e}\n")
        return False

def main():
    """Main setup function"""
    print_header("SPAM DETECTION SYSTEM - SETUP")
    
    # Step 1: Create virtual environment
    print_header("Step 1: Creating Virtual Environment")
    if not os.path.exists('venv'):
        run_command(f'{sys.executable} -m venv venv', 'Creating virtual environment')
    else:
        print("Virtual environment already exists\n")
    
    # Step 2: Install dependencies
    print_header("Step 2: Installing Dependencies")
    
    # Determine pip command based on OS
    if sys.platform == 'win32':
        pip_cmd = 'venv\\Scripts\\pip'
        python_cmd = 'venv\\Scripts\\python'
    else:
        pip_cmd = 'venv/bin/pip'
        python_cmd = 'venv/bin/python'
    
    run_command(f'{pip_cmd} install --upgrade pip', 'Upgrading pip')
    run_command(f'{pip_cmd} install -r requirements.txt', 'Installing Python packages')
    
    # Step 3: Download NLTK data
    print_header("Step 3: Downloading NLTK Data")
    nltk_script = """
import nltk
nltk.download('stopwords')
nltk.download('punkt')
print('NLTK data downloaded successfully')
"""
    with open('temp_nltk_download.py', 'w') as f:
        f.write(nltk_script)
    
    run_command(f'{python_cmd} temp_nltk_download.py', 'Downloading NLTK data')
    
    # Clean up temporary file
    if os.path.exists('temp_nltk_download.py'):
        os.remove('temp_nltk_download.py')
    
    # Step 4: Create models directory
    print_header("Step 4: Creating Project Directories")
    os.makedirs('models', exist_ok=True)
    print("✓ Models directory created\n")
    
    # Step 5: Train the model
    print_header("Step 5: Training Spam Detection Model")
    train_model = input("Do you want to train the model now? (y/n): ").lower().strip()
    
    if train_model == 'y':
        run_command(f'{python_cmd} train_model.py', 'Training model')
    else:
        print("⚠ Skipping model training. You can train it later by running: python train_model.py\n")
    
    # Step 6: Database setup instructions
    print_header("Step 6: Database Setup")
    print("Please ensure MySQL is installed and running on your system.\n")
    print("Steps to configure the database:")
    print("1. Copy .env.example to .env")
    print("2. Update the .env file with your MySQL credentials")
    print("3. Run: python database.py (to create database and tables)\n")
    
    setup_db = input("Do you want to initialize the database now? (y/n): ").lower().strip()
    
    if setup_db == 'y':
        run_command(f'{python_cmd} database.py', 'Initializing database')
    else:
        print("⚠ Skipping database setup. Run 'python database.py' when ready.\n")
    
    # Final instructions
    print_header("✓ SETUP COMPLETE!")
    print("Next steps:")
    print("\n1. Activate the virtual environment:")
    if sys.platform == 'win32':
        print("   .\\venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Configure your database (if not done):")
    print("   - Copy .env.example to .env")
    print("   - Update MySQL credentials in .env")
    print("   - Run: python database.py")
    
    print("\n3. Start the Flask application:")
    print("   python app.py")
    
    print("\n4. Open your browser and navigate to:")
    print("   http://localhost:5000")
    
    print("\n5. AUTHENTICATION FLOW:")
    print("   - First, register a new account at /register")
    print("   - Then login with your credentials at /login")
    print("   - After login, you'll be redirected to the dashboard")
    print("   - All spam detection features require authentication")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
