#!/usr/bin/env python3
"""
CPU-only dependency installer for TakuNet
This script installs CPU-only PyTorch and all other dependencies with specific options
"""
import subprocess
import sys
import os
import time

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(msg):
    print(f"{Colors.HEADER}{Colors.BOLD}[STEP]{Colors.ENDC} {msg}")

def print_error(msg):
    print(f"{Colors.FAIL}{Colors.BOLD}[ERROR]{Colors.ENDC} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}{Colors.BOLD}[SUCCESS]{Colors.ENDC} {msg}")

def print_warning(msg):
    print(f"{Colors.WARNING}{Colors.BOLD}[WARNING]{Colors.ENDC} {msg}")

def run_command(cmd, description=None, check=False, verbose=False):
    """Run a command and return True if successful, False otherwise"""
    if description:
        print_step(description)
    
    print(f"{Colors.BLUE}Running:{Colors.ENDC} {' '.join(cmd)}")
    
    try:
        if verbose:
            # Run with live output for better user feedback
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
            
            process.stdout.close()
            process.wait()
            
            if process.returncode != 0 and check:
                print_error(f"Command failed with error code {process.returncode}")
                return False, None
            
            return process.returncode == 0, None
        else:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with error code {e.returncode}")
        print(f"{Colors.FAIL}Error output:{Colors.ENDC}\n{e.stderr}")
        return False, e.stderr
    except KeyboardInterrupt:
        print_warning("Installation interrupted by user.")
        return False, None

def main():
    # Requirements file path
    req_file = "src/requirements/requirements_cpu.txt"
    
    # Check if requirements file exists
    if not os.path.exists(req_file):
        print_error(f"Could not find requirements file: {req_file}")
        sys.exit(1)
    
    # Step 1: Install PyTorch packages separately with a longer timeout
    print_step("Installing CPU-only PyTorch (this might take a while)")
    success, _ = run_command(
        ["pip", "install", "--no-cache-dir", "--timeout", "180", 
         "torch==2.2.2", "torchvision==0.17.2", "torchaudio==2.2.2", 
         "--index-url", "https://download.pytorch.org/whl/cpu"],
        description="Installing PyTorch CPU packages",
        verbose=True
    )
    
    if not success:
        print_warning("PyTorch installation failed. Trying alternative approach...")
        # Try with a different mirror
        success, _ = run_command(
            ["pip", "install", "--no-cache-dir", "--timeout", "180", 
             "torch==2.2.2", "torchvision==0.17.2", "torchaudio==2.2.2", 
             "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
            description="Trying alternative PyTorch installation",
            verbose=True
        )
        
        if not success:
            print_error("Could not install PyTorch. Please install manually and try again.")
            print_error("Try: pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu")
            sys.exit(1)
    
    print_success("PyTorch installation completed!")
    
    # Step 2: Create a temporary requirements file without PyTorch packages
    temp_req_file = "temp_requirements.txt"
    try:
        print_step("Creating temporary requirements file without PyTorch packages")
        with open(req_file, 'r') as f, open(temp_req_file, 'w') as out:
            for line in f:
                if not line.strip().startswith(("torch==", "torchvision==", "torchaudio==", "#")):
                    out.write(line)
        
        # Step 3: Install remaining packages
        print_step("Installing remaining dependencies")
        success, _ = run_command(
            ["pip", "install", "--no-cache-dir", "--upgrade-strategy", "only-if-needed", 
             "-r", temp_req_file],
            description="Installing remaining dependencies",
            verbose=True
        )
        
        if not success:
            print_error("Failed to install some dependencies.")
            print_warning("You may need to install them manually.")
            sys.exit(1)
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_req_file):
            os.remove(temp_req_file)
    
    print_success("CPU-only installation completed successfully!")
    print(f"{Colors.GREEN}You can now use TakuNet with CPU-only dependencies.{Colors.ENDC}")

if __name__ == "__main__":
    main() 