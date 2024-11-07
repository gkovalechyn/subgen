import os
import sys
import urllib.request
import subprocess
import argparse

def main():
    # Check if the script is run with 'python' or 'python3'
    if 'python3' in sys.executable:
        python_cmd = 'python3'
    elif 'python' in sys.executable:
        python_cmd = 'python'
    else:
        print("Script started with an unknown command")
        sys.exit(1)
        
    if sys.version_info[0] < 3:
        print(f"This script requires Python 3 or higher, you are running {sys.version}")
        sys.exit(1)  # Terminate the script
    
    #Make sure we're saving subgen.py and subgen.env in the right folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(f'Launching subgen{script_name}')
    if branch_name != 'main':
        subprocess.run([f'{python_cmd}', '-u', f'subgen{script_name}'], check=True)
    else:
        subprocess.run([f'{python_cmd}', '-u', 'subgen.py'], check=True)

if __name__ == "__main__":
    main()
