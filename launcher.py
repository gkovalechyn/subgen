import os
import sys
import urllib.request
import subprocess
import argparse

def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')

def prompt_and_save_bazarr_env_variables():
    """
    Prompts the user for Bazarr related environment variables with descriptions and saves them to a file.
    If the user does not input anything, default values are used.
    """
    # Instructions for the user
    instructions = (
        "You will be prompted for several configuration values.\n"
        "If you wish to use the default value for any of them, simply press Enter without typing anything.\n"
        "The default values are shown in brackets [] next to the prompts.\n"
        "Items can be the value of true, on, 1, y, yes, false, off, 0, n, no, or an appropriate text response.\n"
    )
    print(instructions)
    env_vars = {
        'WHISPER_MODEL': ('Whisper Model', 'Enter the Whisper model you want to run: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large, distil-large-v2, distil-medium.en, distil-small.en', 'medium'),
        'WEBHOOKPORT': ('Webhook Port', 'Default listening port for subgen.py', '9000'),
        'TRANSCRIBE_DEVICE': ('Transcribe Device', 'Set as cpu or gpu', 'gpu'),
        'DEBUG': ('Debug', 'Enable debug logging', 'True'),
        'CLEAR_VRAM_ON_COMPLETE': ('Clear VRAM', 'Attempt to clear VRAM when complete (Windows users may need to set this to False)', 'False'),
        'APPEND': ('Append', 'Append \'Transcribed by whisper\' to generated subtitle', 'False'),
    }

    # Dictionary to hold the user's input
    user_input = {}

    # Prompt the user for each environment variable and write to .env file
    with open('subgen.env', 'w') as file:
        for var, (description, prompt, default) in env_vars.items():
            value = input(f"{prompt} [{default}]: ") or default
            file.write(f"{var}={value}\n")

    print("Environment variables have been saved to subgen.env")

def load_env_variables(env_filename='subgen.env'):
    """
    Loads environment variables from a specified .env file and sets them.
    """
    try:
        with open(env_filename, 'r') as file:
            for line in file:
                var, value = line.strip().split('=', 1)
                os.environ[var] = value

        print(f"Environment variables have been loaded from {env_filename}")

    except FileNotFoundError:
        print(f"{env_filename} file not found. Please run prompt_and_save_env_variables() first.")

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
    
    # Construct the argument parser
    parser = argparse.ArgumentParser(prog="python launcher.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--debug', default=False, action='store_true', help="Enable console debugging")
    parser.add_argument('-i', '--install', default=False, action='store_true', help="Install/update all necessary packages")
    parser.add_argument('-a', '--append', default=False, action='store_true', help="Append 'Transcribed by whisper' to generated subtitle")
    parser.add_argument('-u', '--update', default=False, action='store_true', help="Update Subgen")
    parser.add_argument('-x', '--exit-early', default=False, action='store_true', help="Exit without running subgen.py")
    parser.add_argument('-s', '--setup-bazarr', default=False, action='store_true', help="Prompt for common Bazarr setup parameters and save them for future runs")
    parser.add_argument('-b', '--branch', type=str, default='main', help='Specify the branch to download from') 
    parser.add_argument('-l', '--launcher-update', default=False, action='store_true', help="Update launcher.py and re-launch")

    args = parser.parse_args()

    # Set environment variables based on the parsed arguments
    if not convert_to_bool(os.environ.get('DEBUG', '')):
        os.environ['DEBUG'] = str(args.debug)
    if not convert_to_bool(os.environ.get('APPEND', '')):
        os.environ['APPEND'] = str(args.append)

    if args.setup_bazarr: 
        prompt_and_save_bazarr_env_variables()

    load_env_variables()

    print(f'Launching subgen{script_name}')
    if branch_name != 'main':
        subprocess.run([f'{python_cmd}', '-u', f'subgen{script_name}'], check=True)
    else:
        subprocess.run([f'{python_cmd}', '-u', 'subgen.py'], check=True)

if __name__ == "__main__":
    main()
