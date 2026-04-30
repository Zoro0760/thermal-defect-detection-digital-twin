import subprocess

def run_inference_script():
    """
    Automatically runs the local_inference.py script in the Anaconda environment
    and captures the output.
    """
    # Define the exact path to your script
    script_path = r"C:\blade_defect_project\scripts\# local_inference.py"
    
    # Define the conda environment name we saw earlier
    conda_env = "blade_defect"
    
    # Construct the command. 'conda run -n' automatically handles activating 
    # the environment without needing to manually open the Anaconda Prompt app.
    command = f'conda run -n {conda_env} python "{script_path}"'
    
    print(f"Starting inference using environment '{conda_env}'...\n")
    print(f"Running command: {command}\n")
    print("Please wait, this might take a few seconds...\n")
    
    try:
        # Execute the command
        # stderr=subprocess.STDOUT merges the error output into standard output 
        # so we get everything in one place.
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        print("========== INFERENCE OUTPUT ==========\n")
        print(result.stdout.strip())
        print("\n======================================")
        
    except subprocess.CalledProcessError as e:
        print("========== COMMAND FAILED ==========\n")
        print(f"Exit code: {e.returncode}")
        print("Output/Errors generated:")
        print(e.stdout)
        print("======================================")

if __name__ == "__main__":
    run_inference_script()
