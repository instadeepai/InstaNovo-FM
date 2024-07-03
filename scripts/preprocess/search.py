import os
import subprocess
import pandas as pd

# Define the paths
base_dir = r"XX"
mz_dir = os.path.join(base_dir, "convert", "mz")

# Load the project data from the provided table (assuming it's a CSV for this example)
project_data_path = os.path.join(base_dir, "code", "search_data.tsv")
project_data = pd.read_csv(project_data_path, delimiter='\t')

search_dir = r"XX"
result_dir = os.path.join(search_dir, "result")
manifest_dir = os.path.join(search_dir, "manifest")
log_dir = os.path.join(search_dir, "log")

fragpipe_path = r"XX\FragPipe-jre-22.0\fragpipe\bin\fragpipe.bat"
workflow_template_path = os.path.join(search_dir, "workflow", "template.workflow")

print("Starting the FragPipe search process...")

# Function to create a manifest file
def create_manifest(mzml_file, acquisition_type, manifest_path):
    with open(manifest_path, 'w') as manifest_file:
        manifest_file.write(f"{mzml_file}\t\t\t{acquisition_type}\n")
    print(f"Created manifest file: {manifest_path}")

# Iterate through each project and its associated mzML files
for _, row in project_data.iterrows():
    project_folder = row['project']
    mzml_file = row['file path']
    workflow_file = row['workflow']
    acquisition_type = row['acquisition']
    
    project_path = os.path.join(mz_dir, project_folder)
    if not os.path.isdir(project_path):
        print(f"Project path does not exist: {project_path}")
        continue
    
    print(f"\nProcessing project: {project_folder}")
    
    # Create result, log, and manifest directories for the current project
    project_result_dir = os.path.join(result_dir, project_folder)
    project_log_dir = os.path.join(log_dir, project_folder)
    project_manifest_dir = os.path.join(manifest_dir, project_folder)
    os.makedirs(project_result_dir, exist_ok=True)
    os.makedirs(project_log_dir, exist_ok=True)
    os.makedirs(project_manifest_dir, exist_ok=True)
    
    # Create a subfolder in the result directory for the current mzML file
    mzml_file_name = os.path.splitext(os.path.basename(mzml_file))[0]
    mzml_result_dir = os.path.join(project_result_dir, mzml_file_name)
    
    # Check if the result directory exists and is not empty
    if os.path.exists(mzml_result_dir) and os.listdir(mzml_result_dir):
        print(f"Results already exist for {mzml_file_name}, skipping search.")
        continue

    os.makedirs(mzml_result_dir, exist_ok=True)
    print(f"Created result directory: {mzml_result_dir}")
    
    # Create a manifest file for the current file search
    manifest_path = os.path.join(project_manifest_dir, f"{mzml_file_name}.manifest")
    create_manifest(mzml_file, acquisition_type, manifest_path)

    # Log file for the current mzML file
    log_file_path = os.path.join(project_log_dir, f"{mzml_file_name}_fragpipe_log.txt")

    # Call FragPipe for the current mzML file and log the output
    fragpipe_command = f'{fragpipe_path} --headless --workflow "{workflow_file}" --manifest "{manifest_path}" --workdir "{mzml_result_dir}" --ram 60 --threads 12'
    print(f"\nRunning FragPipe command for {mzml_file_name}...")
    
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(fragpipe_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for line in process.stdout:
            decoded_line = line.decode(errors='ignore')
            print(decoded_line, end='')
            log_file.write(decoded_line)
        for line in process.stderr:
            decoded_line = line.decode(errors='ignore')
            print(decoded_line, end='')
            log_file.write(decoded_line)
        process.wait()
    
    if process.returncode != 0:
        print(f"Error running FragPipe for {mzml_file_name}. Check log: {log_file_path}")
    else:
        print(f"Completed FragPipe for {mzml_file_name}. Check log: {log_file_path}\n")

print("FragPipe search completed.")
