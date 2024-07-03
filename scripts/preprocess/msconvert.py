import os
import subprocess
import traceback

# Define the paths
base_dir = r"XX"

raw_dir = os.path.join(base_dir, "raw")
convert_dir = os.path.join(base_dir, "convert")

mz_dir = os.path.join(convert_dir, "mz")
log_dir = os.path.join(convert_dir, "log")
file_list_dir = os.path.join(convert_dir, "files")

msconvert_path = '"XX\ProteoWizard\ProteoWizard 3.0.22088.7a226f0/msconvert.exe"'
error_log_path = os.path.join(log_dir, "conversion_errors.txt")

print("Starting the conversion process...")

# Iterate through each project folder in the raw directory
for project_folder in os.listdir(raw_dir):
    project_path = os.path.join(raw_dir, project_folder)
    
    if os.path.isdir(project_path):
        try:
            print(f"Converting files for project: {project_folder}")

            # Create a directory for the converted mzML files for the current project
            project_mz_dir = os.path.join(mz_dir, project_folder)
            os.makedirs(project_mz_dir, exist_ok=True)

            # Create a file list for the current project
            file_list_path = os.path.join(file_list_dir, f"{project_folder}_to_convert.txt")
            
            with open(file_list_path, "w") as file_list:
                for raw_file in os.listdir(project_path):
                    if raw_file.endswith(".raw") or raw_file.endswith(".d"):
                        ending = raw_file.split(".")[-1]
                        # Add only files that are not in the mzML directory
                        if not os.path.exists(os.path.join(project_mz_dir, raw_file.replace(ending, "mzML.gz"))):
                            file_list.write(os.path.join(project_path, raw_file) + "\n")

            print(f"File list created for project: {project_folder}")

            # Log file for the current project
            log_file_path = os.path.join(log_dir, f"{project_folder}_log.txt")

            print(f"Logging the conversion process for project: {project_folder}")

            # Call MSconvert for the current project and log the output
            msconvert_command = f'{msconvert_path} --mzML --zlib --gzip -f "{file_list_path}" -o "{project_mz_dir}" --filter "peakPicking vendor msLevel=1-" --filter "threshold count 800 most-intense" --filter "demultiplex  optimization=overlap_only massError=10.0ppm" --filter "titleMaker <RunId>.<ScanNumber>.<ScanNumber>.<ChargeState> File:"""^<SourcePath^>""", NativeID:"""^<Id^>""""'

            with open(log_file_path, "w") as log_file:
                process = subprocess.Popen(msconvert_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Stream the output and error to both the log file and the command line
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.strip())
                        log_file.write(output)
                    error = process.stderr.readline()
                    if error:
                        print(error.strip())
                        log_file.write(error)

                process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, msconvert_command)

            print(f"Conversion completed for project: {project_folder}")
        
        except Exception as e:
            print(f"Error during conversion for project: {project_folder}")
            with open(error_log_path, "a") as error_log:
                error_log.write(f"Error in project: {project_folder}\n")
                error_log.write(f"Command: {msconvert_command}\n")
                error_log.write(f"Error message: {str(e)}\n")
                error_log.write(traceback.format_exc())
                error_log.write("\n")

print("Conversion process completed.")
