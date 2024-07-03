import os
import logging
import gzip
import shutil
import warnings
import traceback

import pandas as pd
import pyopenms
from tqdm import tqdm
import polars as pl

print("\nPreprocessing pipeline for ALL spectra initializing...")
print("OpenMS version: " + pyopenms.VersionInfo().getVersion())
print("Pandas version: " + pd.__version__)
print("Polars version: " + pl.__version__)

# Initialize variables
base_dir = r"XX"

convert_dir = os.path.join(base_dir, "convert")
mz_dir = os.path.join(convert_dir, "mz")

extract_dir = os.path.join(base_dir, "extract")
log_dir = os.path.join(extract_dir, "log_acfm")

data_dir = os.path.join(base_dir, "data")
out_dir = os.path.join(data_dir, "acfm")

print(f"\nPath of mzML files: {convert_dir}")
print(f"Path of output files: {out_dir}")
print(f"Path of log files: {log_dir}")

activation_method_mapping = {
    0: "CID", 
    1: "PSD",   
    2: "PD",  
    3: "SID",  
    4: "BIRD", 
    5: "ECD",  
    6: "IMD", 
    7: "SORI",
    8: "HCID", 
    9: "LCID", 
    10: "PHD", 
    11: "ETD",  
    12: "PQD",
    13: "TRAP", 
    14: "HCD", 
}

print("Paths created\n")

# Process folders
print("Processing folders:\n")
for folder in os.listdir(mz_dir):
    try:
        print(f"Starting processing of {folder}")

        # Create a new logger for each folder
        logger = logging.getLogger(folder)
        logger.setLevel(logging.DEBUG)

        # Create a file handler which logs messages to a file
        log_file = os.path.join(log_dir, f"{folder}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Create a console handler for debugging purposes
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        # Create out folder if not exists
        path_out_folder = os.path.join(out_dir, folder)
        os.makedirs(path_out_folder, exist_ok=True)

        for file in os.listdir(os.path.join(mz_dir, folder)):
            try:
                if file.endswith(".mzML.gz") and not os.path.exists(os.path.join(path_out_folder, f"{file[:-8]}.ipc")):
                    print(f"Processing file {file}")
                    logger.info(f"Processing file {file}")

                    # Unzip gzipped mzML file
                    with gzip.open(os.path.join(mz_dir, folder, file), 'rb') as f_in:
                        with open(os.path.join(mz_dir, folder, file[:-8]), 'wb') as f_out:
                            f_out_path = f_out.name
                            shutil.copyfileobj(f_in, f_out)
                    file = file[:-8]
                    logger.info(f"Unzipped file {file}")

                    # Read mzML file
                    try:
                        print(f"Loading file {file}")
                        exp = pyopenms.MSExperiment()
                        pyopenms.MzMLFile().load(os.path.join(mz_dir, folder, file), exp)
                    except Exception as e:
                        print(f"Error loading file {file}: {str(e)}")
                        logger.error(f"Error loading file {file}: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue

                    print(f"File {file} loaded")
                    print(f"Number of spectra in file: {exp.size()}")
                    logger.info(f"Number of spectra in file: {exp.size()}")
                    
                    # clean up unzipped file we copied
                    os.remove(f_out_path)
                    logger.info(f"Removed unzipped file {file}")

                    # Create a list to store data before converting to dataframe
                    data = []

                    # Iterate over spectra
                    for i in tqdm(range(exp.size())):
                        spec = exp[i]

                        # Extract MS2 spectra
                        if spec.getMSLevel() == 2:
                            index = i
                            scan = spec.getNativeID()
                            header = spec.getMetaValue(b'filter string')
                            rt = spec.getRT()

                            precursor = spec.getPrecursors()[0]
                            precursor_mz = precursor.getMZ()
                            precursor_charge = precursor.getCharge()
                            precursor_intensity = precursor.getIntensity()

                            try:
                                activation_method = list(precursor.getActivationMethods())[0]
                                frag_type = activation_method_mapping[activation_method]
                                collision_energy = precursor.getMetaValue(b'collision energy')
                            except:
                                frag_type = "Unknown"
                                collision_energy = "Unknown"

                            try:
                                lower_offset = precursor.getIsolationWindowLowerOffset()
                                upper_offset = precursor.getIsolationWindowUpperOffset()
                                isolation_target = precursor.getMetaValue(b'isolation window target m/z')
                            except:
                                lower_offset = "Unknown"
                                upper_offset = "Unknown"
                                isolation_target = "Unknown"

                            mz, intensity = spec.get_peaks()

                            if len(intensity) > 800:
                                sorted_indices = sorted(range(len(intensity)), key=lambda k: intensity[k], reverse=True)[:800]
                                mz = [mz[i] for i in sorted_indices]
                                intensity = [intensity[i] for i in sorted_indices]

                            scale_factor = max(intensity)
                            intensity = [i/scale_factor for i in intensity]

                            data.append({
                                "index": index, "scan": scan, "header": header, "rt": rt, "frag_type": frag_type,
                                "collision_energy": collision_energy, "precursor_mz": precursor_mz, 
                                "precursor_charge": precursor_charge, "precursor_intensity": precursor_intensity, 
                                "lower_offset": lower_offset, "upper_offset": upper_offset, "isolation_target": isolation_target,
                                "mz": mz, "intensity": intensity, "scale_factor": scale_factor
                            })

                    print(f"File {file} processed, saving dataframe")
                    df = pd.DataFrame(data)
                    df = pl.DataFrame(df)
                    df.write_ipc(os.path.join(path_out_folder, f"{file}.ipc"))

                    print(f"File {file} completed and saved as ipc")
                    logger.info(f"File {file} complete, path of saved file: {os.path.join(path_out_folder, f'{file}.ipc')}")
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                logger.error(f"Error processing file {file}: {str(e)}")
                logger.error(traceback.format_exc())

        print(f"Processing of {folder} completed\n")
        logger.info(f"Processing of {folder} completed")
    except Exception as e:
        print(f"Error processing folder {folder}: {str(e)}")
        with open(os.path.join(log_dir, "general_errors.log"), "a") as error_log:
            error_log.write(f"Error in folder: {folder}\n")
            error_log.write(f"Error message: {str(e)}\n")
            error_log.write(traceback.format_exc())
            error_log.write("\n")

print("\nPreprocessing pipeline for ALL spectra completed. Exiting...\n")
