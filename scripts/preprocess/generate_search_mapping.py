import os
import pandas as pd

# Define the root folder containing the project folders
root_folder = r'XX'

mz_folder = os.path.join(root_folder, 'convert', 'mz')

# List to store the data for each mzML file
data = []

# Walk through the directory tree
for project in os.listdir(mz_folder):
    project_path = os.path.join(mz_folder, project)
    if os.path.isdir(project_path):
        for file in os.listdir(project_path):
            if file.endswith('.mzML'):
                file_path = os.path.join(project_path, file)
                data.append({
                    'project': project,
                    'file path': file_path,
                    'workflow': '',
                    'acquisition': ''
                })

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Write the DataFrame to an Excel file
output_path = os.path.join(root_folder, 'code', 'search_data.tsv')
df.to_csv(output_path, index=False, sep='\t')

print(f'Tsv file has been created: {output_path}')
