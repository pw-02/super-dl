import glob
from collections import defaultdict

def create_job_report(job_id, log_out_folder):
    import pandas as pd
    import os
    import yaml
    from collections import defaultdict
    import glob

    # Check if a file called 'summary_report.csv' exists in log_out_folder
    output_file_path = os.path.join(log_out_folder, 'summary_report.xlsx')
    variable_names = ['train/batch', 'train/epoch', 'train/job', 'val/batch', 'val/epoch', 'val/job']

    # Get a list of all files in log_out_folder ending with 'metrics.csv'
    metrics_files = glob.glob(os.path.join(log_out_folder, '*metrics.csv'))

    # Create a dictionary to store merged information with default value as a list
    merged_dict = defaultdict(list)

    # Iterate through each metrics file
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        for metrics_file in metrics_files:
            # Load the csv file into a pandas data frame
            df = pd.read_csv(metrics_file, sep=',')
            # Sort the columns alphabetically
            df = df.reindex(sorted(df.columns), axis=1)

            # Define variable names
            for var_name in variable_names:
                var_columns = [col for col in df.columns if col.startswith(var_name + '-')]
                var_dict = df[var_columns].to_dict(orient='list')
                # Remove NaN values from the dictionary
                var_dict_no_nan = {key: [value for value in values if pd.notna(value)] for key, values in var_dict.items()}
                if len(var_dict_no_nan.values()) > 0:
                    # Append dictionaries to the list for each variable name
                    merged_dict[var_name].append(var_dict_no_nan)

        # Merge the data within each variable name
        for var_name, var_data in merged_dict.items():
            merged_data = {}
            for data_dict in var_data:
                for key, value in data_dict.items():
                    if key in merged_data:
                        merged_data[key].extend(value)
                    else:
                        merged_data[key] = value
            # Replace invalid characters in sheet name
            clean_var_name = var_name.replace('/', '_').replace('\\', '_').replace('*', '').replace('?', '').replace(':', '').replace('[', '').replace(']', '')
            # Write each merged dictionary to a separate sheet
            pd.DataFrame(merged_data).to_excel(writer, sheet_name=clean_var_name, index=False)
    pass
           
'''

            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(metrics_file_path, sep=',')
            # Sort the columns alphabetically
            df = df.reindex(sorted(df.columns), axis=1)
            # Define variable names
            variable_names = ['train-batch', 'train-epoch', 'train-job', 'val-batch', 'val-epoch', 'val-job']

            #Create dictionaries with filtered columns
            dicts = {}
            for var_name in variable_names:
                var_columns = [col for col in df.columns if col.startswith(var_name + '/')]
                var_dict = df[var_columns].to_dict(orient='list')
                # Remove NaN values from the dictionary
                var_dict_no_nan = {key: [value for value in values if pd.notna(value)] for key, values in var_dict.items()}
                if len(var_dict_no_nan.values())>0:
                    dicts[var_name] = var_dict_no_nan
        
            # Read 'hparams.yaml' file
            hparams_file_path = os.path.join(log_out_folder, 'hparams.yaml')
            hparams_data = {}
            if os.path.isfile(hparams_file_path):
                with open(hparams_file_path, 'r') as hparams_file:
                    hparams_data = yaml.safe_load(hparams_file)

            # Save dictionaries to Excel file with each dictionary as a sheet
            excel_file_path =  os.path.join(log_out_folder, f"job_{job_id}_report.xlsx") 

            with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
                # Save hparams to a separate sheet
                df_hparams = pd.DataFrame.from_dict(hparams_data, orient='index')
                df_hparams.to_excel(writer, sheet_name='hparams', header=False)
                
                for sheet_name, data_dict in dicts.items():
                    df_sheet = pd.DataFrame(data_dict)
                    df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

        # Delete the 'metrics.csv' file
            #os.remove(metrics_file_path)
            print(f"Report tidied up and saved to: {excel_file_path}. 'metrics.csv' file deleted.")
        else:
            print("Error: 'metrics.csv' file not found in the specified folder.")
'''

if __name__ == "__main__":
    create_job_report(1,'mlworkloads/vision/logs/cifar10/version_2')        
