def tidy_up_report(log_out_folder):
        import pandas as pd
        import os
        import yaml

        #Check if a file called 'metrics.csv' exits in log_out_folder
        metrics_file_path = os.path.join(log_out_folder, 'metrics.csv')

        if os.path.isfile(metrics_file_path):
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
                dicts[var_name] = var_dict_no_nan
        
            # Read 'hparams.yaml' file
            hparams_file_path = os.path.join(log_out_folder, 'hparams.yaml')
            hparams_data = {}
            if os.path.isfile(hparams_file_path):
                with open(hparams_file_path, 'r') as hparams_file:
                    hparams_data = yaml.safe_load(hparams_file)

            # Save dictionaries to Excel file with each dictionary as a sheet
            excel_file_path =  os.path.join(log_out_folder, 'job_report.xlsx') 

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

if __name__ == "__main__":
    tidy_up_report('mlworkloads/vision/logs/cifar10/version_6')
    pass
        
