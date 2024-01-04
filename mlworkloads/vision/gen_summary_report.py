import glob
import pandas as pd
import os

def create_job_report(job_id, folder_path):
    # Check if a file called 'summary_report.xlsx' exists in the folder
    output_file_path = os.path.join(folder_path, 'summary_report.xlsx')

    # Get a list of all CSV files in the specified folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # Specify column categories
    categories = ['train/batch', 'train/epoch', 'train/job', 'val/batch', 'val/epoch', 'val/job']

    # Create a dictionary to store accumulated data for each category
    category_data = {category: {} for category in categories}

    # Iterate through each CSV file
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Iterate through each category
        for category in categories:
            # Select columns starting with the current category
            selected_columns = [col for col in df.columns if col.startswith(category)]

            # Update the dictionary for the current category with the selected columns
            if selected_columns:

                data_dict = df[selected_columns].to_dict(orient='list')

                # Remove NaN values from the dictionary
                data_dict_no_nan = {key: [value for value in values if pd.notna(value)] for key, values in data_dict.items()}

                # Accumulate data for the current category across all CSV files
                if category_data[category]:
                    for key, values in data_dict_no_nan.items():
                        category_data[category][key].extend(values)
                else:
                    category_data[category] = data_dict_no_nan

    # Create an Excel writer for the output file
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        # Iterate through each category
        for category, data_dict in category_data.items():
            # Skip creating sheets if the data for the current category is empty
            if data_dict:
                # Replace invalid characters in the sheet name
                clean_category = category.replace('/', '_').replace('\\', '_').replace('*', '').replace('?', '').replace(':', '').replace('[', '').replace(']', '')

                # Convert the dictionary to a DataFrame
                df_sorted = pd.DataFrame.from_dict(data_dict, orient='columns')

                # Sort the DataFrame by the 'timestamp' column
                timestamp_column = next((col for col in data_dict.keys() if 'timestamp' in col.lower()), None)
                if timestamp_column:
                    df_sorted.sort_values(by=timestamp_column, inplace=True)
                                # Change the data type of 'train/batch-id' column to string
                if 'train/batch-id' in df_sorted.columns:
                    df_sorted['train/batch-id'] = df_sorted['train/batch-id'].astype(str).apply(lambda x: "{:.0f}".format(float(x)) if pd.notna(x) else x)
                # Convert the sorted DataFrame back to a dictionary
                sorted_data_dict = df_sorted.to_dict(orient='list')

                # Write the sorted and accumulated data for the current category to a separate sheet in the Excel file
                pd.DataFrame.from_dict(sorted_data_dict, orient='columns').to_excel(writer, sheet_name=clean_category, index=False)

if __name__ == "__main__":
    create_job_report(1, 'mlworkloads/vision/logs/cifar10/version_3')
