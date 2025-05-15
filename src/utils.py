import pickle



def load_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pickle.UnpicklingError:
        print("Error: The file could not be unpickled.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def save_to_pickle(data, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def save_pickle_file(data, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage:
# data_to_save = {'key': 'value'}
# save_pickle_file(data_to_save, '/path/to/your/file.pkl')


def read_lzma_file(file_path):
    try:
        with lzma.open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except lzma.LZMAError:
        print("Error: The file could not be decompressed.")
    except pickle.UnpicklingError:
        print("Error: The file could not be unpickled.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


#functio to read csv
def read_csv_file(file_path):
    try:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except csv.Error:
        print("Error: The file could not be read as CSV.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

#read xlsx file

def read_xlsx_file(file_path, header = None):
    try:
        data = pd.read_excel(file_path, header=header)
        #data = pd.read_excel(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except ValueError:
        print("Error: The file could not be read as an Excel file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

#save to csv

def save_to_csv(data, file_path):
    try:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")