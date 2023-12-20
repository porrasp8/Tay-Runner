import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def read_list_from_file(file_path):
    return np.load(file_path)

def plot_list(data):
    plt.plot(data, marker='o', color='blue', linestyle='None')
    plt.title('Plotting Data from File')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    # Finds the directory
    folder_path = os.getcwd()

    if len(sys.argv) > 1:
        folder_path = sys.argv[1]

    file_name = 'rewards.npy'
    file_path = os.path.join(folder_path, file_name)

    # Check if the file exists
    if os.path.exists(file_path):

        read_data = read_list_from_file(file_path)
        plot_list(read_data)

    else:
        print(f"Error: File {file_name} not found in the specified folder or current directory.")
