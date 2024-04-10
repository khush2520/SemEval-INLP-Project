# import pandas as pd

# df_tokenized = pd.read_pickle('Datasets/task1_tokenized.pkl')

# print(df_tokenized)

import os
import pickle



def get_combined_data(files_directory):
    combined_data = []
    # files_directory = 'Datasets/scienceie2017_train/train2'

    txt_files = [file for file in os.listdir(files_directory) if file.endswith('.txt')]

    for txt_file in txt_files:
        txt_filename = os.path.join(files_directory, txt_file)
        ann_filename = os.path.join(files_directory, txt_file[:-3] + 'ann') 

        if os.path.exists(ann_filename):
            with open(txt_filename, 'r') as txt_file_handle:
                txt_content = txt_file_handle.read().strip()  # Read entire text content
                

            with open(ann_filename, 'r') as ann_file_handle:
                # Define a list to store annotations for the current file
                annotations = []

                for line in ann_file_handle:
                    if line.strip() and line.strip()[0] == "T":

                        # print(line.strip())
                        
                        parts = line.strip().split('\t')
                        if len(parts) == 3:
                            # Extract annotation type, start, end, and text
                            annotation_type = parts[0]
                            text = parts[2]  
                            if (';' in parts[1]):
                                two_parts = parts[1].split(';')
                                # print(two_parts)
                                ktype = two_parts[0].split()[0]
                                start1 = int(two_parts[0].split()[1])
                                start2 = int(two_parts[1].split()[0])
                                end1 = int(two_parts[0].split()[2])
                                end2 = int(two_parts[1].split()[1])
                                annotations.append([annotation_type, ktype, start1, end1, text])
                                annotations.append([annotation_type, ktype, start2, end2, text])
                                # print([annotation_type, start2, end2, text])
                                # print(line.strip())

                            else:
                                start = int(parts[1].split()[1])  # e.g., 0 from "Process 0 30"
                                end = int(parts[1].split()[2])  # e.g., 30 from "Process 0 30"
                                ktype = parts[1].split()[0]
                                
                                annotations.append([annotation_type, ktype, start, end, text])
                                # print([annotation_type, ktype, start, end, text])
                                # print(line.strip())
                            
            combined_data.append([txt_content, annotations])
    return combined_data

def write_to_pickle(filename, combined_data):
    with open(filename, "wb") as pickle_file:
        
        pickle.dump(combined_data, pickle_file)


def load_from_pickle(filename):
    with open(filename, "rb") as pickle_file:
        loaded_combined_data = pickle.load(pickle_file)
    return loaded_combined_data

train_data = get_combined_data('Datasets/scienceie2017_train/train2')
write_to_pickle("train_data.pkl", train_data)