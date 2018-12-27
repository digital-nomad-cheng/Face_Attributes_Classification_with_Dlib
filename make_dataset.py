import os
import timeit
import cv2
from skimage import io as io
import face_recognition as fr
import numpy as np
import pickle
from tqdm import tqdm
from sklearn import datasets, svm, metrics
import pandas as pd
data_path = "/home/ubuntu/ihandy_seg/data/img_align_celeba" 
table = pd.read_csv("/home/ubuntu/ihandy_seg/data/list_attr_celeba.csv")
print(table.columns.values)
img_label = table.set_index('image_id').to_dict()["Smiling"]
def main():
    count = 0
    gender_data = list()
    img_files = [f for f in os.listdir(data_path) if not f.startswith('.')]
    for img_file in tqdm(img_files[0:5000]):
        try:
            # print('Processing {}'.format(img_file))
            print(os.path.join(data_path, img_file))
            img = io.imread(os.path.join(data_path, img_file))
            face_embedding = fr.face_encodings(img)
            if len(face_embedding) != 1:
                print("Failed")
                continue
            single_data = list()
            single_data.append(face_embedding[0])
            single_data.append(img_label[img_file])
            gender_data.append(single_data)
        except Exception as e:
            print(e)
            continue
    print('Saving as a pkl file')
    with open('celeba_smiling_data.pkl','wb') as f:
        pickle.dump(gender_data, f)
    print('Finished')

if __name__ == '__main__':
    main()



