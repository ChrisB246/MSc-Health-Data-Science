"""
Christopher Brook
10603660
MSc Health Data Science

"""
# imports for this project
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, StandardScaler
import cv2

# importing the CSV containing the location of the diseases, finding within the images and population data.
xrays_ds = pd.read_csv('C:/MSCDISS/FULL_Data_Entry_2017_updated.csv')

#setting the image size
image_size = 226

# collecting then converting the images to a 2d then 1d array.
def get_data():
    array_img_list = []
    for img_path in tqdm(xrays_ds['location']):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        np_img = np.asarray(img)
        np_img = np_img.ravel()
        array_img_list.append(np_img)

    image = np.asarray(array_img_list)

    return image
print("images to arrays")
image = get_data()
# doing standard scaling
scaler = StandardScaler()
image = scaler.fit_transform(image)

print("finding codes")

# creating a dict so the classes have a number.
finding_dict = {
          'Atelectasis': '0',
          'Cardiomegaly': '1',
          'Consolidation': '2',
          'Edema': '3',
          'Effusion': '4',
          'Emphysema': '5',
          'Fibrosis': '6',
          'Hernia': '7',
          'Infiltration': '8',
          'Mass': '9',
          'No Finding': '10',
          'Nodule': '11',
          'Pleural_Thickening': '12',
          'Pneumonia': '13',
          'Pneumothorax': '14'}

img_class = []
# converting findings to a class number.
for img_f in tqdm(xrays_ds['Finding Labels']):
    img_f = img_f.split('|', 1)[0]
    class_num = finding_dict[img_f]
    img_class.append(class_num)

# converting to a numpy array.
img_class = np.asarray(img_class, dtype=np.float32)

# creating one hot encoding for the classes
img_class = label_binarize(img_class, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

# setting the number of classes for plots later on
n_classes = img_class.shape[1]

#settign the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(image, img_class, test_size=0.2)

# setting the classification model.
logisticRegr = OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver= 'lbfgs', max_iter=60000))

print("fitting model")

# fitting the model just created
y_score = logisticRegr.fit(X_train, y_train).decision_function(X_test)


# predicting using the test data, to assess the accuracy of the model
y_pred = logisticRegr.predict_proba(X_test)

auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr')
score = logisticRegr.score(X_test, y_test)

print("auc score:", auc_score)
print("Score", score)



# Compute ROC curve and ROC area for each class
y_test_lr = y_test

y_score_lr = y_score

# saving the output for plotting on another script.
# import pickle
# with open('y_test_LR.pickle', 'wb') as handle:
#     pickle.dump(y_test_lr, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('y_score_LR.pickle', 'wb') as handle:
#     pickle.dump(y_score_lr, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("done")