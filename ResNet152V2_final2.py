"""
Christopher Brook
10603660
MSc Health Data Science

"""
# imports for this project
import pandas as pd
import tensorflow as tf
from keras.applications import ResNet152V2
from keras.optimizers import SGD
from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Dense
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics
from sklearn.model_selection import train_test_split
import keras.metrics
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import *
from keras import backend as K
import os

# removing warnings
pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#setting up GPU learning and enabling memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print("Number of Physical GPUs =", len(gpus))
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# importing the CSV containing the location of the diseases, finding within the images and population data.
xrays_ds = pd.read_csv('C:/MSCDISS/FULL_Data_Entry_2017_updated.csv')


# creating a list of the classes
finding_labels = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema","Fibrosis",
 "Hernia","Infiltration","Mass", "No Finding","Nodule","Pleural_Thickening","Pneumonia","Pneumothorax"]

# removing incomplete samples.
finding_labels = [x for x in finding_labels]
for c_label in finding_labels:
    if len(c_label) > 1:
        xrays_ds[c_label] = xrays_ds['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)

#spliting into training and validation data, with a 80%, 20% split.
train_df, valid_df = train_test_split(xrays_ds, test_size=0.20)

# setting the size of the images.
image_size = (224, 224)
input_shape=(224, 224, 3)
# setting the batch size.
batch_size = 32
#setting the number of epochs
epochs=100

#creating one hot encoding.
train_df['one_hot_findings'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
valid_df['one_hot_findings'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)

# configuring the alterations to the images.
img_gen = ImageDataGenerator(rescale= 1./255, preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input, brightness_range=[0.2, 0.7], fill_mode='nearest',
                                  zoom_range=0.125, samplewise_std_normalization= True)

#creating the training and validation batches
train_batches = img_gen.flow_from_dataframe(train_df, x_col='location', y_col="one_hot_findings", class_mode="categorical", target_size=image_size,  batch_size=batch_size)

valid_batches = img_gen.flow_from_dataframe(train_df, x_col='location', y_col="one_hot_findings", class_mode="categorical", target_size=image_size, batch_size=64)

test_X, test_Y = next(img_gen.flow_from_dataframe(valid_df, x_col='location', y_col="one_hot_findings", class_mode="categorical", target_size=image_size, batch_size=1024))

#creation of the model and compiler.
def create_model():
    base_model = ResNet152V2(include_top=False, pooling = "max", input_shape = input_shape, classes = 15)
    print(base_model.summary())
    model = Sequential()
    model.add(base_model)
    model.add(Dense(580))
    model.add(Dropout(0.5))
    model.add(Dense(len(finding_labels), activation='sigmoid'))

    METRICS = [
          # keras.metrics.TruePositives(name='tp'),
          # keras.metrics.FalsePositives(name='fp'),
          # keras.metrics.TrueNegatives(name='tn'),
          # keras.metrics.FalseNegatives(name='fn'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='AUC'),
    ]


    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer= SGD(lr=0.01, momentum=0.01),
                  metrics= METRICS)
    return model

model = create_model()

#saving the weights of the model
filepath="weights-improvement.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                save_best_only=True, save_weights_only=True, mode='min')

# setting early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta= 0.0001,
                               patience=5, verbose=1, mode="min")

# saving the output after each epoch.
model_log = CSVLogger('ResNet150V2_output.csv', separator=",", append=False)

# altering the learning rate if it failed to improve.
learning_rate = ReduceLROnPlateau(monitor='val_loss', patience=3,
                                            verbose=1,
                                            factor=0.2,
                                            min_lr=0.0001)


callbacks = [checkpoint, early_stopping, model_log, learning_rate]

# fitting the model
fit_history = model.fit_generator(
    train_batches,
    steps_per_epoch=train_batches.n/train_batches.batch_size,
    epochs=epochs,
    validation_data=(test_X, test_Y),
    validation_steps=valid_batches.n/valid_batches.batch_size,
    callbacks= callbacks,
    workers= 1,
    shuffle=True
)

pred_y = model.predict(test_X, batch_size=1024, verbose=True)

# creating functions to evaluate the performance with the test data.
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# evaluate the preformance and assigning the variable names.
loss, accuracy, precision, recall = model.evaluate(test_X, test_Y,  verbose=2)
auc = roc_auc_score(test_Y, pred_y)

print("loss: ")
print(loss)
print("Precision: ")
print(precision)
print("Recall: ")
print(recall)
print("Accuracy: ")
print(accuracy)
print("AUC: ")
print(auc)

# # roc curve of individual classes
# Make prediction based on my fitted model
# deep_model_predictions = model.predict(test_X, verbose = 1)
# #quick_model_predictions = model.predict(test_X, batch_size = 64, verbose = 1) #2
# # import pickle

# saving the output from the model.
# # with open('y_test.pickle', 'wb') as handle:
# #     pickle.dump(test_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
# #
# # with open('y_score.pickle', 'wb') as handle:
# #     pickle.dump(deep_model_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# # *** create plots *****
# # Plot linewidth.
# lw = 2
#
# fpr_2 = dict()
# tpr_2 = dict()
# roc_auc_2 = dict()
#
#
# fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
# for (i, c_label) in enumerate(finding_labels):
#     fpr, tpr, thresholds = roc_curve(test_Y[:,i].astype(int), deep_model_predictions[:,i])
#     c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, metrics.auc(fpr, tpr)))
#     fpr_2[i] = fpr
#     tpr_2[i] = tpr
#     roc_auc_2[i] = metrics.auc(fpr_2[i],tpr_2[i])
#
# # Set labels for plot
# c_ax.legend()
# c_ax.set_xlabel('False Positive Rate')
# c_ax.set_ylabel('True Positive Rate')
# fig.savefig('deep_trained_model.png')
#
#
# # Compute micro-average ROC curve and ROC area
# fpr_2["micro"], tpr_2["micro"], _ = roc_curve(test_Y.ravel(), deep_model_predictions.ravel())
# roc_auc_2["micro"] = metrics.auc(fpr_2["micro"], tpr_2["micro"])
#
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr_2[i] for i in range(len(finding_labels))]))
#
# # Then interpolate all ROC curves at this points
#
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(len(finding_labels)):
#     mean_tpr += interp(all_fpr, fpr_2[i], tpr_2[i])
#
# # Finally average it and compute AUC
# mean_tpr /= len(finding_labels)
#
# fpr_2["macro"] = all_fpr
# tpr_2["macro"] = mean_tpr
# roc_auc_2["macro"] = metrics.auc(fpr_2["macro"], tpr_2["macro"])
#
#
# # Plot all ROC curves
# plt.figure(1)
# plt.plot(fpr_2["micro"], tpr_2["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc_2["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
#
# plt.plot(fpr_2["macro"], tpr_2["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc_2["macro"]),
#          color='navy', linestyle=':', linewidth=4)
#
# # colours = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# # for i, colour in zip(range(len(finding_labels)), colours):
# #     plt.plot(fpr_2[i], tpr_2[i], color=colour, lw=lw,
# #              label='ROC curve of class {0} (area = {1:0.2f})'
# #              ''.format(i, roc_auc_2[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# #plt.show()
# plt.savefig('deep_trained_model_2.png')
