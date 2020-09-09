"""
Christopher Brook
10603660
MSc Health Data Science

"""
# imports for this project

import pickle
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, brier_score_loss
import numpy as np

# with open('y_testCNN.pickle', 'rb') as handle:
#     y_testCNN = pickle.load(handle)
#
# with open('y_scoreCNN.pickle', 'rb') as handle:
#     y_scoreCNN = pickle.load(handle)

with open('y_test.pickle', 'rb') as handle:
    y_test = pickle.load(handle)

with open('y_score.pickle', 'rb') as handle:
    y_score = pickle.load(handle)

y_score = (y_score > 0.5)

np.argmax(y_score, axis=1)

prec = precision_score(y_test, y_score, average='micro')

rec = recall_score(y_test, y_score, average='micro')

acc = accuracy_score(y_test, y_score)

loss = sklearn.metrics.log_loss(y_test, y_score)


print(acc)
print(prec)
print(rec)
print(loss)

## creating the plot for both models.

# n_classes = 15
#
# fpr_CNN = dict()
# tpr_CNN = dict()
# roc_auc_CNN = dict()
#
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
#
#
# for i in range(n_classes):
#     if i == 15:
#         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#         fpr[i], tpr[i], _ = roc_curve(y_testCNN[:, i], y_scoreCNN[:, i])
#         roc_auc_CNN[i] = auc(fpr_CNN[i], tpr_CNN[i])
#     else:
#         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#         fpr_CNN[i], tpr_CNN[i], _ = roc_curve(y_testCNN[:, i], y_scoreCNN[:, i])
#         roc_auc[i] = auc(fpr_CNN[i], tpr_CNN[i])
#

# # Plot of a ROC curve for a specific class
# for i in range(n_classes):
#     plt.figure()
#     plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(i)
#     plt.legend(loc="lower right")
# plt.show()

# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='red',
#          lw=lw, label='ROC curve for Logistic Regression (area = %0.0f)' % roc_auc[2])
# plt.plot(fpr_CNN[2], tpr_CNN[2], color='green',
#          lw=lw, label='ROC curve for Pre-trained CNN (area = %0.0f)' % roc_auc_CNN[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='')
# ax = plt.gca()
# ax.set_facecolor('grey')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic plot')
# plt.legend(loc="lower right")
# plt.show()
