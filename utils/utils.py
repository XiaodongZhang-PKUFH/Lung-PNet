import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
# from keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn import metrics as me
from sklearn.metrics import confusion_matrix


def to_categorical(y, num_classes=None, dtype='float32'):
  """Converts a class vector (integers) to binary class matrix.
  """
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical

    
def plot_feature_importances(model, columns):
    nr_f = 10
    imp = pd.Series(data = model.best_estimator_.feature_importances_, 
                    index=columns).sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    plt.title("Feature importance")
    ax = sns.barplot(y=imp.index[:nr_f], x=imp.values[:nr_f], orient='h')    

def show_confusion_matrix_classification_report(truth,pred_class,target_names=['0','1'],title='Confusion matrix'):

    cm = (confusion_matrix(truth,pred_class))
    
    print ("The f1-score gives you the harmonic mean of precision and recall. The scores corresponding to every class will tell you the accuracy of the classifier in classifying the data points in that particular class compared to all other classes.The support is the number of samples of the true response that lie in that class.")
    print(me.classification_report( truth, pred_class, target_names=target_names,digits=4))
    print ('Summary')
    print ("Precision: ", (me.precision_score( truth, pred_class,average='weighted')))
    print ("Recall: ", (me.recall_score( truth, pred_class,average='weighted')))
    print ("F1 Score: ", me.f1_score( truth, pred_class,average='weighted'))
    plot_confusion_matrix(cm, classes=target_names,title=title)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()    
#Spot-Check functions

def createROCplots(Y_true, Y_cv_pred ):
	lw = 2

	Y_true=np.asarray(Y_true)
	Y_cv_pred=np.asarray(Y_cv_pred)
	fpr = dict()
	tpr = dict()		
	n_classes=Y_true.max()
	n_classes=n_classes+1
	roc_auc = dict()
	y_test=to_categorical(Y_true, num_classes=n_classes+1)
	y_score=Y_cv_pred
	# y_score=to_categorical(Y_cv_pred, num_classes=n_classes+1)
	for i in range(n_classes):
	    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	# Plot all ROC curves
	plt.figure()

	plt.plot(fpr["macro"], tpr["macro"],
	         label='macro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["macro"]),
	         color='navy', linestyle=':', linewidth=4)

	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(n_classes), colors):
	    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
	             label='ROC curve of class {0} (area = {1:0.2f})'
	             ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	plt.show()  
