import SimpleITK as sitk
from faimed3d.all import *
from ssl_augment import ssl_transforms_3d
from torchvision.models.video import r3d_18
from fastai.distributed import *
from fastai.callback.all import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


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


def show_confusion_matrix_classification_report(truth, pred_class, target_names=['0', '1'], title='Confusion matrix'):
    cm = (confusion_matrix(truth, pred_class))

    print(
        "The f1-score gives you the harmonic mean of precision and recall. The scores corresponding to every class will tell you the accuracy of the classifier in classifying the data points in that particular class compared to all other classes.The support is the number of samples of the true response that lie in that class.")
    print(me.classification_report(truth, pred_class, target_names=target_names, digits=4))
    print('Summary')
    print("Precision: ", (me.precision_score(truth, pred_class, average='weighted')))
    print("Recall: ", (me.recall_score(truth, pred_class, average='weighted')))
    print("F1 Score: ", me.f1_score(truth, pred_class, average='weighted'))
    plot_confusion_matrix(cm, classes=target_names, title=title)


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



gpu = None
if torch.cuda.is_available():
    if gpu is not None: torch.cuda.set_device(gpu)
    n_gpu = torch.cuda.device_count()
else:
    n_gpu = None

PROCESSED_DATA = Path('./Dataset/csv_data')
model_dir = Path('./pre_trained')

IMG_PATH = PROCESSED_DATA
train_df = pd.read_csv(PROCESSED_DATA/'data_pid_patch_ssl.csv')

path_model_state_dict = Path('./pre_trained')
state_dict_r3d18 = torch.load(path_model_state_dict/'r3d_18_ssl_state_dict.pth')

clip = [50,50]
train_df = pd.read_csv(PROCESSED_DATA/f'random_split_pggn_train_swn_{clip[0]}_{clip[1]}.csv')
test_df = pd.read_csv(PROCESSED_DATA/f'random_split_pggo_test_swn_{clip[0]}_{clip[1]}.csv')

dataID = 4
labelID = 26
arch = r3d_18 # r3d_18,mc3_18,r2plus1d_18, resnet34_3d, resnet50_3d,resnet101_3d
num_epoch = 100


def main():

    # Phase Stage 1
    dls = ImageDataLoaders3D.from_df(train_df,
                                     path=IMG_PATH,
                                     fn_col=dataID,
                                     label_col=labelID,
                                     valid_col='Cat',
                                     item_tfms=[ResizeCrop3D(crop_by=(0, 0, 0),  # don't crop the images
                                                             resize_to=(16, 32, 32)),
                                                *TioTransforms(p_all=0.2)
                                                ],
                                     batch_tfms=aug_transforms_3d(p_all=0.2),
                                     # all tfms with p = 0.2 for each tfms to get called
                                     bs=16, val_bs=16)

    learn = cnn_learner_3d(dls,
                           arch,
                           metrics=[accuracy, F1Score(), RocAucBinary()],
                           opt_func=ranger,
                           loss_func=LabelSmoothingCrossEntropy(),
                           model_dir=model_dir
                           )
    learn.model[0].load_state_dict(state_dict_r3d18);

    if n_gpu > 1:
        learn.model = nn.DataParallel(learn.model)

    learn.to_fp16()


    learn.fit_flat_cos(30, 3e-3)
    learn.unfreeze()
    learn.fit_flat_cos(num_epoch, lr=slice(3e-6, 3e-4))
    learn.save(f'model_class_{arch.__name__}_stage1')


    # Phase Stage 2
    dls = ImageDataLoaders3D.from_df(train_df,
                                     path=IMG_PATH,
                                     fn_col=dataID,
                                     label_col=labelID,
                                     valid_col='Cat',
                                     item_tfms=[ResizeCrop3D(crop_by=(0, 0, 0),  # don't crop the images
                                                             resize_to=(32, 64, 64)),
                                                *TioTransforms(p_all=0.2)
                                                ],
                                     batch_tfms=aug_transforms_3d(p_all=0.2),
                                     # all tfms with p = 0.2 for each tfms to get called
                                     bs=8, val_bs=8)

    learn = cnn_learner_3d(dls,
                           arch,
                           metrics=[accuracy, F1Score(), RocAucBinary()],
                           opt_func=ranger,
                           loss_func=LabelSmoothingCrossEntropy(),
                           model_dir=model_dir
                           )
    if n_gpu > 1:
        learn.model = nn.DataParallel(learn.model)

    learn.to_fp16()
    learn = learn.load(f'model_clas_{arch.__name__}_stage1')
    learn.fit_flat_cos(30, 3e-3)
    learn.unfreeze()
    learn.fit_flat_cos(num_epoch, lr=slice(1e-6, 1e-4))
    learn.save(f'final_clas_model_pggn_{arch.__name__}_swn_{clip[0]}_{clip[1]}')
    

    # confusion matrix, ROCplot
    preds, y, losses = learn.get_preds(with_loss=True, ds_idx=1)
    pred_class = preds.argmax(dim=1)
    truth = y.numpy()
    predictedoutput = preds.numpy()
    createROCplots(truth, predictedoutput)
    show_confusion_matrix_classification_report(truth, pred_class, target_names=['IAC = 0', 'nonIAC = 1'],
                                                title='Confusion matrix - Valid')
 
    return None

if __name__=="__main__":
    main()


