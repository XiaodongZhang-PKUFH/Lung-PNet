import SimpleITK as sitk
from faimed3d.all import *
from ssl_augment import ssl_transforms_3d
from torchvision.models.video import r3d_18
from fastai.distributed import *
from fastai.callback.all import *
from sklearn.model_selection import train_test_split


def dice(input, target):
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return ((2. * intersection) /
           (iflat.sum() + tflat.sum()))

def dice_score(input, target):
    return dice(input.argmax(1), target)

def dice_loss(input, target):
    return 1 - dice(input.softmax(1)[:, 1], target)

def loss(input, target):
    return dice_loss(input, target) + nn.CrossEntropyLoss()(input, target[:, 0])

def accuracy_seg(input, targs):
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()

class SegCrossEntropy(nn.CrossEntropyLoss):
    def forward(self, input, target):
        n,c,*_ = input.shape
        return super().forward(input.view(n, c, -1), target.view(n, -1))


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
labelID = 5
arch = r3d_18  # r3d_18,mc3_18,r2plus1d_18, resnet34_3d
num_epoch1 = 300
num_epoch2 = 700


def main():

    # Phase Stage 1
    dls = SegmentationDataLoaders3D.from_df(train_df,
                                            path=IMG_PATH,
                                            valid_col='Cat',
                                            fn_col=dataID,
                                            label_col=labelID,
                                            item_tfms=ResizeCrop3D((0, 0, 0), (16, 32, 32)),
                                            batch_tfms=[RandomPerspective3D(32, 0.5),
                                                        *aug_transforms_3d(p_all=0.15, noise=False)],
                                            bs=4,
                                            val_bs=4)

    learn = unet_learner_3d(dls, arch, n_out=2,
                            loss_func=loss,
                            metrics=[dice_score],
                            opt_func=ranger,
                            model_dir=model_dir
                            )
    learn.model[0].load_state_dict(state_dict_r3d18);

    if n_gpu > 1:
        learn.model = nn.DataParallel(learn.model)

    learn.to_fp16()


    learn.fit_flat_cos(30, 3e-3)
    learn.unfreeze()
    learn.fit_flat_cos(num_epoch1, lr=slice(3e-6, 3e-4))
    learn.save(f'model_seg_{arch.__name__}_stage1')


    # Phase Stage 2
    dls = SegmentationDataLoaders3D.from_df(train_df,
                                            path=IMG_PATH,
                                            valid_col='Cat',
                                            fn_col=dataID,
                                            label_col=labelID,
                                            item_tfms=ResizeCrop3D((0, 0, 0), (32, 64, 64)),
                                            batch_tfms=[RandomPerspective3D(32, 0.5),
                                                        *aug_transforms_3d(p_all=0.15, noise=False)],
                                            bs=4,
                                            val_bs=4)

    learn = unet_learner_3d(dls, arch, n_out=2,
                            loss_func=loss,
                            opt_func=ranger,
                            metrics=[dice_score],
                            model_dir=model_dir
                            )
    if n_gpu > 1:
        learn.model = nn.DataParallel(learn.model)

    learn.to_fp16()
    learn = learn.load(f'model_seg_{arch.__name__}_stage1')
    learn.fit_flat_cos(30, 3e-3)
    learn.unfreeze()
    learn.fit_flat_cos(num_epoch2, lr=slice(1e-6, 1e-4))
    learn.save(f'final_seg_model_pggn_{arch.__name__}_swn_{clip[0]}_{clip[1]}')
    

    # your data path
 
    return None

if __name__=="__main__":
    main()


