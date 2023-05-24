import SimpleITK as sitk
from faimed3d.all import *
from ssl_augment import ssl_transforms_3d
from torchvision.models.video import r3d_18
from fastai.distributed import *
from fastai.callback.all import *


gpu = None
if torch.cuda.is_available():
    if gpu is not None: torch.cuda.set_device(gpu)
    n_gpu = torch.cuda.device_count()
else:
    n_gpu = None

PROCESSED_DATA = Path('./Dataset/ssl_data')
model_dir = Path('./pre_trained')

IMG_PATH = PROCESSED_DATA
train_df = pd.read_csv(PROCESSED_DATA/'data_pid_patch_ssl.csv')

dataID = 1
labelID = 1
arch = r3d_18 # resnet18_3d, resnet34_3d, resnet50_3d
num_epoch1 = 30 # 30
num_epoch2 = 1000 # 1000


def main():

    # Phase Stage 1
    size = (16,16,16) # (32,64,64)#
    bs =  1 # 64 16 # 
    item_tfms = Resize3D2(size=size)
    batch_tfms= ssl_transforms_3d(p_all = 0.8)


    dls = SegmentationDataLoaders3D.from_df(train_df, 
                                            path = IMG_PATH,
                                            valid_col = 'Cat',
                                            fn_col = dataID, 
                                            item_tfms = item_tfms, 
                                            batch_tfms= batch_tfms,
                                            bs = bs, 
                                            val_bs = bs)


    learn = unet_learner_3d(dls, 
                            arch, 
                            n_out=1, 
                            model_dir = model_dir,
                            loss_func=MSELossFlat()
                           )

    if n_gpu > 1:
        learn.model = nn.DataParallel(learn.model)

    learn.to_fp16()


    learn.fit_flat_cos(num_epoch1, 3e-3)
    learn.unfreeze()
    learn.fit_flat_cos(num_epoch2, lr=slice(3e-6, 3e-4))    
    learn.save(f'model_ssl_{arch.__name__}_stage1')


    # Phase Stage 2
    size = (16,32,32)
    bs = 64
    item_tfms = Resize3D2(size=size)
    batch_tfms= ssl_transforms_3d(p_all = 0.3)

    dls = SegmentationDataLoaders3D.from_df(train_df, 
                                            path = IMG_PATH,
                                            valid_col = 'Cat',
                                            fn_col = dataID, 
                                            item_tfms = item_tfms, 
                                            batch_tfms= batch_tfms,
                                            bs = bs, 
                                            val_bs = bs)


    learn = unet_learner_3d(dls, 
                            arch, 
                            n_out=1, 
                            model_dir = model_dir,
                            loss_func=MSELossFlat()
                           )
    if n_gpu > 1:
        learn.model = nn.DataParallel(learn.model)

    learn.to_fp16()
    learn = learn.load(f'model_ssl_{arch.__name__}_stage1')
    learn.fit_flat_cos(num_epoch1, 3e-4)
    learn.unfreeze()
    learn.fit_flat_cos(num_epoch2, lr=slice(1e-7, 1e-5))
    learn.save(f'model_ssl_{arch.__name__}_stage2')

    # Phase Stage 3
    size = (16,32,32)
    bs = 64
    item_tfms = Resize3D2(size=size)
    batch_tfms= ssl_transforms_3d(p_all = 0.3)


    dls = SegmentationDataLoaders3D.from_df(train_df, 
                                            path = IMG_PATH,
                                            valid_col = 'Cat',
                                            fn_col = dataID, 
                                            item_tfms = item_tfms, 
                                            batch_tfms= batch_tfms,
                                            bs = bs, 
                                            val_bs = bs)
    learn = unet_learner_3d(dls, 
                            arch, 
                            n_out=1, 
                            model_dir = model_dir,
                            loss_func=MSELossFlat()
                           )
    if n_gpu > 1:
        learn.model = nn.DataParallel(learn.model)

    learn.to_fp16()
    learn = learn.load(f'model_ssl_{arch.__name__}_stage2')
    learn.fit_flat_cos(num_epoch1, 3e-4)
    learn.unfreeze()
    learn.fit_flat_cos(num_epoch2, lr=slice(1e-7, 1e-5))
    torch.save(learn.model[0].state_dict(), model_dir/f'{arch.__name__}_ssl_state_dict.pth')
    

    # your data path
 
    return None

if __name__=="__main__":
    main()


