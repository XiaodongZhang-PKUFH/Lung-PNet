
__all__ = ['TorchIOTransform', 'TioTransforms', 'TioRandomBiasField', 'TioRandomGhosting',
           'TioRandomMotion', 'TioRandomGamma', 'TioRandomAnisotropy', 'TioRandomElasticDeformation',
           'Resize3D2','crop', 'CenterCrop', 'RandomCrop', 'bernstein_poly', 'bezier_curve',
           'RandomBezierContrast3D', 'RandomLocalPixelShuffling3D', 'RandomImageInPainting3D',
           'RandomImageOutPainting3D', 'ssl_transforms_3d']

# Cell
# default_exp augment
from fastai.torch_basics import *
from fastai.basics import *
from fastai.vision.augment import *
import torchvision.transforms.functional as _F
import torchvision
import SimpleITK as sitk
import torchio as tio
import copy
from scipy.special import comb

# Cell
from .basics import *


# Cell
@patch
def gaussian_noise(t:(TensorDicom3D), std):
    noise = torch.randn(t.shape).to(t.device)
    return t + (std**0.5)*noise*t

class RandomNoise3D(RandTransform):
    def __init__(self, p=0.5, std_range=[0.01, 0.1]):
        super().__init__(p=p)
        self.lwr_std = np.min(std_range)
        self.upr_std = np.max(std_range)

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.std = random.choice(np.arange(self.lwr_std,
                                           self.upr_std,
                                           self.lwr_std))
    def encodes(self, x:(TensorDicom3D)):
        return x.gaussian_noise(self.std)

    def encodes(self, x:TensorMask3D): return x

# Cell
class RandomBlur3D(RandTransform):
    def __init__(self, p=0.5, kernel_size_range=[5, 11], sigma=0.5):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        sizes = range(self.kernel_size_range[0],
                      self.kernel_size_range[1],
                      2)
        self.kernel = random.choices(sizes, k = 2)

    def encodes(self, x:(TensorDicom3D)):
        return _F.gaussian_blur(x, self.kernel, self.sigma)

    def encodes(self, x:TensorMask3D): return x

# Cell

@patch
def rescale(t: TensorDicom3D, new_min = 0, new_max = 1):
    return (new_max - new_min)/(t.max()-t.min()) *(t - t.max()) + t.max()

@patch
def adjust_brightness(x:TensorDicom3D, beta):
    return torch.clamp(x + beta, x.min(), x.max())


class RandomBrightness3D(RandTransform):
    def __init__(self, p=0.5, beta_range=[-0.3, 0.3], beta_step=25):
        super().__init__(p=p)
        self.lwr_beta = np.min(beta_range)
        self.upr_beta = np.max(beta_range)
        self.step_beta = beta_step

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.beta = random.choice(np.arange(self.lwr_beta,
                                           self.upr_beta,
                                           self.step_beta))
    def encodes(self, x:TensorDicom3D):
        return x.adjust_brightness(self.beta)

    def encodes(self, x:TensorMask3D): return x

# Cell
@patch
def adjust_contrast(x:TensorDicom3D, alpha):
    x2 = x*alpha
    min = x2.min() + abs(x2.min() - x.min())
    return torch.clamp(x2, min, x.max())



class Resize3D2(RandTransform):
    split_idx,order = None, 1
    "Resize a 3D image"

    def __init__(self, size, scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None, **kwargs):
        store_attr()
        self.size = _process_sz_3d(self.size)
        super().__init__(**kwargs)

    def encodes(self, x: (TensorDicom3D)):
        dev=x.device
        x = x.resize_3d(self.size, self.scale_factor, self.mode, self.align_corners, self.recompute_scale_factor)
        return x.to(dev)

    def encodes(self, x: (TensorMask3D)):
        dev=x.device
        x = x.resize_3d(self.size, self.scale_factor, self.mode, self.align_corners, self.recompute_scale_factor)
        return x.to(dev)


# Cell
def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


# https://github.com/pytorch/pytorch/issues/1552
def interpolate(x, xp, fp):
    i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
    return (fp[i - 1] *  (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1])

@patch
def bezier_curve_trans(x: (TensorDicom3D, TensorMask3D)):
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)

    # x = np.interp(x.cpu().numpy(), xvals, yvals)
    x = interpolate(x,torch.from_numpy(xvals).cuda(),torch.from_numpy(yvals).cuda())
    return TensorDicom3D.create(x)

class RandomBezierContrast3D(RandTransform):
    split_idx, p = None, 1

    def __init__(self, p=1):
        super().__init__(p=p)

    def __call__(self, b, split_idx=None, **kwargs):
        "change in __call__ to enforce, that the Transform is always applied on every dataset. "
        return super().__call__(b, split_idx=split_idx, **kwargs)

    def encodes(self, x:(TensorDicom3D)):
        return x.bezier_curve_trans()
    def encodes(self, x:TensorMask3D):
        return x

# Cell
@patch
def local_pixel_shuffling(x: (TensorDicom3D, TensorMask3D)):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_bat, img_deps, img_rows, img_cols = x.shape #input image with dimension B x D x H x W

    num_block = 100
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//5)
        block_noise_size_y = random.randint(1, img_cols//5)
        block_noise_size_z = random.randint(1, img_deps//5)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[:,
                            noise_z:noise_z+block_noise_size_z,
                            noise_x:noise_x+block_noise_size_x,
                            noise_y:noise_y+block_noise_size_y,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((img_bat,
                                 block_noise_size_z,
                                 block_noise_size_x,
                                 block_noise_size_y
                                 ))
        image_temp[:,
                   noise_z:noise_z+block_noise_size_z,
                   noise_x:noise_x+block_noise_size_x,
                   noise_y:noise_y+block_noise_size_y
                      ] = window
    local_shuffling_x = image_temp
    return local_shuffling_x

class RandomLocalPixelShuffling3D(RandTransform):
    split_idx, p = None, 1

    def __init__(self, p=1):
        super().__init__(p=p)

    def __call__(self, b, split_idx=None, **kwargs):
        "change in __call__ to enforce, that the Transform is always applied on every dataset. "
        return super().__call__(b, split_idx=split_idx, **kwargs)

    def encodes(self, x:(TensorDicom3D)):
        return x.local_pixel_shuffling()
    def encodes(self, x:TensorMask3D):
        return x

# Cell
@patch
def image_in_painting(x: (TensorDicom3D, TensorMask3D)):

    img_bat,img_deps, img_rows, img_cols = x.shape #input image with dimension B x D x H x W

    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:,
          noise_z:noise_z+block_noise_size_z,
          noise_x:noise_x+block_noise_size_x,
          noise_y:noise_y+block_noise_size_y] = TensorDicom3D.create(np.random.rand(img_bat,block_noise_size_z,
                                                               block_noise_size_x,
                                                               block_noise_size_y, ) * 1.0)
        cnt -= 1
    return x

class RandomImageInPainting3D(RandTransform):
    split_idx, p = None, 1

    def __init__(self, p=1):
        super().__init__(p=p)

    def __call__(self, b, split_idx=None, **kwargs):
        "change in __call__ to enforce, that the Transform is always applied on every dataset. "
        return super().__call__(b, split_idx=split_idx, **kwargs)

    def encodes(self, x:(TensorDicom3D)):
        return x.image_in_painting()
    def encodes(self, x:TensorMask3D):
        return x

# Cell
@patch
def image_out_painting(x: (TensorDicom3D, TensorMask3D)):

    img_bat, img_deps, img_rows, img_cols = x.shape #input image with dimension B x D x H x W

    image_temp = copy.deepcopy(x)
    x = TensorDicom3D.create(np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0)
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    noise_z = random.randint(3, img_deps-block_noise_size_z-3)
    x[:,
      noise_z:noise_z+block_noise_size_z,
      noise_x:noise_x+block_noise_size_x,
      noise_y:noise_y+block_noise_size_y] = image_temp[:,
                                                       noise_z:noise_z+block_noise_size_z,
                                                       noise_x:noise_x+block_noise_size_x,
                                                       noise_y:noise_y+block_noise_size_y]
    cnt = 100
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:,
          noise_z:noise_z+block_noise_size_z,
          noise_x:noise_x+block_noise_size_x,
          noise_y:noise_y+block_noise_size_y] = image_temp[:,
                                                           noise_z:noise_z+block_noise_size_z,
                                                           noise_x:noise_x+block_noise_size_x,
                                                           noise_y:noise_y+block_noise_size_y]
        cnt -= 1
    return x

class RandomImageOutPainting3D(RandTransform):
    split_idx, p = None, 1

    def __init__(self, p=1):
        super().__init__(p=p)

    def __call__(self, b, split_idx=None, **kwargs):
        "change in __call__ to enforce, that the Transform is always applied on every dataset. "
        return super().__call__(b, split_idx=split_idx, **kwargs)

    def encodes(self, x:(TensorDicom3D)):
        return x.image_out_painting()
    def encodes(self, x:TensorMask3D):
        return x

# Cell
def ssl_transforms_3d(p_all = .5,
                      bezier = True, p_bezier = None,
                      p_inpaint = None,
                      p_outpaint = None,
                      noise = False, p_noise = None,
                      brightness = False, p_brightness = None,
                      contrast = False, p_contrast = None,
                      blur = False, p_blur = None):
    tfms = []
    inpaint = random.choice([True,False])
    outpaint = not inpaint

    if bezier: tfms.append(RandomBezierContrast3D(p=_set_p_tfms(p_bezier, p_all)))
    if inpaint: tfms.append(RandomImageInPainting3D(p=_set_p_tfms(p_inpaint, p_all)))
    if outpaint: tfms.append(RandomImageOutPainting3D(p=_set_p_tfms(p_outpaint, p_all)))
    if noise: tfms.append(RandomNoise3D(p=_set_p_tfms(p_noise, p_all)))
    if brightness: tfms.append(RandomBrightness3D(p=_set_p_tfms(p_brightness, p_all)))
    if contrast: tfms.append(RandomContrast3D(p=_set_p_tfms(p_contrast, p_all)))
    if blur: tfms.append(RandomBlur3D(p=_set_p_tfms(p_blur, p_all)))

    return tfms

# Cell
class TorchIOTransform(RandTransform):
    def __init__(self, tfm_name, p=0.2, **kwargs):
        super().__init__(p=p)
        self.transform = getattr(tio, tfm_name)(**kwargs)

    def encodes(self, x:TensorMask3D):
        return x

    def encodes(self, x:TensorDicom3D):
        assert x.device == torch.device('cpu'), "No cuda support for torchIO transforms"
        meta = x._metadata
        if x.ndim == 3: x = x.unsqueeze(0)
        x = TensorDicom3D(self.transform(x)).squeeze()
        x._metadata = meta
        return x

# Cell
def TioTransforms(p_all=None,
                  tfms = ['RandomBiasField', 'RandomGhosting', 'RandomMotion',
                          'RandomGamma', 'RandomAnisotropy']):
    if not p_all: p_all = 1/len(tfms)
    return [TorchIOTransform(tfm_name, p=p_all) for tfm_name in tfms]

# Cell
class TioRandomBiasField(TorchIOTransform):
    def __init__(self, **kwargs):
        super().__init__(tfm_name = 'RandomBiasField', **kwargs)

class TioRandomGhosting(TorchIOTransform):
    def __init__(self, **kwargs):
        super().__init__(tfm_name = 'RandomGhosting', **kwargs)

class TioRandomMotion(TorchIOTransform):
    def __init__(self, **kwargs):
        super().__init__(tfm_name = 'RandomMotion', **kwargs)

class TioRandomGamma(TorchIOTransform):
    def __init__(self, **kwargs):
        super().__init__(tfm_name = 'RandomGamma', **kwargs)

class TioRandomAnisotropy(TorchIOTransform):
    def __init__(self, **kwargs):
        super().__init__(tfm_name = 'RandomAnisotropy', **kwargs)

class TioRandomElasticDeformation(TorchIOTransform):
    def __init__(self, **kwargs):
        super().__init__(tfm_name = 'RandomElasticDeformation', **kwargs)