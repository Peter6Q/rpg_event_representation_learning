import numpy as np
import torch
from .torch_percentile import percentile

# Obtained and modified from https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting/11886564#11886564
def outlier1d(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def phase_correlation(a, b):
    B, H, W = a.size()
    a = a.unsqueeze(dim=-1).expand(B, H, W, 2)
    b = b.unsqueeze(dim=-1).expand(B, H, W, 2)
    G_a = torch.fft(a, signal_ndim=2)
    G_b = torch.fft(b, signal_ndim=2)
    conj_b = torch.conj(G_b)
    R = G_a * conj_b
    R /= torch.abs(R)
    r = torch.ifft(R, signal_ndim=2)
    r = torch.split(r, 1, dim=-1)[0].squeeze(-1)
    shift = r.view(B, -1).argmax(dim=1)
    shift = torch.cat(((shift / W).view(-1, 1), (shift % W).view(-1, 1)), dim=1)
    return shift

def fftshift(image):
    # Original with size (B, H, W, 2)
    image = image.permute(3,0,1,2)
    real, imag = image[0], image[1]
    for dim in range(1, len(real.size())):
        real = torch.fft.roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = torch.fft.roll_n(imag, axis=dim, n=imag.size(dim)//2)
    image = torch.stack([real, imag], dim=0).permute(1,2,3,0)
    return image

def ifftshift(image):
    # Original with size (B, H, W, 2)
    image = image.permute(3,0,1,2)
    real, imag = image[0], image[1]
    for dim in range(len(real.size()) - 1, 0, -1):
        real = torch.fft.roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = torch.fft.roll_n(imag, axis=dim, n=imag.size(dim)//2)
    image = torch.stack([real, imag], dim=0).permute(1,2,3,0)
    return image