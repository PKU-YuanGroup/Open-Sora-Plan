
import numpy as np
import scipy
from torch.utils.data import DataLoader, TensorDataset
import torch
import hashlib
import os
import glob
import requests
import re
import html
import io
import uuid

_feature_detector_cache = dict()

# this is a helper function that allows to download a file from the internet cache it and open it as if it was a normal file
def open_url(url, num_attempts=10, verbose=False, cache_dir=None):
    assert num_attempts >=1

    if cache_dir is None:
        cache_dir = './loaded_models'
    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
    if len(cache_files) == 1:
        f_name = cache_files[0]
        return open(f_name, 'rb')
    
    with requests.Session() as session:
        if verbose:
            print("Downloading ", url, flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")
                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise Exception("Interupted")
            except:
                if not attempts_left:
                    if verbose:
                        print("failed!")
                    raise
                if verbose:
                    print('.')
        
    safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
    cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
    temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
    os.makedirs(cache_dir, exist_ok=True)
    with open(temp_file, 'wb') as f:
        f.write(url_data)
    os.replace(temp_file, cache_file)

    return io.BytesIO(url_data)

# load the feature extractor either from cache or the specified URL
def get_feature_detector(detector_url, device):
    key = (detector_url, device)
    if key not in _feature_detector_cache:
        with open_url(detector_url, verbose=True) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
    return _feature_detector_cache[key]


"""
This function is used to first extract feature representation vectors of the videos using a pretrained model
Then the mean and covariance of the representation vectors are calculated and returned
"""
def compute_feature_stats(data, detector_url, detector_kwargs, batch_size, max_items, device):
    # if wanted reduce the number of elements used for calculating the FVD
    num_items = len(data)
    if max_items:
        num_items = min(num_items, max_items)
    data = data[:num_items]
    
    # load the pretrained feature extraction modeö
    detector = get_feature_detector(detector_url, device=device)

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size)
    all_features = []
    for batch in loader:
        batch = batch[0]
        # if more than 3 channels are available we split the channel dimension into chunks of 3 and concatenate to batch dimension
        if batch.size(1) != 3:
            pad_size = 3 - (batch.size(1) % 3)
            pad = torch.zeros(batch.size(0), pad_size, batch.size(2), batch.size(3), batch.size(4), device=batch.device)
            batch = torch.cat([batch, pad], dim=1)
            batch = torch.cat(torch.chunk(batch, chunks=batch.size(1)//3, dim=1), dim=0)
        batch = batch.to(device)
        # extract feature vector using pretrained model
        features = detector(batch, **detector_kwargs)
        features = features.detach().cpu().numpy()
        all_features.append(features)
    # concatenate batches to one numpy array
    stacked_features = np.concatenate(all_features, axis=0)

    # calculate mean and covariance matrix across the extracted features
    mu = np.mean(stacked_features, axis=0)
    sigma = np.cov(stacked_features, rowvar=False)

    return mu, sigma

def calculate_fvd(y_true: torch.Tensor, y_pred: torch.Tensor, device: torch.device):
    '''
        y_true: (bz,c,t,h,w) `num_videos x channels x num_frames x width x height`
        y_pred: (bz,c,t,h,w) `num_videos x channels x num_frames x width x height`
    '''
    # print(y_true.shape) # torch.Size([5, 20, 3, 64, 64])
    # print(y_pred.shape) # torch.Size([5, 20, 3, 64, 64])
    y_true = torch.permute(y_true,(0,2,1,3,4)).contiguous()
    y_pred = torch.permute(y_pred,(0,2,1,3,4)).contiguous()

    batch_size = y_true.shape[0]
    max_items = batch_size
    print("calculate_fvd...")
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=True, resize=True, return_features=True) # Return raw features before the softmax layer.

    # calculate the mean and covariance matrix of the representation vectors for ground truth and predicted videos
    mu_true, sigma_true = compute_feature_stats(y_true, detector_url, detector_kwargs, batch_size, max_items, device)
    mu_pred, sigma_pred = compute_feature_stats(y_pred, detector_url, detector_kwargs, batch_size, max_items, device)
    m = np.square(mu_pred - mu_true).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_pred, sigma_true), disp=False)
    fvd = np.real(m + np.trace(sigma_pred + sigma_true - s * 2))

    return float(fvd)
