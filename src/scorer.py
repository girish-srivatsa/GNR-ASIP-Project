import numpy as np
import torch
import scipy.io as sio
import os
from numba import prange, njit


class TorchStandardScaler:

    def fit(self, x):
        self.mean = x.mean(1, keepdim=True)
        self.std = x.std(1, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x


# Load all paths
curr_path = os.getcwd()
root_path = curr_path[:-3]
data_path = root_path + 'data/Indian_pines_corrected.mat'
gt_path = root_path + 'data//Indian_pines_gt.mat'
# Load all data files
image_array = sio.loadmat(data_path)['indian_pines_corrected']
ground_truth = sio.loadmat(gt_path)['indian_pines_gt']
# Preprocess into expected forms
flattened_image = np.reshape(image_array, (-1, image_array.shape[-1])).astype(np.int16)
flattened_ground = np.reshape(ground_truth, (-1,)).astype(np.int16)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
x_tensor = torch.Tensor(flattened_image).to(device)
y_tensor = torch.Tensor(flattened_ground).to(device)
y_idx = y_tensor.nonzero(as_tuple=True)[0].to(device)
y_tensor = y_tensor[y_idx].to(device)
x_tensor = x_tensor[y_idx].to(device)
scaler = TorchStandardScaler()
scaler.fit(x_tensor)
x_tensor = scaler.transform(x_tensor).to(device)
ones = torch.ones(size=(x_tensor.shape[0],)).to(device)
y_tensor = torch.sub(y_tensor,ones).to(device)
num_bands = x_tensor.shape[1]
x_tensor_slice = [x_tensor[:, i] for i in range(num_bands)]
torch.set_num_threads(4)


# Preprocessing to improve speed
def preprocess(x_tensor=x_tensor, y_tensor=y_tensor, num_class=16):
    classes = [x_tensor[(y_tensor == i).nonzero(as_tuple=True)[0], :] for i in range(0, num_class)]
    res = {"mean": {
        k: torch.mean(classes[k], dim=0) for k in range(0, num_class)
    }}

    res["mean_stack"] = torch.stack(tuple([v for k, v in res["mean"].items()]))
    res["band_mean"] = torch.mean(x_tensor, dim=0).view(1, -1)

    res["band_mean_product"] = torch.mm(torch.t(res["band_mean"]), res["band_mean"])

    res["covariance"] = torch.Tensor(np.cov(torch.t(x_tensor).cpu())).to(device)

    res["mean_diffs"] = (res["mean_stack"].unsqueeze(1) - res["mean_stack"])

    res["correlation"] = torch.add(res["covariance"], res["band_mean_product"])
    # REMOVE COMMON CORRELATIONS
    res["correlation"] = res["correlation"].fill_diagonal_(0)

    return res


@njit(cache=True, parallel=True)
def indexFunc4(x):
    max_indices = np.zeros((x.shape[0],), dtype=np.float32)
    for i in prange(x.shape[0]):
        max_indices[i] = np.argmax(x[i])
    return max_indices


# Code warmup
if not torch.cuda.is_available():
    test = np.arange(100).reshape((10, 10))
    for _ in range(10):
        indexFunc4(test)

res = preprocess()
one_mask = torch.ones(size=(num_bands,)).to(device)
alpha = 1
beta = 0


def score(x_tensor=x_tensor, y_tensor=y_tensor, mask=one_mask, num_class=16, alpha=alpha, beta=beta, verbose=False):
    # SAM CLASSIFICATION
    x_tensor_masked = torch.mul(x_tensor, mask)  # MUL1
    mask_mean = mask.broadcast_to((num_class, num_bands))
    mean_stack_masked = torch.mul(res["mean_stack"], mask_mean)  # MUL2
    mean_stack_masked = mean_stack_masked.t()

    dots = torch.mm(x_tensor_masked, mean_stack_masked)  # MM1

    norm_x = torch.sqrt(torch.sum(x_tensor_masked.pow_(2), dim=1)).view(-1, 1)  # SUM1(10000*2000)

    norm_mean = torch.sqrt(torch.sum(mean_stack_masked.pow_(2), dim=0)).view(1, -1)  # SUM2(16*200)

    norm_product = torch.mm(norm_x, norm_mean)  # MM2

    cosines = torch.div(dots, norm_product)  # DIV1

    result = torch.argmax(cosines, dim=1) if torch.cuda.is_available() else torch.from_numpy(
        indexFunc4(cosines.numpy()))
    # ACCURACY MEASUREMENT
    accuracy = torch.sum(result == y_tensor, dtype=torch.int16) / x_tensor.shape[0]  # SUM3(10000)
    # DISTANCE BETWEEN MEANS CALCULATION
    mask_diffs = mask.broadcast_to((num_class, num_class, num_bands))
    masked_mean_diffs = torch.mul(res["mean_diffs"], mask_diffs)  # MUL3

    masked_mean_distances = torch._C._VariableFunctions.frobenius_norm(masked_mean_diffs, 2, False)
    mean_distance = torch.sum(masked_mean_distances) / (num_class * (num_class - 1))  # SUM4(256)
    # BAND CORRELATION CALCULATION
    mask_2d = torch.mm(mask.view(-1, 1), mask.view(1, -1))  # MM3
    no_bands = torch.sum(mask, dtype=torch.int32)  # SUM5(200)

    masked_correlation = torch.mul(res["correlation"], mask_2d)  # MUL4
    band_corr = torch.sum(masked_correlation) / (no_bands * (no_bands - 1))  #
    if verbose:
        print(accuracy,mean_distance,band_corr)
    return alpha * accuracy + beta * mean_distance + (1 - alpha - beta) * band_corr


for _ in range(10):
    score()
