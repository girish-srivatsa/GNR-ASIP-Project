import numpy as np
import torch
import scipy.io as sio
import os
from numba import prange, njit
import time
import argparse
import matplotlib.pyplot as plt
from numpy.random.mtrand import _rand as global_randstate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score


def pin_states():
    global_randstate.seed(42)
    np.random.seed(42)
    torch.manual_seed(0)


class TorchStandardScaler:

    def fit(self, x):
        self.mean = x.mean(1, keepdim=True)
        self.std = x.std(1, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x


parser = argparse.ArgumentParser(description='GSA Parameters')

#Input paths
parser.add_argument('data_path', type=str, help='path for input data')
parser.add_argument('gt_path', type=str, help='path for ground truth data')
#Params
parser.add_argument('alpha', type=float, help='alpha')
parser.add_argument('beta', type=float, help='beta')
parser.add_argument('compressedbands', type=int, help='compressed bands')
parser.add_argument('totalbands', type=int, help='total bands')
parser.add_argument('totalsamples', type=int, help='total samples')
parser.add_argument('pcross', type=float, help='pcross')
parser.add_argument('pmut', type=float, help='pmut')
parser.add_argument('pswap', type=float, help='pswap')
parser.add_argument('generations', type=int, help='generations')
#Output paths
parser.add_argument('output_dir', type=str, help='output directory')

args = parser.parse_args()


data_path = args.data_path
gt_path = args.gt_path 
output_dir = args.output_dir 

# Load all data files
image_array = sio.loadmat(data_path)['indian_pines_corrected']
ground_truth = sio.loadmat(gt_path)['indian_pines_gt']
# Preprocess into expected forms
flattened_image = np.reshape(image_array, (-1, image_array.shape[-1])).astype(np.int16)
flattened_ground = np.reshape(ground_truth, (-1,)).astype(np.int16)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
x_tensor = torch.Tensor(flattened_image).to(device)
y_tensor = torch.Tensor(flattened_ground).to(device)
# Discard all unlabelled classes(0)
y_idx = y_tensor.nonzero(as_tuple=True)[0].to(device)
y_tensor = y_tensor[y_idx].to(device)
x_tensor = x_tensor[y_idx].to(device)
# Normalize the data
scaler = TorchStandardScaler()
scaler.fit(x_tensor)
x_tensor = scaler.transform(x_tensor).to(device)
ones = torch.ones(size=(x_tensor.shape[0],)).to(device)
y_tensor = torch.sub(y_tensor, ones).to(device)
x_tensor_np = x_tensor.cpu().numpy()
y_tensor_np = y_tensor.cpu().numpy()
x_tensor_train, x_tensor_test, y_tensor_train, y_tensor_test = train_test_split(x_tensor_np, y_tensor_np,test_size=0.1,random_state=42)
x_tensor = torch.from_numpy(x_tensor_train).to(device)
y_tensor = torch.from_numpy(y_tensor_train).to(device)
y_tensor_test = torch.from_numpy(y_tensor_test).to(device)
x_tensor_test = torch.from_numpy(x_tensor_test).to(device)
num_bands = x_tensor.shape[1]
torch.set_num_threads(4)
print("ALL DATA LOADED")


def corrcoef(x):
    # calculate covariance matrix of rows
    c = x.t().mm(x)
    c = c / (x.size(0) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c


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

    res["correlation"] = corrcoef(x_tensor)
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
print("PREPROCESS DONE")
one_mask = torch.ones(size=(num_bands,)).to(device)

# PARAMS


alpha = args.alpha
beta = args.beta
compressedbands = args.compressedbands
totalbands = args.totalbands
totalsamples = args.totalsamples
pcross = args.pcross
pmut = args.pmut
pswap = args.pswap
generations = args.generations


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
    band_corr = torch.sum(torch.abs(masked_correlation)) / (no_bands * (no_bands - 1))  #
    if verbose:
        print(accuracy, mean_distance, band_corr)
    return alpha * accuracy + beta * mean_distance + (1 - alpha - beta) * band_corr


def fast_score(x_tensor=x_tensor, y_tensor=y_tensor, mask=one_mask, num_class=16, verbose=False):
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
    return accuracy


for _ in range(10):
    score()
    fast_score()

true_labels = y_tensor.cpu().numpy()



class GSA:
    def __init__(self, fast=False):
        self.population = [torch.zeros((totalbands,), device=device) for _ in range(totalsamples)]

        for j in range(totalsamples):
            randomiser = np.random.choice(totalbands, size=compressedbands, replace=False)
            for i in range(compressedbands):
                self.population[j][randomiser[i]] = 1
        self.calc = fast_score if fast else score

    def selection(self):
        newpop = [self.population[i] for i in range(totalsamples)]
        self.population = newpop

    def crossover(self):
        """
        BASELINE BENCHMARK - 1 ms for a totsamples = 10
      """
        orig = np.arange(totalsamples)
        orig = np.random.choice(orig, size=int(pcross * totalsamples))
        if len(orig) == 1:
            pass
        else:
            w = len(orig)
            if (w % 2) == 1:
                w -= 1
            for i in range(0, w, 2):
                self.population += self.cross(self.population[orig[i]], self.population[orig[i + 1]])

    def print(self):
        print(self.population)

    def size(self):
        print(len(self.population))

    @staticmethod
    def cross(x, y):
        x_c = x.clone().detach().cpu()
        y_c = y.clone().detach().cpu()
        pos10 = np.where((x_c == 1) & (y_c == 0))[0]
        pos01 = np.where((x_c == 0) & (y_c == 1))[0]
        swap10 = np.random.choice(pos10, size=int(pswap * len(pos10)), replace=False)
        swap01 = np.random.choice(pos01, size=int(pswap * len(pos10)), replace=False)
        for i in swap10:
            x_c[i] = 0
            y_c[i] = 1
        for i in swap01:
            x_c[i] = 1
            y_c[i] = 0
        return [x_c.to(device), y_c.to(device)]

    def mutate(self):
        """
       BASELINE BENCHMARK - 0.3 ms for totalsamples = 10
      """
        l = len(self.population)
        for i in range(l):
            toss = np.random.binomial(1, pmut)
            if toss:
                modpop = self.population[i].clone().detach().cpu()
                pos1 = np.where(modpop == 1)[0]
                oneto0 = np.random.choice(pos1, size=1)
                pos0 = np.where(modpop == 0)[0]
                zeroto1 = np.random.choice(pos0, size=1)
                modpop[oneto0] = 0
                modpop[zeroto1] = 1
                self.population += [modpop.to(device)]

    def fit(self):
        """
        BASELINE BENCHMARK - 6 ms per individual
      """
        fits = {p: self.calc(mask=p) for p in self.population}
        self.population = sorted(self.population, key=lambda x: fits[x], reverse=True)

    def generate(self):
        """
        BASELINE BENCHMARK - 7.8 s for 100 generations with totsamples = 10
                           - 0.62 s for 100 generations with totsamples = 10 on CUDA backend
                           - 18 s for 100 generations with totsamples = 20
                           - 1.4 s for 100 generations with totsamples = 20 on CUDA backend
      """
        self.crossover()
        self.mutate()
        self.fit()
        self.selection()

    @staticmethod
    def benchmark():
        pop_sizes = [10 * i for i in range(1, 11)]
        gen_nums = [100 * i for i in range(1, 11)]
        time_benchmark_pop = []
        time_benchmark_gen = []
        score_benchmark_pop = []
        score_benchmark_gen = []
        if torch.cuda.is_available():
            for gen_num in gen_nums:
                print(f"RUNNING FOR number of generations = {gen_num}")
                test_GSA = GSA()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(gen_num):
                    test_GSA.generate()
                end.record()
                torch.cuda.synchronize()
                time_benchmark_gen.append(start.elapsed_time(end))
                score_benchmark_gen.append(score(mask=test_GSA.population[0]))
            for pop_size in pop_sizes:
                print(f"RUNNING FOR population = {pop_size}")
                test_GSA = GSA()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    test_GSA.generate()
                end.record()
                torch.cuda.synchronize()
                time_benchmark_pop.append(start.elapsed_time(end))
                score_benchmark_pop.append(score(mask=test_GSA.population[0]))
        else:
            for gen_num in gen_nums:
                test_GSA = GSA()
                start = time.time()
                for _ in range(gen_num):
                    test_GSA.generate()
                end = time.time()
                time_benchmark_gen.append((end - start) * 1000)
                score_benchmark_gen.append(score(mask=test_GSA.population[0]))
            for pop_size in pop_sizes:
                test_GSA = GSA()
                start = time.time()
                for _ in range(100):
                    test_GSA.generate()
                end = time.time
                time_benchmark_pop.append((end - start) * 1000)
                score_benchmark_pop.append(score(mask=test_GSA.population[0]))

        benchmarks = {
            "time_benchmark_pop": np.array(time_benchmark_pop),
            "time_benchmark_gen": np.array(time_benchmark_gen),
            "score_benchmark_pop": np.array(score_benchmark_pop),
            "score_benchmark_gen": np.array(score_benchmark_gen)
        }

        for name, val in benchmarks.items():
            with open(name + ".npy", "wb") as f:
                print(f"SAVING {name}")
                np.save(f, val)

        open("time-population.png", 'w').close()
        open("score-population.png", 'w').close()
        open("time-generations.png", 'w').close()
        open("score-generations.png", 'w').close()

        fig = plt.figure(1, figsize=(6, 6))
        plt.plot(pop_sizes, time_benchmark_pop, marker="o", color="blue")
        plt.savefig("time-population.png")
        plt.close(fig)

        fig = plt.figure(2, figsize=(6, 6))
        plt.plot(pop_sizes, score_benchmark_pop, marker="o", color="blue")
        plt.savefig("score-population.png")
        plt.close(fig)

        fig = plt.figure(3, figsize=(6, 6))
        plt.plot(gen_nums, time_benchmark_gen, marker="o", color="blue")
        plt.savefig("time-generations.png")
        plt.close(fig)

        fig = plt.figure(4, figsize=(6, 6))
        plt.plot(gen_nums, score_benchmark_gen, marker="o", color="blue")
        plt.savefig("score-generations.png")
        plt.close(fig)

        return benchmarks

    @staticmethod
    def test(func="mutate"):
        import time
        test_GSA = GSA()
        if func == "mutate":
            counter = 0
            for _ in range(1000):
                test_GSA.selection()
                s1 = time.time()
                test_GSA.mutate()
                e1 = time.time()
                counter += (e1 - s1)
            print(counter)
        elif func == "fit":
            counter = 0
            for _ in range(100):
                test_GSA = GSA()
                s1 = time.time()
                test_GSA.fit()
                e1 = time.time()
                counter += (e1 - s1)
            print(counter * 10)
        elif func == "init":
            counter = 0
            for _ in range(1000):
                s1 = time.time()
                test_GSA = GSA()
                e1 = time.time()
                counter += (e1 - s1)
            print(counter)
        elif func == "cross":
            counter = 0
            for _ in range(100):
                test_GSA.selection()
                s1 = time.time()
                test_GSA.crossover()
                e1 = time.time()
                counter += (e1 - s1)
            print(counter * 10)
        elif "generation":
            counter = 0
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                for _ in range(100):
                    test_GSA.generate()
                end.record()
                torch.cuda.synchronize()
                print(start.elapsed_time(end))
                # print(prof.key_averages().table(sort_by="cuda_time_total"))
            else:
                for _ in range(100):
                    s1 = time.time()
                    test_GSA.generate()
                    e1 = time.time()
                    counter += (e1 - s1)
                print(counter * 10)
        else:
            counter = 0
            for _ in range(1000):
                test_GSA.mutate()
                s1 = time.time()
                test_GSA.selection()
                e1 = time.time()
                counter += (e1 - s1)
            print(counter)


pin_states()
gsa = GSA(fast=True)
print("GSA GENERATIONS STARTING")
for _ in range(generations):
    gsa.generate()
print("GENERATING STATS AND PLOTS")
num_class = 16
winner = gsa.population[0]
bands = torch.nonzero(winner, as_tuple=True)[0]
compressed_x = flattened_image[:, bands.cpu().numpy()]
compressed_image = compressed_x.reshape((145, 145, -1))
np.save(output_dir + '/compressed_indianpines.npy', compressed_image)

x_tensor_test_masked = torch.mul(x_tensor_test, winner)
mask_mean = winner.broadcast_to((num_class, num_bands))
mean_stack_masked = torch.mul(res["mean_stack"], mask_mean)
mean_stack_masked = mean_stack_masked.t()
mean_stack_one = res["mean_stack"]
mean_stack_one = mean_stack_one.t()
dots = torch.mm(x_tensor_test_masked, mean_stack_masked)
dots_one = torch.mm(x_tensor_test, mean_stack_one)
norm_x = torch.sqrt(torch.sum(x_tensor_test_masked.pow_(2), dim=1)).view(-1, 1)
norm_x_one = torch.sqrt(torch.sum(x_tensor_test.pow_(2), dim=1)).view(-1, 1)
norm_mean = torch.sqrt(torch.sum(mean_stack_masked.pow_(2), dim=0)).view(1, -1)
norm_mean_one = torch.sqrt(torch.sum(mean_stack_one.pow_(2), dim=0)).view(1, -1)
norm_product = torch.mm(norm_x, norm_mean)
norm_product_one = torch.mm(norm_x_one, norm_mean_one)
cosines = torch.div(dots, norm_product)
cosines_one = torch.div(dots_one, norm_product_one)
result_comp = torch.argmax(cosines, dim=1) if torch.cuda.is_available() else torch.from_numpy(
    indexFunc4(cosines.numpy()))
result_orig = torch.argmax(cosines_one, dim=1) if torch.cuda.is_available() else torch.from_numpy(
    indexFunc4(cosines_one.numpy()))


np.save(output_dir + '/compressed_results.npy', result_comp.cpu().numpy())
np.save(output_dir + '/original_results.npy', result_orig.cpu().numpy())
np.save(output_dir + '/true_labels.npy', y_tensor_test.cpu().numpy())
map_class = {
    0: "Alfalfa",
    1: "Corn-notill",
    2: "Corn-mintill",
    3: "Corn",
    4: "Grass-pasture",
    5: "Grass-trees",
    6: "Grass-pasture-mowed",
    7: "Hay-windrowed",
    8: "Oats",
    9: "Soybean-notill",
    10: "Soybean-mintill",
    11: "Soybean-clean",
    12: "Wheat",
    13: "Woods",
    14: "Buildings-Grass-Trees-Drives",
    15: "Stone-Steel-Towers"
}
classes = [map_class[i] for i in range(16)]
y_comp = result_comp.cpu().numpy().tolist()
y_full = result_orig.cpu().numpy().tolist()
y_true = y_tensor_test.cpu().numpy().tolist()
y_comp = [map_class[x] for x in y_comp]
y_full = [map_class[x] for x in y_full]
y_true = [map_class[x] for x in y_true]
print("KAPPA SCORES:")
print("FULL BAND CASE :",cohen_kappa_score(y_full, y_true))
print("COMPRESSED BAND CASE :",cohen_kappa_score(y_comp, y_true))
matrix_full = confusion_matrix(y_true, y_full, labels=classes)
matrix_comp = confusion_matrix(y_true, y_comp, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix_full, display_labels=classes)
disp.plot()
disp.figure_.savefig(output_dir + "/full_band_confusion.png")
disp = ConfusionMatrixDisplay(confusion_matrix=matrix_comp, display_labels=classes)
disp.plot()
disp.figure_.savefig(output_dir + "/compressed_band_confusion.png")
