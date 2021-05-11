from kmeans_pytorch import kmeans
from sklearn.metrics import mutual_info_score as mi
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import normalized_mutual_info_score as nmi
from tqdm import tqdm
import torch
from functools import partialmethod
from scorer import x_tensor, one_mask, y_tensor
import numpy as np
import params
from GSA import GSA
import random

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
true_labels = y_tensor.cpu().numpy()


def comp(mask=one_mask):
    cluster_ids_x, cluster_centers = kmeans(
        X=torch.mul(x_tensor, mask), num_clusters=16, distance='euclidean', device=torch.device('cuda:0')
    )
    cluster_ids_x = cluster_ids_x.cpu().numpy()
    return ami(cluster_ids_x, true_labels)


map_alpha_beta = {}
counter = 0
space = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for alpha in space:
    divider = round(alpha / 0.1)
    beta_space = space[:-divider]
    if alpha == 0.0:
        beta_space = space
    for beta in beta_space:
        map_alpha_beta[counter] = (alpha, beta)
        counter += 1

hyper_space = np.ones((55, 11, 11, 11, 20, 20, 3)) * 100000
pop_sizes = [10 * i for i in range(20)]
gen_nums = [50 * i for i in range(20)]
restore_alpha_beta = lambda ab: map_alpha_beta[ab]
wrap_ab = lambda ab: ab if ab >= 0 and ab < 55 else ab % 55
wrap_prob = lambda p: p % 11
wrap_pop = lambda pop: pop % 20
wrap_gen = lambda gen: gen % 20
from numpy.random.mtrand import _rand as global_randstate


def pin_states():
    global_randstate.seed(42)
    np.random.seed(42)
    torch.manual_seed(0)


def calc(point=(0, 0, 0, 0, 0, 0)):
    x1, x2, x3, x4, x5, x6 = point
    params.alpha, params.beta = restore_alpha_beta(x1)
    params.pcross, parmas.pmut, parmas.pswap = space[x2], space[x3], space[x4]
    params.totalsamples = pop_sizes[x5]
    generations = gen_nums[x6]
    if hyper_space[x1][x2][x3][x4][x5][x6][0] < 10000:
        return 0.9 * hyper_space[x1][x2][x3][x4][x5][x6][1] - 0.1 * (hyper_space[x1][x2][x3][x4][x5][x6][2] / 20000)
    pin_states()
    test_GSA = GSA()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(generations):
        test_GSA.generate()
    end.record()
    torch.cuda.synchronize()
    time_used = start.elapsed_time(end)
    hyper_space[x1][x2][x3][x4][x5][x6][2] = time_used
    scores = [comp(test_GSA.population[i]) for i in range(10)]
    hyper_space[x1][x2][x3][x4][x5][x6][1] = sum(scores) / len(scores)
    hyper_space[x1][x2][x3][x4][x5][x6][0] = max(scores)
    return 0.9 * hyper_space[x1][x2][x3][x4][x5][x6][1] - 0.1 * (hyper_space[x1][x2][x3][x4][x5][x6][2] / 20000)


def grad(point=(0, 0, 0, 0, 0, 0)):
    x1, x2, x3, x4, x5, x6 = point
    x5 = wrap_pop(x5)
    x1 = wrap_ab(x1)
    x2 = wrap_prob(x2)
    x3 = wrap_prob(x3)
    x4 = wrap_prob(x4)
    x6 = wrap_gen(x6)
    point_val = calc((x1, x2, x3, x4, x5, x6))
    p_space = [
        (wrap_ab(x1 + 1), x2, x3, x4, x5, x6),
        (x1, wrap_prob(x2 + 1), x3, x4, x5, x6),
        (x1, x2, wrap_prob(x3 + 1), x4, x5, x6),
        (x1, x2, x3, wrap_prob(x4 + 1), x5, x6),
        (x1, x2, x3, x4, wrap_pop(x5 + 1), x6),
        (x1, x2, x3, x4, x5, wrap_gen(x6 + 1))
    ]
    result = [(point, point_val)]
    for adj in p_space:
        result.append((adj, calc(adj)))
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result[0][0]


def rand_initialize():
    return (
        random.randint(0, 54), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10),
        random.randint(0, 20),
        random.randint(0, 20))
