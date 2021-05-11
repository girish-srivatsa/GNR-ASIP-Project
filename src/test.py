import scorer
import params
from GSA import GSA
from kmeans_pytorch import kmeans
from sklearn.metrics import adjusted_mutual_info_score as ami
import torch
from numpy.random.mtrand import _rand as global_randstate
import numpy as np


def pin_states():
    global_randstate.seed(42)
    np.random.seed(42)
    torch.manual_seed(0)

print("IMPORTS DONE")


true_labels = scorer.y_tensor.cpu().numpy()

def comp(mask=scorer.one_mask):
    cluster_ids_x, cluster_centers = kmeans(
        X=torch.mul(scorer.x_tensor, mask), num_clusters=16, distance='euclidean', device=torch.device('cuda:0')
    )
    cluster_ids_x = cluster_ids_x.cpu().numpy()
    return ami(cluster_ids_x, true_labels)


#pos_list = [(0.5, 0.2, 0.1, 0.8, 0.3, 110, 350),(0.2, 0.8, 0.4, 0.9, 0.1, 60, 700),(0.1, 0.4, 0.7, 1.0, 0.3, 150, 600),(0.2, 0.2, 1.0, 0.9, 0.4, 110, 350),(0.3, 0.4, 0.3, 0.5, 0.1, 160, 800),(0.2, 0.1, 0.9, 0.6, 0.2, 120, 250),(0.2, 0.6, 0.6, 0.2, 0.3, 90, 350),(0.4, 0.2, 0.3, 1.0, 0.8, 10, 200),(0.5, 0.2, 0.9, 0.7, 0.1, 190, 500),(0.5, 0.1, 0.7, 0.8, 0.3, 60, 200),(0.0, 0.6, 0.5, 0.5, 0.5, 140, 750),(0.2, 0.7, 0.7, 0.5, 0.9, 190, 100)]
#pos_list = [(0.4, 0.2, 0.3, 1.0, 0.8, 10, 200),(0.5, 0.1, 0.7, 0.8, 0.3, 60, 200),(0.2, 0.7, 0.7, 0.5, 0.9, 190, 100),(0.2, 0.1, 0.9, 0.6, 0.2, 120, 250),(0.2, 0.6, 0.6, 0.2, 0.3, 90, 350),(0.2, 0.2, 1.0, 0.9, 0.4, 110, 350),(0.1, 0.4, 0.7, 1.0, 0.3, 150, 600),(0.5, 0.2, 0.9, 0.7, 0.1, 190, 500),(0.3, 0.4, 0.3, 0.5, 0.1, 160, 800)]
pos_list = [(0.4, 0.2, 0.3, 1.0, 0.8, 10, 200),(0.2, 0.7, 0.7, 0.5, 0.9, 190, 100),(0.2, 0.1, 0.9, 0.6, 0.2, 120, 250),(0.2, 0.6, 0.6, 0.2, 0.3, 90, 350),(0.5, 0.2, 0.1, 0.8, 0.3, 110, 350),(0.2, 0.2, 1.0, 0.9, 0.4, 110, 350),(0.2, 0.8, 0.4, 0.9, 0.1, 60, 700),(0.1, 0.4, 0.7, 1.0, 0.3, 150, 600),(0.5, 0.2, 0.9, 0.7, 0.1, 190, 500),(0.3, 0.4, 0.3, 0.5, 0.1, 160, 800)]
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
print(counter)
print(len(map_alpha_beta))
print("ITERATIONS STARTING")
for pos in range(66):
    print("#" * 20)
    print(pos)
    params.alpha,params.beta = map_alpha_beta[pos]
    params.pcross = 0.7
    params.pmut = 0.8
    params.pswap = 0.3
    params.totalsamples = 60
    generations = 200
    pin_states()
    gsa=GSA()
    for _ in range(generations):
        gsa.generate()
    scores = [comp(gsa.population[i]) for i in range(10)]
    print(max(scores),sum(scores)/len(scores))
    print("#"*20)
