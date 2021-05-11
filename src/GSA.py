import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import params
from scorer import score
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class GSA:

    def __init__(self):
        self.population = [torch.zeros((params.totalbands,), device=device) for _ in range(params.totalsamples)]

        for j in range(params.totalsamples):
            randomiser = np.random.choice(params.totalbands, size=params.compressedbands, replace=False)
            for i in range(params.compressedbands):
                self.population[j][randomiser[i]] = 1

    def selection(self):
        new_pop = [self.population[i] for i in range(params.totalsamples)]
        self.population = new_pop

    def crossover(self):
        """
          BASELINE BENCHMARK - 1 ms for a totsamples = 10
        """
        orig = np.arange(params.totalsamples)
        orig = np.random.choice(orig, size=int(params.pcross * params.totalsamples))
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
        swap10 = np.random.choice(pos10, size=int(params.pswap * len(pos10)), replace=False)
        swap01 = np.random.choice(pos01, size=int(params.pswap * len(pos10)), replace=False)
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
            toss = np.random.binomial(1, params.pmut)
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
        self.scores = {p: score(mask=p,alpha=params.alpha,beta=params.beta) for p in self.population}
        self.population = sorted(self.population, key=lambda x: self.scores[x], reverse=True)

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
        params.totalsamples = 10
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
                params.totalsamples = pop_size
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
                params.totalsamples = pop_size
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
