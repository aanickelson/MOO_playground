import numpy as np
from math import sqrt, pi
import heapq as hq
from pymoo.visualization.scatter import Scatter
from matplotlib import pyplot as plt


class SimpleMOO:
    def __init__(self, problem):
        self.problem = problem
        self.n_pop = 100
        self.r = np.random.uniform(0, 10, int(self.n_pop / 2))
        self.h = np.random.uniform(0, 20, int(self.n_pop / 2))
        self.learning_rate = 1

    def run_moo(self, it):
        g = list(map(self.problem, self.r, self.h))
        self.plot_it(g)
        for _ in range(it):
            self.binary_tournament(g)
            self.mutate()
            g = list(map(self.problem, self.r, self.h))
            self.learning_rate /= 1.0001

        self.plot_it(g)

    def plot_it(self, g):
        x_vals = []
        y_vals = []
        xy_vals = [i for i in g if i[0] < 1000]
        for [x1, y1] in xy_vals:
            x_vals.append(x1)
            y_vals.append(y1)
        x = np.array(x_vals)
        y = np.array(y_vals)
        pareto = self.is_pareto_efficient_simple(xy_vals)
        plt.clf()
        plt.scatter(x, y, c='red')
        plt.scatter(x[pareto], y[pareto], c="blue")
        plt.show()

    def binary_tournament(self, scores):
        """
        Run a binary tournament to keep half of the policies
        :param scores:
        :return:
        """
        dummy_ranking = np.random.randint(0, 10000, self.n_pop)

        # create priority queue to randomly match two policies
        pq = []
        for i in range(len(scores)):
            pq.append([dummy_ranking[i], i])
        hq.heapify(pq)

        # Compare two randomly matched policies and keep one
        keep_idx = []
        for j in range(int(len(scores)/2)):
            sc0, idx0 = hq.heappop(pq)
            sc1, idx1 = hq.heappop(pq)
            [g0_0, g0_1] = scores[idx0]
            [g1_0, g1_1] = scores[idx1]
            if g0_0 <= g1_0 and g0_1 <= g1_1:
                keep_idx.append(idx0)
            elif g1_0 <= g0_0 and g1_1 <= g0_1:
                keep_idx.append(idx1)
            else:
                pick_one = np.random.choice([idx0, idx1])
                keep_idx.append(pick_one)

        self.r = [self.r[k] for k in keep_idx]
        self.h = [self.h[k] for k in keep_idx]

    def mutate(self):
        r_del = np.random.uniform(-1, 1, len(self.r)) * self.learning_rate
        h_del = np.random.uniform(-1, 1, len(self.h)) * self.learning_rate
        new_r = self.r + r_del
        new_h = self.h + h_del
        self.r = np.append(self.r, new_r)
        self.h = np.append(self.h, new_h)

    def is_pareto_efficient_simple(self, xyvals):
        """
        Find the pareto-efficient points
        This function copied from here: https://stackoverflow.com/a/40239615
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        costs = np.array(xyvals)
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        return is_efficient


def simple_problem(r, h):
    if r > 10:
        r = 10
    elif r < 0:
        r = 0
    if h > 20:
        h = 20
    elif h < 0:
        h = 0

    s = sqrt(r**2 + h**2)
    B = pi * r**2
    V = B * h / 3.0
    S = pi * r * s
    T = B + S
    if V < 200:
        S = 1000
        T = 1000

    return [S, T]


if __name__ == '__main__':
    moo = SimpleMOO(simple_problem)
    moo.run_moo(5000)
