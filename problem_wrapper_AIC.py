import numpy as np
from pymoo.core.problem import ElementwiseProblem
from MOO_playground.NSGAII_wrapper import NSGAIIWrap as NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from torch import from_numpy
import os
from copy import deepcopy
from AIC.aic import aic
import pymap_elites_multiobjective.parameters as Params
from pymap_elites_multiobjective.scripts_data.run_env import run_env
from evo_playground.learning.neuralnet import NeuralNetwork as NN
from parameters02 import Parameters as p
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp
import datetime
import time
import multiprocessing


class RoverWrapper(ElementwiseProblem):
    def __init__(self, env):
        self.env = env
        self.st_size = env.state_size()
        self.hid = lp.hid
        self.act_size = env.action_size()
        self.l1_size = self.st_size * self.hid
        self.l2_size = self.hid * self.act_size
        self.model = NN(self.st_size, self.hid, self.act_size)
        self.last_two_0 = [0, 0]
        self.last_two_1 = [0, 0]
        self.n_eval = 0
        self.gen = 0
        super().__init__(n_var=self.l1_size+self.l2_size, n_obj=2, n_ieq_constr=0, xl=-5, xu=5)

    def _evaluate(self, x, out, *args, **kwargs):
        self.env.reset()
        l1_wts = from_numpy(np.reshape(x[:self.l1_size], (self.hid, self.st_size)))
        l2_wts = from_numpy(np.reshape(x[self.l1_size:], (self.act_size, self.hid)))
        self.model.set_weights([l1_wts, l2_wts])
        self.env.reset()
        G = run_env(self.env, [self.model], self.env.params)
        out["F"] = -G
        # print(out['F'])
        self.last_two_0 = self.last_two_1.copy()
        self.last_two_1 = x[:2]
        self.n_eval += 1
        if not self.n_eval % 100:
            self.gen = int(self.n_eval / 100)


def get_unique_fname(rootdir, date_time=None):
    greatest = 0
    # Walk through all the files in the given directory
    for sub, dirs, files in os.walk(rootdir):
        for d in dirs:
            pos = 3
            from re import split
            splitstr = split('_|/', d)
            try:
                int_str = int(splitstr[-pos])
            except (ValueError, IndexError):
                continue

            if int_str > greatest:
                greatest = int_str
        break

    return os.path.join(rootdir, f'{greatest + 1:03d}{date_time}')


def main(run_info):
    par, fpath, stat = run_info
    np.random.seed(stat + np.random.randint(0, 10000))
    print(f'running {fpath}')
    env = aic(par)
    problem = RoverWrapper(env)
    algorithm = NSGA2(fpath, pop_size=100)
    start = time.time()
    res = minimize(problem, algorithm, ('n_gen', 2000))
    tot_time = time.time() - start
    with open(fpath + '_time.txt', 'w') as f:
        f.write(str(tot_time))
    # print(stat, tot_time, '\n', -res.F)


def multiprocess_main(batch_for_multi):
    cpus = multiprocessing.cpu_count() - 1
    # cpus = 2
    with multiprocessing.Pool(processes=cpus) as pool:
        pool.map(main, batch_for_multi)


if __name__ == '__main__':

    now = datetime.datetime.now()
    base_path = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    now_str = now.strftime("_%Y%m%d_%H%M%S")
    dirpath = get_unique_fname(base_path, now_str)
    # dirpath = path.join(getcwd(), now_str)
    os.mkdir(dirpath)

    batch = []
    for param in [Params.p249]:  # , p04]:
        p = deepcopy(param)
        p.n_agents = 1
        lp.n_stat_runs = 10
        for i in range(lp.n_stat_runs):
            filepath = os.path.join(dirpath, f'{p.param_idx:03d}_run{i}')
            os.mkdir(filepath)
            batch.append([p, filepath, i])

    # Use this one
    multiprocess_main(batch)

    # This runs a single experiment / setup at a time for debugging
    # main(batch[0])

    # for b in batch:
    #     main(b)


    # NN weights are stored in res.X
    # print(res.X)
    # print(problem.last_two_0, problem.last_two_1)

    # plot = Scatter()
    # plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    # plot.add(-res.F, facecolor="none", edgecolor="red")
    # plot.show()
