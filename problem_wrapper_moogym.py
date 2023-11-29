import numpy as np
from pymoo.core.problem import ElementwiseProblem
from MOO_playground.NSGAII_wrapper import NSGAIIWrap as NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from torch import from_numpy
import os
from copy import deepcopy
import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics


import pymap_elites_multiobjective.parameters as Params
from evo_playground.support.neuralnet import NeuralNetwork as NN
from evo_playground.test_morl.sar_wrapper import SARWrap
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp
import datetime
import time
import multiprocessing


class MOOGymWrapper(ElementwiseProblem):
    def __init__(self, env):
        self.env = env
        self.st_size = env.state_size()
        self.hid = lp.hid
        self.act_size = env.action_size()
        self.l1_size = self.st_size * self.hid
        self.l2_size = self.hid * self.act_size
        self.model = NN(self.st_size, self.hid, self.act_size)
        self.n_eval = 0
        self.gen = 0
        super().__init__(n_var=self.l1_size+self.l2_size, n_obj=self.env.n_obj, n_ieq_constr=0, xl=-5, xu=5)

    def _evaluate(self, x, out, *args, **kwargs):
        self.env.reset()
        l1_wts = from_numpy(np.reshape(x[:self.l1_size], (self.hid, self.st_size)))
        l2_wts = from_numpy(np.reshape(x[self.l1_size:], (self.act_size, self.hid)))
        self.model.set_weights([l1_wts, l2_wts])
        self.env.reset()
        G = self.env.run(self.model)
        out["F"] = -G
        self.n_eval += 1
        if not self.n_eval % 1000:
            self.gen = int(self.n_eval / 100)
            print(self.gen)


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
    fpath, gym_env, stat = run_info
    np.random.seed(stat + np.random.randint(0, 10000))
    print(f'running {fpath}')
    moo_gym_env = MORecordEpisodeStatistics(mo_gym.make(gym_env), gamma=0.99)
    # eval_env = mo_gym.make(gym_env)
    wrap = SARWrap(moo_gym_env)
    problem = MOOGymWrapper(wrap)
    algorithm = NSGA2(fpath, pop_size=100)
    res = minimize(problem, algorithm, ('n_gen', 1000))

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(-res.F, facecolor="none", edgecolor="red")
    plot.show()


def multiprocess_main(batch_for_multi):
    cpus = multiprocessing.cpu_count() - 1
    # cpus = 2
    with multiprocessing.Pool(processes=cpus) as pool:
        pool.map(main, batch_for_multi)


if __name__ == '__main__':

    env_name = "mo-lunar-lander-continuous-v2"
    env_shorthand = 'lander'
    now = datetime.datetime.now()
    base_path = os.path.join(os.getcwd(), 'data', env_shorthand)
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    now_str = now.strftime("_%Y%m%d_%H%M%S")
    dirpath = get_unique_fname(base_path, now_str)
    # dirpath = path.join(getcwd(), now_str)
    os.mkdir(dirpath)

    batch = []
    lp.n_stat_runs = 10
    problem_nm = 'lander'
    for i in range(lp.n_stat_runs):
        filepath = os.path.join(dirpath, f'{problem_nm}_run{i}')
        os.mkdir(filepath)
        batch.append([filepath, env_name, i])

    # Use this one
    # multiprocess_main(batch)

    # This runs a single experiment / setup at a time for debugging
    main(batch[0])

    # for b in batch:
    #     main(b)


    # NN weights are stored in res.X
    # print(res.X)
    # print(problem.last_two_0, problem.last_two_1)


