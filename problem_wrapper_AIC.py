import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from torch import from_numpy

from AIC.aic import aic
from pymap_elites_multiobjective.parameters import p010, p349
from pymap_elites_multiobjective.scripts_data.run_env import run_env
from evo_playground.learning.neuralnet import NeuralNetwork as NN
from parameters02 import Parameters as p
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp


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
        self.gen += 1
        if not self.gen % 1000:
            print(self.gen / 100)
            print(G)


if __name__ == '__main__':
    p_num = p010
    env = aic(p_num)
    problem = RoverWrapper(env)
    algorithm = NSGA2(pop_size=100)
    res = minimize(problem, algorithm, ('n_gen', 2000))
    print(-res.F)
    # NN weights are stored in res.X
    # print(res.X)
    # print(problem.last_two_0, problem.last_two_1)

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(-res.F, facecolor="none", edgecolor="red")
    plot.show()
