import numpy as np
from pymoo.core.problem import ElementwiseProblem
from teaming.domain import DiscreteRoverDomain
from parameters00 import Parameters as p
from MOO_playground.learning.neuralnet import NeuralNetwork as NN
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from torch import from_numpy


class RoverWrapper(ElementwiseProblem):
    def __init__(self, env):
        self.env = env
        self.st_size = env.state_size()
        self.hid = env.p.hid
        self.act_size = env.get_action_size()
        self.l1_size = self.st_size * self.hid
        self.l2_size = self.hid * self.act_size
        self.model = NN(self.env.state_size(), self.env.p.hid, self.env.get_action_size())
        super().__init__(n_var=self.l1_size + self.l2_size, n_obj=2, n_ieq_constr=0, xl=0, xu=1)


    def _evaluate(self, x, out, *args, **kwargs):
        l1_wts = from_numpy(np.reshape(x[:self.l1_size], (self.hid, self.st_size)))
        l2_wts = from_numpy(np.reshape(x[self.l1_size:], (self.act_size, self.hid)))
        self.model.set_weights([l1_wts, l2_wts])
        self.env.run_sim([self.model])
        out["F"] = -self.env.multiG()


if __name__ == '__main__':
    env = DiscreteRoverDomain(p)
    problem = RoverWrapper(env)
    l1_size = env.state_size() * p.hid
    l2_size = p.hid * env.get_action_size()

    wts = NN(env.state_size(), p.hid, env.get_action_size()).get_weights()
    wts_npy = [wt.detach().numpy() for wt in wts]
    algorithm = NSGA2(pop_size=200)
    res = minimize(problem, algorithm, ('n_gen', 200))
    print(res.F)
    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(-res.F, facecolor="none", edgecolor="red")
    plot.show()
