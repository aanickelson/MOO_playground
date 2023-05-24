"""
copied from PyMoo Genetic specifically to be able to save data. Is dumb
"""


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
import numpy as np

class NSGAIIWrap(NSGA2):

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.gen = 0
        self.fpath = filepath

    def _advance(self, infills=None, **kwargs):
        # the current population
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs)

        if not self.gen % 100 or self.gen==1999:
            fits = np.array([-p.F for p in self.pop])
            np.savetxt(self.fpath + f'/fits{self.gen}.npy', fits)

        self.gen += 1