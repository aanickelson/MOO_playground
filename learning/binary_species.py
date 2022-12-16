
# Python packages
import numpy as np
import heapq as hq
import torch
from os import getcwd, path

# Custom packages
from evo_playground.learning.neuralnet import NeuralNetwork as NN
from teaming.domain import DiscreteRoverDomain as Domain
import evo_playground.parameters as param


class Species:
    def __init__(self, env, p):
        self.n_pol = p.n_policies
        self.p = p
        self.env = env
        self.sigma = p.sigma
        self.learning_rate = 1
        a = self.env.state_size()
        b = self.p.hid
        c = self.env.get_action_size()
        self.model = NN(self.env.state_size(), self.p.hid, self.env.get_action_size())
        self.weights = self._species_setup()

    def _species_setup(self):
        # a set of randomly initilaized policies
        species = [NN(self.env.state_size(), self.p.hid, self.env.get_action_size()).get_weights()
                   for _ in range(int(self.n_pol / 2))]
        return species

    def mutate_weights(self):
        # Mutate each set of weights that was kept
        new_weights = []
        for wts in self.weights:
            noise = self.sigma * torch.normal(0, 1, size=wts[0].shape) * self.learning_rate
            noise2 = self.sigma * torch.normal(0, 1, size=wts[1].shape) * self.learning_rate
            new_weights.append([wts[0] + noise, wts[1] + noise2])
        # Have to do this separately otherwise it creates an infinite loop (whoopsies)
        for wts_01 in new_weights:
            self.weights.append(wts_01)

        # Reduce the learning rate slightly at each mutation
        self.learning_rate *= 0.999

    def binary_tournament(self, scores):
        """
        Run a binary tournament to keep half of the policies
        :param scores:
        :return:
        """
        dummy_ranking = np.random.randint(0, 10000, len(scores))

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
            if scores[idx0] >= scores[idx1]:
                keep_idx.append(idx0)
            else:
                keep_idx.append(idx1)

        self.weights = [self.weights[k] for k in keep_idx]

    def save_model(self, trial, stat, gen, prepend, wts, species=''):

        pth = path.join(getcwd(), 'weights', 't{:03d}_{}_{}weights_s{}_g{}.pth'.format(trial, stat, prepend, species, gen))
        torch.save(wts, pth)



if __name__ == '__main__':
    p = param.p318
    env = Domain(p)
    spec = Species(env, p)
    spec.mutate_weights()
    dummy_scores = np.random.randint(0, 100, 50)
    spec.binary_tournament(dummy_scores)
    print("Nothing")