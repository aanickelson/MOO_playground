"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies

Structure and main functions for basic
"""


from evo_playground.support.neuralnet import NeuralNetwork
import torch
from os import path, getcwd
import numpy as np


class EvolveNN:
    def __init__(self, env, p):
        self.env = env
        self.sigma = p.sigma
        self.n_policies = p.n_policies
        self.learning_rate = p.learning_rate
        self.model = NeuralNetwork(env.state_size(), p.hid, env.get_action_size())
        self.start_weights = self.model.get_weights()

    def score_genome(self, weights):
        self.model.set_weights(weights)
        self.env.new_env()
        G, avg_false = self.env.run_sim([self.model])
        return G, avg_false

    def mutate_weights(self, weights):
        weights_to_try = []
        for _ in range(self.n_policies):
            noise = self.sigma * torch.normal(0, 1, size=weights[0].shape)
            noise2 = self.sigma * torch.normal(0, 1, size=weights[1].shape)
            weights_to_try.append([weights[0] + noise, weights[1] + noise2])
        return weights_to_try

    def update_weights(self, start_weights, weights, scores):
        if scores.std() == 0:
            return start_weights
        scores = (scores - scores.mean()) / scores.std()
        new_weights = []
        for index, w in enumerate(start_weights):
            layer_pop = [p[index] for p in weights]
            update_factor = self.learning_rate / (len(scores) * self.sigma)
            nw = 0
            for j, layer in enumerate(layer_pop):
                nw += np.dot(layer, scores[j])
            nw = start_weights[index] + update_factor * nw
            new_weights.append(nw)
        return new_weights

    def save_model(self, trial, gen, prepend, wts=None, species=''):

        pth = path.join(getcwd(), 'weights', 't{:03d}_{}weights_s{}_g{}.pth'.format(trial, prepend, species, gen))
        if wts:
            torch.save(wts, pth)

        else:
            torch.save(self.start_weights, pth)
