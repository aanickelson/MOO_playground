import jax.numpy as jnp
import jax
from typing import Tuple

from functools import partial
import flax
import chex
import qdax
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.mome import MOME
from qdax.core.emitters.mutation_operators import (
    polynomial_mutation,
    polynomial_crossover,
)
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.plotting import plot_2d_map_elites_repertoire, plot_mome_pareto_fronts

from qdax.utils.metrics import default_moqd_metrics
from MOO_playground.problem_wrapper import RoverWrapper
import numpy as np
import matplotlib.pyplot as plt

from qdax.types import Fitness, Descriptor, RNGKey, ExtraScores
from teaming.domain import DiscreteRoverDomain
from MOO_playground.learning.neuralnet_no_hid import NeuralNetwork as NN
from torch import from_numpy
from parameters04 import Parameters as p

env = DiscreteRoverDomain(p)
st_size = env.state_size()
act_size = env.get_action_size()
l1_size = st_size * act_size
hid = 0
model = NN(st_size, hid, act_size)

pareto_front_max_length = 50  # @param {type:"integer"}
num_variables = l1_size  # @param {type:"integer"}
num_iterations = 1000  # @param {type:"integer"}

num_centroids = 64  # @param {type:"integer"}
minval = 0  # @param {type:"number"}
maxval = 1  # @param {type:"number"}
proportion_to_mutate = 0.6  # @param {type:"number"}
eta = 1  # @param {type:"number"}
proportion_var_to_change = 0.5  # @param {type:"number"}
crossover_percentage = 1.  # @param {type:"number"}
batch_size = 100  # @param {type:"integer"}
lag = 2.2  # @param {type:"number"}
base_lag = 0  # @param {type:"number"}



def rastrigin_scorer(genotypes: jnp.ndarray, l1, act, st, nn_model, rov_env) -> Tuple[Fitness, Descriptor]:
    """
    Rastrigin Scorer with first two dimensions as descriptors
    """
    descriptors = genotypes[:, :2]
    scores = []
    for i in range(genotypes.shape[0]):
        l1_wts = genotypes[i]
        l1_max = l1_wts.argmax()
        l1_wts = from_numpy(np.reshape(np.array(genotypes[i]), (act, st)))
        nn_model.set_weights([l1_wts])
        rov_env.run_sim([nn_model])
        multi_g = rov_env.multiG()

    # scores = jnp.stack([f1, f2], axis=-1)

    return scores, descriptors


scoring_function = partial(rastrigin_scorer, l1=l1_size, act=act_size, st=st_size, nn_model=model, rov_env=env)


def scoring_fn(genotypes: jnp.ndarray, random_key: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    fitnesses, descriptors = scoring_function(genotypes)
    return fitnesses, descriptors, {}, random_key


reference_point = jnp.array([ -150, -150])

# how to compute metrics from a repertoire
metrics_function = partial(
    default_moqd_metrics,
    reference_point=reference_point
)

# initial population
random_key = jax.random.PRNGKey(42)
random_key, subkey = jax.random.split(random_key)
init_genotypes = jax.random.uniform(
    random_key, (batch_size, num_variables), minval=minval, maxval=maxval, dtype=jnp.float32
)

# crossover function
crossover_function = partial(
    polynomial_crossover,
    proportion_var_to_change=proportion_var_to_change
)

# mutation function
mutation_function = partial(
    polynomial_mutation,
    eta=eta,
    minval=minval,
    maxval=maxval,
    proportion_to_mutate=proportion_to_mutate
)

# Define emitter
mixing_emitter = MixingEmitter(
    mutation_fn=mutation_function,
    variation_fn=crossover_function,
    variation_percentage=crossover_percentage,
    batch_size=batch_size
)

centroids, random_key = compute_cvt_centroids(
    num_descriptors=2,
    num_init_cvt_samples=20000,
    num_centroids=num_centroids,
    minval=minval,
    maxval=maxval,
    random_key=random_key,
)

mome = MOME(
    scoring_function=scoring_fn,
    emitter=mixing_emitter,
    metrics_function=metrics_function,
)

repertoire, emitter_state, random_key = mome.init(
    init_genotypes,
    centroids,
    pareto_front_max_length,
    random_key
)


# Run the algorithm
(repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
    mome.scan_update,
    (repertoire, emitter_state, random_key),
    (),
    length=num_iterations,
)

moqd_scores = jnp.sum(metrics["moqd_score"], where=metrics["moqd_score"] != -jnp.inf, axis=-1)

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 5))

steps = batch_size * jnp.arange(start=0, stop=num_iterations)
ax1.plot(steps, moqd_scores)
ax1.set_xlabel('Num steps')
ax1.set_ylabel('MOQD Score')

ax2.plot(steps, metrics["max_hypervolume"])
ax2.set_xlabel('Num steps')
ax2.set_ylabel('Max Hypervolume')

ax3.plot(steps, metrics["max_sum_scores"])
ax3.set_xlabel('Num steps')
ax3.set_ylabel('Max Sum Scores')

ax4.plot(steps, metrics["coverage"])
ax4.set_xlabel('Num steps')
ax4.set_ylabel('Coverage')
plt.show()

fig, axes = plt.subplots(figsize=(18, 6), ncols=3)

# plot pareto fronts
axes = plot_mome_pareto_fronts(
    centroids,
    repertoire,
    minval=minval,
    maxval=maxval,
    color_style='spectral',
    axes=axes,
    with_global=True
)

# add map elites plot on last axe
plot_2d_map_elites_repertoire(
    centroids=centroids,
    repertoire_fitnesses=metrics["moqd_score"][-1],
    minval=minval,
    maxval=maxval,
    ax=axes[2]
)
plt.show()