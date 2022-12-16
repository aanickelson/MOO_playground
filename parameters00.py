"""
Example parameters file for the rover domain.
"""


class Parameters:
    # This should match the file name -- parameters##
    trial_num = 0

    rew_str = 'G'

    # Domain:
    n_agents = 1
    n_agent_types = 1
    n_poi_types = 2
    rooms = [[0, 10], [10, 0]]
    size = 20
    time_threshold = 20  # How long before information drops out of the state
    time_steps = 50
    sensor_range = 10
    rand_action_rate = 0.05
    n_stat_runs = 5

    # POI:
    value = 1
    obs_radius = 1
    couple = 1
    strong_coupling = False

    # Agent:
    capabilities = False

    # Neural Network:
    hid = 30

    # Evolve nn:
    sigma = 0.1
    learning_rate = 0.05
    n_policies = 50

    # Evo learner
    n_gen = 1000