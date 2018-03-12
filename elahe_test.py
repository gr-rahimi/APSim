from deap import base, creator

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))