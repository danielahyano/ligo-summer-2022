import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.gw import conversion
from bilby.core.prior import PriorDict, Uniform, Sine, Cosine, PowerLaw, Constraint, DeltaFunction
from sifce import population
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    prior_gw150914 = dict(
                    mass_1= PowerLaw(alpha=-1, name='mass_1', minimum=10, maximum=80),
                    mass_2= PowerLaw(alpha=-1, name='mass_2', minimum=10, maximum=80),
                    a_1 = Uniform(name='a_1', minimum=0, maximum=0.99),
                    a_2 = Uniform(name='a_2', minimum=0, maximum=0.99),
                    tilt_1 = Sine(name='tilt_1'),
                    tilt_2 = Sine(name='tilt_2'),
                    phi_12 = Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic'),
                    phi_jl = Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic'),
                    theta_jn =  Sine(name='theta_jn'),
                    phase =  Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic'),
                    reference_frequency = DeltaFunction(20))
    
    prior = PriorDict(dictionary = prior_gw150914)
    sim = population.SimulationSet(prior)
    sim.sample_distribution(10)

    print(sim.simulations_dataframe)

    # sim.sample_distribution(10)

    # print(sim.simulations_dataframe)


if __name__ == "__main__":
    main()


