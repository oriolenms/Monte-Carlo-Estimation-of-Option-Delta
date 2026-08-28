import numpy as np
from path_simulator import *
from payoffs import *

# Compute price as discounted mean payoff
def discounted_payoff(payoffs,r,T):
    return np.exp(-r * T) * np.mean(payoffs)


# Compute the estimated derivatives prices via Monte Carlo simulations

def mc_call_price(s,K,T,r,vol,reps,Z=None):
    simulations = one_step_paths(s,T,r,vol,reps,Z)
    payoffs = call_payoff(simulations,K)
    return discounted_payoff(payoffs,r,T)

def mc_asian_call_price(s,K,T,r,vol,steps,reps,Z=None):
    simulations = multiple_step_paths(s,T,r,vol,steps,reps,Z)
    payoffs = asian_call_payoff(simulations,K)
    return discounted_payoff(payoffs,r,T)

def mc_digital_call_price(s,K,Q,T,r,vol,reps,Z=None):
    simulations = one_step_paths(s,T,r,vol,reps,Z)
    payoffs = digital_call_payoff(simulations,K,Q)
    return discounted_payoff(payoffs,r,T)

def mc_up_and_out_call_price(s,K,B,T,r,vol,steps,reps,Z=None):
    simulations = multiple_step_paths(s,T,r,vol,steps,reps,Z)
    payoffs = up_and_out_call_payoff(simulations,K,B,T,vol,steps)
    return discounted_payoff(payoffs,r,T)

