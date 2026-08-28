import path_simulator
import numpy as np
from scipy.stats import norm


### EUROPEAN CALL


# Delta estimator (Monte Carlo, Pathwise)
def mc_call_delta_pw_estimate(s,K,T,r,vol,reps,Z=None):
    paths = path_simulator.one_step_paths(s,T,r,vol,reps,Z)
    path_sensitivities = paths/s
    deltas = np.exp(-r * T) * np.where(paths > K, path_sensitivities, 0)
    avg_delta = np.mean(deltas)
    return avg_delta


### ASIAN CALL


# Delta estimator (Monte Carlo, Pathwise)
def mc_asian_call_delta_pw_estimate(s,K,T,r,vol,steps,reps,Z=None):
    paths = path_simulator.multiple_step_paths(s,T,r,vol,steps,reps,Z)
    path_averages = np.mean(paths, axis=1)
    average_gradients = np.mean(paths/s, axis=1) # Derivative of each arithmetic average with respect to s
    delta_observations = (np.exp(-r * T) * (path_averages > K) * average_gradients)
    delta = np.mean(delta_observations)
    return delta


### DIGITAL CALL


def mc_digital_call_delta_pw_estimate(s,K,Q,T,r,vol,reps,epsilon,Z=None):
    paths = path_simulator.one_step_paths(s,T,r,vol,reps,Z)
    path_sensitivities = paths/s
    pdf_obs = norm.pdf((paths - K)/epsilon)
    delta = np.mean(path_sensitivities * pdf_obs) * (Q*np.exp(-r*T) / epsilon)
    return delta


### BARRIER CALL


def mc_barrier_call_delta_pw_estimate(s,K,barrier,T,r,vol,steps,reps,Z=None):
    # Use the supplied shocks when provided, allowing identical paths to be reused
    paths = path_simulator.multiple_step_paths(s,T,r,vol,steps,reps,Z)
    dt = T/steps

    # Only paths whose discrete skeleton remains below the barrier can survive
    valid_paths = np.all(paths < barrier, axis=1)
    
    delta_observations = np.zeros(reps)

    # Work only with valid paths
    valid_states = paths[valid_paths]

    # Form pairs of consecutive states for each Brownian-bridge interval
    left_states = valid_states[:, :-1]
    right_states = valid_states[:, 1:]

    # Conditional probability of crossing the barrier between each pair of states
    crossing_probs = np.exp(-2 * np.log(barrier/left_states) * np.log(barrier/right_states) / (vol**2 * dt))

    # Compute the total survival probability and its pathwise derivative
    survival_probs = np.prod(1-crossing_probs, axis=1)
    crossing_prob_gradients = (crossing_probs * 2 * (np.log(barrier/left_states) + np.log(barrier/right_states)) / (vol**2 * dt * s))
    survival_prob_gradients = (-survival_probs * np.sum(crossing_prob_gradients/(1-crossing_probs), axis=1))

    # Pathwise derivative of the terminal asset value with respect to s
    terminal_states = valid_states[:, -1]
    terminal_gradients = terminal_states/s

    # Combine the terminal-payoff derivative and survival-probability derivative
    delta_observations[valid_paths] = np.exp(-r*T) * ((terminal_states > K) * terminal_gradients * survival_probs + np.maximum(terminal_states-K,0) * survival_prob_gradients)

    delta = np.mean(delta_observations)
    return delta
