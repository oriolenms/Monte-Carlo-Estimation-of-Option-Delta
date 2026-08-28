import path_simulator
import payoffs
import numpy as np


### EUROPEAN CALL


# Delta estimator (Monte Carlo, Likelihood ratio)
def mc_call_delta_lr_estimate(s,K,T,r,vol,reps):
    Z = np.random.standard_normal(reps)
    paths = path_simulator.one_step_paths(s,T,r,vol,reps,Z)
    deltas = np.exp(-r * T) * (Z/(s*vol*np.sqrt(T))) * np.where(paths > K, paths - K, 0)
    avg_delta = np.mean(deltas)
    return avg_delta

# Delta estimator (Monte Carlo, Likelihood ratio, Payoff centring)
def mc_call_delta_lr_centred_estimate(s,K,T,r,vol,reps,c):
    Z = np.random.standard_normal(reps)
    paths = path_simulator.one_step_paths(s,T,r,vol,reps,Z)
    deltas = (np.exp(-r * T) * np.where(paths > K, paths - K, 0) - c) * (Z/(s*vol*np.sqrt(T)))
    avg_delta = np.mean(deltas)
    return avg_delta


### ASIAN CALL


# Delta estimator (Monte Carlo, Likelihood ratio)
def mc_asian_call_delta_lr_estimate(s,K,T,r,vol,steps,reps):
    Z = np.random.standard_normal((reps,steps))
    paths = path_simulator.multiple_step_paths(s,T,r,vol,steps,reps,Z)
    discount = np.exp(-r*T)
    path_averages = np.mean(paths,axis=1)
    H = discount*payoffs.asian_call_payoff(paths,K)
    dt = T/steps
    W = Z[:,0] / (s*vol*np.sqrt(dt))
    G = discount*(path_averages > K)/paths.shape[1]
    deltas = H*W + G
    avg_delta = np.mean(deltas)
    return avg_delta


# Delta estimator (Monte Carlo, Likelihood ratio, Payoff centring)
def mc_asian_call_delta_lr_centred_estimate(s,K,T,r,vol,steps,reps,c):
    Z = np.random.standard_normal((reps,steps))
    paths = path_simulator.multiple_step_paths(s,T,r,vol,steps,reps,Z)
    discount = np.exp(-r*T)
    path_averages = np.mean(paths,axis=1)
    H = discount*payoffs.asian_call_payoff(paths,K)
    dt = T/steps
    W = Z[:,0] / (s*vol*np.sqrt(dt))
    G = discount*(path_averages > K)/paths.shape[1]
    centred_deltas = (H-c)*W + G
    avg_delta = np.mean(centred_deltas)
    return avg_delta


### DIGITAL CALL


# Delta estimator (Monte Carlo, Likelihood ratio)
def mc_digital_call_delta_lr_estimate(s,K,Q,T,r,vol,reps):
    Z = np.random.standard_normal(reps)
    paths = path_simulator.one_step_paths(s,T,r,vol,reps,Z)
    deltas = np.exp(-r * T) * (Z/(s*vol*np.sqrt(T))) * np.where(paths > K, Q, 0)
    avg_delta = np.mean(deltas)
    return avg_delta

# Delta estimator (Monte Carlo, Likelihood ratio, Payoff centring)
def mc_digital_call_delta_lr_centred_estimate(s,K,Q,T,r,vol,reps,c):
    Z = np.random.standard_normal(reps)
    paths = path_simulator.one_step_paths(s,T,r,vol,reps,Z)
    deltas = (np.exp(-r * T) * np.where(paths > K, Q, 0) - c) * (Z/(s*vol*np.sqrt(T)))
    avg_delta = np.mean(deltas)
    return avg_delta


### BARRIER CALL


# Delta estimator (Monte Carlo, Likelihood ratio)
def mc_barrier_call_delta_lr_estimate(s,K,B,T,r,vol,steps,reps):
    dt = T/steps
    Z = np.random.standard_normal((reps,steps))
    paths = path_simulator.multiple_step_paths(s,T,r,vol,steps,reps,Z)

    # Paths with an observed value above the barrier have zero survival probability
    valid_paths = np.all(paths < B,axis=1)
    delta_observations = np.zeros(reps)
    valid_states = paths[valid_paths]
    valid_Z = Z[valid_paths]

    # Form pairs of consecutive path values for the Brownian-bridge probabilities
    left_states = valid_states[:,:-1]
    right_states = valid_states[:,1:]

    crossing_probs = np.exp(-2*np.log(B/left_states)*np.log(B/right_states)/(vol**2*dt))
    survival_probs = np.prod(1-crossing_probs,axis=1)

    # Only the first transition enters the LR score when path values are held fixed
    scores = valid_Z[:,0]/(s*vol*np.sqrt(dt))

    # Direct dependence on s enters through the first bridge interval
    p0 = crossing_probs[:,0]
    dp0 = p0*2*np.log(B/right_states[:,0])/(s*vol**2*dt)

    remaining_survival = np.prod(1-crossing_probs[:,1:],axis=1)
    survival_gradients = -dp0*remaining_survival

    payoff = np.maximum(valid_states[:,-1]-K,0)

    # Combine the LR score term with the direct survival-probability gradient
    delta_observations[valid_paths] = np.exp(-r*T)*payoff*(survival_probs*scores + survival_gradients)
    avg_delta = np.mean(delta_observations)
    return avg_delta

# Delta estimator (Monte Carlo, Likelihood ratio)
def mc_barrier_call_delta_lr_centred_estimate(s,K,B,T,r,vol,steps,reps,c):
    dt = T/steps
    Z = np.random.standard_normal((reps,steps))
    paths = path_simulator.multiple_step_paths(s,T,r,vol,steps,reps,Z)

    # Scores are required for every path because centring introduces the term -cW
    scores = Z[:,0]/(s*vol*np.sqrt(dt))

    conditional_payoffs = np.zeros(reps)
    survival_gradient_terms = np.zeros(reps)

    # Paths with an observed value above the barrier contribute zero conditional payoff
    valid_paths = np.all(paths < B,axis=1)
    valid_states = paths[valid_paths]
    left_states = valid_states[:,:-1]
    right_states = valid_states[:,1:]

    crossing_probs = np.exp(-2*np.log(B/left_states)*np.log(B/right_states)/(vol**2*dt))
    survival_probs = np.prod(1-crossing_probs,axis=1)

    # Differentiate only the first bridge interval explicitly with respect to s
    p0 = crossing_probs[:,0]
    dp0 = p0*2*np.log(B/right_states[:,0])/(s*vol**2*dt)
    remaining_survival = np.prod(1-crossing_probs[:,1:],axis=1)
    survival_gradients = -dp0*remaining_survival

    payoff = np.maximum(valid_states[:,-1]-K,0)
    conditional_payoffs[valid_paths] = payoff*survival_probs
    survival_gradient_terms[valid_paths] = payoff*survival_gradients

    # Centring changes the score term but leaves the direct gradient unchanged
    delta_observations = np.exp(-r*T)*((conditional_payoffs-c)*scores + survival_gradient_terms)
    avg_delta = np.mean(delta_observations)
    return avg_delta