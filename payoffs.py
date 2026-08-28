import numpy as np

# Basic definitions of payoff functions for each derivative

def call_payoff(s,K):
    return np.maximum(s - K, 0)

def asian_call_payoff(states,K):
    return np.maximum(np.mean(states, axis=-1) - K, 0) # Notice that axis=-1 ensures operation is done for each row in case of a 2D array, and as usual for a 1D array. The -1 chooses the last dimension

# Conditional expected payoff of a continuously monitored up-and-out call, given the simulated path skeleton
def up_and_out_call_payoff(states, K, B, T, vol, steps):
    left_states = states[..., :-1]
    right_states = states[..., 1:]
    dt = T / steps
    valid_intervals = (left_states < B) & (right_states < B)
    safe_left_states = np.where(valid_intervals, left_states, B)
    safe_right_states = np.where(valid_intervals, right_states, B)
    crossing_probs = np.exp(-2 * np.log(B / safe_left_states) * np.log(B / safe_right_states) / (vol**2 * dt))
    crossing_probs = np.where(valid_intervals, crossing_probs, 1.0)
    survival_probs = np.prod(1.0 - crossing_probs, axis=-1)
    terminal_payoffs = np.maximum(states[..., -1] - K, 0.0)
    return terminal_payoffs * survival_probs

def digital_call_payoff(s,K,Q):
    return np.where(s > K, Q, 0)

