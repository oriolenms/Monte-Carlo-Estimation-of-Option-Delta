import numpy as np

# Generate multiple simulations of one-step random paths under geometric Brownian motion
def one_step_paths(s,T,r,vol,reps,Z=None):
    if Z is None:
        Z = np.random.standard_normal(reps)
    else:
        Z = np.asarray(Z) # Ensure array given is a NumPy array
    if len(Z) != reps:
        raise ValueError("Length of Z must equal reps.")
    paths = s * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z)
    return paths

# Generate multiple-step paths as a NumPy 2D array containing all path states
def multiple_step_paths(s,T,r,vol,steps,reps,Z=None):
    if Z is None:
        Z = np.random.standard_normal((reps,steps))
    else:
        Z = np.asarray(Z) # Ensure array given is a NumPy array
    if Z is not None and (Z.shape[0] != reps or Z.shape[1] != steps):
        raise ValueError("Dimensions of Z must equal reps (rows) x steps (columns).")
    dt = T/steps
    increments = (r - 0.5 * vol**2) * (dt) + vol * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis = 1) # Cumulative sum horizontally, along each individual path
    paths = s * np.exp(log_paths)
    initial_column = np.full((reps, 1), s)
    paths = np.hstack((initial_column, paths))
    return paths