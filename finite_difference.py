import mc_pricer
import numpy as np


### EUROPEAN CALL


# Delta estimator (Monte Carlo, Forward difference)
def mc_call_delta_fd_estimate(s,K,T,r,vol,reps,h):
    price1 = mc_pricer.mc_call_price(s,K,T,r,vol,reps)
    price2 = mc_pricer.mc_call_price(s+h,K,T,r,vol,reps)
    delta = (price2 - price1)/h
    return delta

# Delta estimator (Monte Carlo, Central difference)
def mc_call_delta_cd_estimate(s,K,T,r,vol,reps,h):
    price1 = mc_pricer.mc_call_price(s-h,K,T,r,vol,reps)
    price2 = mc_pricer.mc_call_price(s+h,K,T,r,vol,reps)
    delta = (price2 - price1)/(2*h)
    return delta

# Delta estimator (Monte Carlo, Forward difference, CRN)
def mc_call_delta_fd_estimate_crn(s,K,T,r,vol,reps,h,Z=None):
    # Allow Z to be predetermined
    if Z is None:
        Z = np.random.standard_normal(reps)
    else:
        Z = np.asarray(Z) # Ensure array given is a NumPy array
    if len(Z) != reps:
        raise ValueError("Length of Z must equal reps.")
    price1 = mc_pricer.mc_call_price(s,K,T,r,vol,reps,Z)
    price2 = mc_pricer.mc_call_price(s+h,K,T,r,vol,reps,Z)
    delta = (price2 - price1)/h
    return delta

# Delta estimator (Monte Carlo, Central difference, CRN)
def mc_call_delta_cd_estimate_crn(s,K,T,r,vol,reps,h,Z=None):
    # Allow Z to be predetermined
    if Z is None:
        Z = np.random.standard_normal(reps)
    else:
        Z = np.asarray(Z) # Ensure array given is a NumPy array
    if len(Z) != reps:
        raise ValueError("Length of Z must equal reps.")
    price1 = mc_pricer.mc_call_price(s-h,K,T,r,vol,reps,Z)
    price2 = mc_pricer.mc_call_price(s+h,K,T,r,vol,reps,Z)
    delta = (price2 - price1)/(2*h)
    return delta

### Note: Z can be predetermined inside some path_simulator functions, but also inside the two CRN functions above since
#         that allows for extra functionality inside the Jupyter Notebook and when using teh functions; sometimes it's useful
#         to pre-set it in the above functions, and sometimes one level deeper inside the more basic path_simulator functions.


### ASIAN CALL


# Delta estimator (Monte Carlo, Forward difference)
def mc_asian_call_delta_fd_estimate(s,K,T,r,vol,steps,reps,h):
    price1 = mc_pricer.mc_asian_call_price(s,K,T,r,vol,steps,reps)
    price2 = mc_pricer.mc_asian_call_price(s+h,K,T,r,vol,steps,reps)
    delta = (price2 - price1)/h
    return delta

# Delta estimator (Monte Carlo, Central difference)
def mc_asian_call_delta_cd_estimate(s,K,T,r,vol,steps,reps,h):
    price1 = mc_pricer.mc_asian_call_price(s-h,K,T,r,vol,steps,reps)
    price2 = mc_pricer.mc_asian_call_price(s+h,K,T,r,vol,steps,reps)
    delta = (price2 - price1)/(2*h)
    return delta

# Delta estimator (Monte Carlo, Forward difference, CRN)
def mc_asian_call_delta_fd_estimate_crn(s,K,T,r,vol,steps,reps,h,Z=None):
    # Allow Z to be predetermined
    if Z is None:
        Z = np.random.standard_normal((reps,steps))
    else:
        Z = np.asarray(Z) # Ensure array given is a NumPy array
    price1 = mc_pricer.mc_asian_call_price(s,K,T,r,vol,steps,reps,Z)
    price2 = mc_pricer.mc_asian_call_price(s+h,K,T,r,vol,steps,reps,Z)
    delta = (price2 - price1)/h
    return delta

# Delta estimator (Monte Carlo, Central difference, CRN)
def mc_asian_call_delta_cd_estimate_crn(s,K,T,r,vol,steps,reps,h,Z=None):
    # Allow Z to be predetermined
    if Z is None:
        Z = np.random.standard_normal((reps,steps))
    else:
        Z = np.asarray(Z) # Ensure array given is a NumPy array
    price1 = mc_pricer.mc_asian_call_price(s-h,K,T,r,vol,steps,reps,Z)
    price2 = mc_pricer.mc_asian_call_price(s+h,K,T,r,vol,steps,reps,Z)
    delta = (price2 - price1)/(2*h)
    return delta


### DIGITAL CALL


# Delta estimator (Monte Carlo, Forward difference)
def mc_digital_call_delta_fd_estimate(s,K,Q,T,r,vol,reps,h):
    price1 = mc_pricer.mc_digital_call_price(s,K,Q,T,r,vol,reps)
    price2 = mc_pricer.mc_digital_call_price(s+h,K,Q,T,r,vol,reps)
    delta = (price2 - price1)/h
    return delta

# Delta estimator (Monte Carlo, Central difference)
def mc_digital_call_delta_cd_estimate(s,K,Q,T,r,vol,reps,h):
    price1 = mc_pricer.mc_digital_call_price(s-h,K,Q,T,r,vol,reps)
    price2 = mc_pricer.mc_digital_call_price(s+h,K,Q,T,r,vol,reps)
    delta = (price2 - price1)/(2*h)
    return delta

# Delta estimator (Monte Carlo, Forward difference, CRN)
def mc_digital_call_delta_fd_estimate_crn(s,K,Q,T,r,vol,reps,h,Z=None):
    # Allow Z to be predetermined
    if Z is None:
        Z = np.random.standard_normal(reps)
    else:
        Z = np.asarray(Z) # Ensure array given is a NumPy array
    if len(Z) != reps:
        raise ValueError("Length of Z must equal reps.")
    price1 = mc_pricer.mc_digital_call_price(s,K,Q,T,r,vol,reps,Z)
    price2 = mc_pricer.mc_digital_call_price(s+h,K,Q,T,r,vol,reps,Z)
    delta = (price2 - price1)/h
    return delta

# Delta estimator (Monte Carlo, Central difference, CRN)
def mc_digital_call_delta_cd_estimate_crn(s,K,Q,T,r,vol,reps,h,Z=None):
    # Allow Z to be predetermined
    if Z is None:
        Z = np.random.standard_normal(reps)
    else:
        Z = np.asarray(Z) # Ensure array given is a NumPy array
    if len(Z) != reps:
        raise ValueError("Length of Z must equal reps.")
    price1 = mc_pricer.mc_digital_call_price(s-h,K,Q,T,r,vol,reps,Z)
    price2 = mc_pricer.mc_digital_call_price(s+h,K,Q,T,r,vol,reps,Z)
    delta = (price2 - price1)/(2*h)
    return delta


### BARRIER CALL ###


# Delta estimator (Up-And-Out Barrier Call, Approximation to continuously-monitored underlying price, Monte Carlo, Forward difference, CRN)
def mc_barrier_call_delta_fd_estimate_crn(s,K,B,T,r,vol,steps,reps,h,Z=None):
    # Allow Z to be predetermined
    if Z is None:
        Z = np.random.standard_normal((reps,steps))
    else:
        Z = np.asarray(Z) # Ensure array given is a NumPy array
    price1 = mc_pricer.mc_up_and_out_call_price(s,K,B,T,r,vol,steps,reps,Z)
    price2 = mc_pricer.mc_up_and_out_call_price(s+h,K,B,T,r,vol,steps,reps,Z)
    delta = (price2 - price1)/h
    return delta

# Delta estimator (Up-And-Out Barrier Call, Approximation to continuously-monitored underlying price, Monte Carlo, Forward difference, CRN)
def mc_barrier_call_delta_cd_estimate_crn(s,K,B,T,r,vol,steps,reps,h,Z=None):
    # Allow Z to be predetermined
    if Z is None:
        Z = np.random.standard_normal((reps,steps))
    else:
        Z = np.asarray(Z) # Ensure array given is a NumPy array
    price1 = mc_pricer.mc_up_and_out_call_price(s-h,K,B,T,r,vol,steps,reps,Z)
    price2 = mc_pricer.mc_up_and_out_call_price(s+h,K,B,T,r,vol,steps,reps,Z)
    delta = (price2 - price1)/(2*h)
    return delta