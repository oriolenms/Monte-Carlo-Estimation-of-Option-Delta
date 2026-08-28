# Monte Carlo Estimation of Option Delta

This project implements and compares several Monte Carlo estimators for the Delta of financial derivatives under the Black-Scholes-Merton model. The methods considered are Forward and Central Differences, Pathwise Differentiation, and the Likelihood-Ratio method. Common Random Numbers, payoff centring, payoff smoothing, and Brownian-bridge techniques are also implemented where appropriate.

The estimators are studied across European, arithmetic Asian, digital, and continuously monitored up-and-out barrier call options. Their performance is evaluated using mean squared error (MSE), relative MSE, runtime, and Runtime $\times$ MSE across different parameter regions. The main aim is to compare numerical accuracy and determine how the regularity and path dependence of an option payoff should influence the choice of a Delta estimator.

The project is accompanied by five Jupyter Notebooks containing the theoretical derivations, implementations, numerical experiments, tables, and plots, as well as a report summarising the methodology and main results.

## Main features

- Simulation of asset paths under the Black-Scholes-Merton model.
- Monte Carlo estimation of Delta using Forward and Central Differences, Pathwise Differentiation, and Likelihood-Ratio estimators.
- Implementation of Common Random Numbers for Finite-Difference estimators.
- Numerical selection of approximately optimal finite-difference step sizes.
- Implementation of payoff centring for Likelihood-Ratio estimators.
- Smoothing of the digital payoff for Pathwise Differentiation, including numerical selection of the smoothing parameter.
- Brownian-bridge treatment of continuously monitored barrier calls.
- Analysis across moneyness-volatility grids and method-specific parameter grids.
- Comparison using MSE, relative MSE, runtime, and combined error-runtime metrics.
- Validation against closed-form Deltas (where available) and against a high-precision Pathwise estimate for the arithmetic Asian call.

## Main findings

The results indicate that the best estimator depends mainly on payoff regularity.

- **European and arithmetic Asian calls:** Pathwise Differentiation provides the most accurate and computationally efficient estimates. For the Asian call, its MSE also remains comparatively stable as the number of averaging intervals increases.
- **Digital calls:** the payoff-centred Likelihood-Ratio estimator performs best. Smoothing makes Pathwise Differentiation applicable, but introduces bias and an additional tuning parameter.
- **Barrier calls:** the preferred estimator changes with barrier proximity. Remote barriers produce increasingly European-like behaviour, favouring Pathwise Differentiation. When the barrier is close and its discontinuity becomes important, the Likelihood-Ratio method becomes more suitable when Finite Differences are restricted to sufficiently local step sizes.
- **General-purpose estimation:** Central Differences with Common Random Numbers remain competitive across all the payoff structures considered. They provide a robust alternative when payoff regularity is unknown, although an approximately optimal step size must be selected.

The full discussion, including the limitations of these conclusions, is available in the [project report](report/Monte_Carlo_Estimation_of_Option_Delta.pdf).

## Repository structure

```text
.
├── figures/
│   └── Generated figures used in the report
├── notebooks/
│   ├── 01_estimator_theory.ipynb
│   ├── 02_european_call.ipynb
│   ├── 03_asian_call.ipynb
│   ├── 04_digital_call.ipynb
│   └── 05_barrier_call.ipynb
├── report/
│   ├── Monte_Carlo_Estimation_of_Option_Delta.pdf
├── src/
│   ├── estimators/
│   │   ├── finite_difference.py
│   │   ├── pathwise.py
│   │   └── likelihood_ratio.py
│   ├── mc_pricer.py
│   ├── path_simulator.py
│   └── payoffs.py
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## File descriptions

- `src/path_simulator.py`: simulates one-step and multi-step asset-price paths under the Black-Scholes-Merton model.
- `src/payoffs.py`: defines the payoff functions used by the pricing and Delta estimators, including the Brownian-bridge conditional payoff for the barrier call.
- `src/mc_pricer.py`: contains the Monte Carlo pricing functions used by the Finite-Difference estimators.
- `src/estimators/finite_difference.py`: contains the Forward- and Central-Difference estimators, with both independent sampling and Common Random Numbers.
- `src/estimators/pathwise.py`: contains the naive and option-specific Pathwise estimators, including digital-payoff smoothing and the barrier Brownian-bridge implementation.
- `src/estimators/likelihood_ratio.py`: contains the ordinary and payoff-centred Likelihood-Ratio estimators, together with their option-specific implementations.
- `notebooks/01_estimator_theory.ipynb`: presents the estimator theory, derivations, bias and variance properties, convergence results, and references used throughout the project.
- `notebooks/02_european_call.ipynb`: analyses the estimators for the European call and establishes the main experimental framework.
- `notebooks/03_asian_call.ipynb`: studies the effect of path dependence and averaging frequency for the arithmetic Asian call.
- `notebooks/04_digital_call.ipynb`: studies the consequences of payoff discontinuity, including Pathwise smoothing and payoff-centred Likelihood-Ratio estimation.
- `notebooks/05_barrier_call.ipynb`: studies the effect of barrier proximity and implements Brownian-bridge estimators for continuous monitoring.
- `report/Monte_Carlo_Estimation_of_Option_Delta.pdf`: summarises the methodology, experimental design, main results, discussion, and conclusions.

## Requirements and installation

The project is intended to be run with Python 3. The required packages are listed in `requirements.txt`.

It is recommended to create a virtual environment before installing them:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

The editable installation makes the modules under `src/` available to the notebooks while retaining the repository structure.

## Running the project

Launch Jupyter from the repository root:

```bash
jupyter lab
```

The notebooks are intended to be read in numerical order. `01_estimator_theory.ipynb` explains the methods and theoretical results, while the remaining notebooks apply them successively to European, Asian, digital, and barrier calls.

Some experiments use large simulation budgets or repeated evaluations over parameter grids and may take a considerable amount of time to reproduce fully. Particularly expensive cells are commented out in the notebooks. The saved outputs and figures contain the results used in the report.

Because the project uses random simulation, exact numerical values may vary slightly between runs. The main qualitative patterns and estimator comparisons should remain consistent.

## Report

The complete report can be found [here](report/Monte_Carlo_Estimation_of_Option_Delta.pdf). It contains a more concise presentation of the experimental design and results, as well as the overall estimator-selection rule and limitations of the investigation.
