import numpy as np
import copy
from tqdm import tqdm

initial_conditions = {
    # ---- Population ----
    'N': 1000,
    'growth_rate': 0.015,

    # ---- Interaction parameters ----
    'm': 0.01,
    'k': 50,

    # ---- Government structure ----
    'G': 1,
    'H_0': 1,
    'H': 0,
    'rho': 1.5,
    'theta': 3.5,

    # ---- Enforcement capacity ----
    'F': 0.1,
    'eta': 0.7,

    # ---- Latency ----
    'alpha': 0.5,
    'beta': 0.5,
    'gamma': 0.5,

    # ---- Surveillance cost ----
    'delta': 20,
    'epsilon': 10,
    'zeta': 0.1,

    # ---- Economics ----
    'mu_0': 0,
    'sigma': 0.1,
    'tau': 0.05,

    # ---- Laffer curve ----
    # W(τ) = W_base · exp(-kappa_tau · τ)
    # Revenue-maximising τ* = 1 / kappa_tau
    # kappa_tau = 1 → peak at τ = 1.0; gentle suppression at realistic rates
    'kappa_tau': 1.0,

    # ---- Government growth ----
    'C_B': 5,

    # ---- Corruption ----
    # omega: fraction of surplus retained; (1-omega) is stolen each cycle
    'omega': 0.95,

    # ---- Divergence ----
    # D = min(1, D_0 + lambda_D · log(1+H)^gamma_D)
    'D_0': 0.001,
    'lambda_D': 0.00245,
    'gamma_D': 1.70,

    # ---- Legitimacy ----
    # Builds with effective governance, erodes with divergence and corruption
    'Lambda': 0.5,
    'lambda_build': 0.002,
    'lambda_erode': 0.01,
    'lambda_corr': 0.02,

    # ---- Compliance ----
    # Blends voluntary (legitimacy-weighted) and coerced compliance
    'tau_ref': 0.01,
    'phi_tau': 0.5,

    # ---- Social enforcement cost ----
    # C_social = phi_e · N · (D + (1 - mean_c))
    'phi_e': 0.002,

    # ---- Public goods (Barro 1990) ----
    # Effect on mu_0: lambda_g · (g_share)^phi_g · (1 - g_share)
    # Peaks at g_share* = phi_g / (1 + phi_g)
    'phi_g': 0.3,
    'lambda_g': 0.05,

    # ---- Inequality ----
    # Gini drifts upward naturally; public goods investment pushes it down
    'gini': 0.35,
    'gini_drift': 0.0001,
    'gini_redist': 0.002,
    'phi_ineq': 0.2,      # compliance penalty per unit of Gini

    # ---- Failure probability ----
    # P_f = sigmoid(kappa_pf · (D - D_crit))
    'kappa_pf': 10.0,
    'D_crit': 0.3,

    # ---- Democratic correction ----
    # Every T_election cycles, if D > D_ref, reform partially realigns policy
    'T_election': 50,
    'D_ref': 0.05,
    'reform_strength': 0.3,
    'reform_cost_tau': 0.005,

    # ---- Persistent state ----
    'D': 0.0,
    'mean_c': 1.0,
    'P_f': 0.0,
    'Lambda_current': 0.5,
    'collapse_events': 0,
    'reform_events': 0,
    'insolvent_count': 0,

    # ---- Per-cycle budget (written back for history snapshots) ----
    'C_surveillance': 0.0,
    'C_social': 0.0,
    'C_total_computed': 0.0,
    'R': 0.0,
    'W': 0.0,
    'collapse': 0,
    'reform': 0,
    'insolvent': 0,

    # ---- Accumulated state ----
    'stolen_funds': 0,
    'savings': 0,
}


# ---------------------------------------------------------------------------
# Core government calculations
# ---------------------------------------------------------------------------

def calculate_interactions(parameters):
    return parameters['m'] * parameters['k'] * parameters['N']


def calculate_enforced_interactions(parameters, I):
    return I * parameters['F']


def update_hierarchical_depth(parameters):
    H = parameters['H_0'] + parameters['rho'] * (parameters['G'] ** parameters['theta'])
    parameters['H'] = H


def calculate_information_capacity(parameters):
    update_hierarchical_depth(parameters)
    return parameters['eta'] * parameters['G'] * parameters['H']


def calculate_government_latency(parameters):
    return (parameters['alpha']
            * (parameters['N'] ** parameters['beta'])
            * (parameters['H'] ** parameters['gamma']))


def calculate_enforcement_cost(parameters):
    """Surveillance cost per enforced interaction."""
    return (parameters['epsilon']
            * (parameters['k'] * parameters['m']) ** parameters['delta']
            * parameters['N'] ** parameters['zeta'])


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

def calculate_divergence(parameters):
    """
    D = min(1, D_0 + lambda_D · log(1+H)^gamma_D)

    Policy divergence grows as bureaucratic depth H increases. Represents
    how far government behaviour has drifted from population preferences.
    """
    D = min(1.0, (parameters['D_0']
                  + parameters['lambda_D']
                  * np.log1p(parameters['H']) ** parameters['gamma_D']))
    parameters['D'] = D
    return D


# ---------------------------------------------------------------------------
# Legitimacy
# ---------------------------------------------------------------------------

def update_legitimacy(parameters, B, I_E):
    """
    Legitimacy Λ builds when government is effective and aligned with the
    population; it erodes with divergence and corruption.

    Build: lambda_build · min(1, B/I_E) · (1 - D)
    Erode: lambda_erode · D + lambda_corr · (1 - omega)
    """
    D = parameters['D']
    capacity_ratio = min(1.0, B / max(I_E, 1e-10))
    build = parameters['lambda_build'] * capacity_ratio * (1.0 - D)
    erode = (parameters['lambda_erode'] * D
             + parameters['lambda_corr'] * (1.0 - parameters['omega']))
    parameters['Lambda'] = float(np.clip(parameters['Lambda'] + build - erode, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

def calculate_compliance(parameters, tau_current):
    """
    Compliance blends a legitimacy-weighted voluntary component and a
    coercion-based component:

        c̄ = Λ · (1 - D) + (1 - Λ) · max(0, 1 - D - phi_tau · max(0, τ - τ_ref))

    A high-legitimacy government retains compliance even under tax pressure;
    a low-legitimacy government loses it rapidly. Both components are
    additionally penalised by inequality (Gini).
    """
    D = parameters['D']
    Lambda = parameters['Lambda']

    c_voluntary = max(0.0, 1.0 - D)

    tax_pressure = parameters['phi_tau'] * max(0.0, tau_current - parameters['tau_ref'])
    c_coerced = max(0.0, 1.0 - D - tax_pressure)

    mean_c = Lambda * c_voluntary + (1.0 - Lambda) * c_coerced

    # Inequality penalty: high Gini suppresses average compliance
    mean_c *= max(0.0, 1.0 - parameters['phi_ineq'] * parameters['gini'])
    mean_c = max(0.0, mean_c)

    parameters['mean_c'] = mean_c
    return mean_c


# ---------------------------------------------------------------------------
# Social enforcement cost
# ---------------------------------------------------------------------------

def calculate_social_cost(parameters, mean_c):
    """
    C_social = phi_e · N · (D + (1 - c̄))

    D captures the cost of enforcing laws the population regards as
    illegitimate regardless of individual disposition. (1 - c̄) captures
    the direct cost of processing active violators.
    """
    return parameters['phi_e'] * parameters['N'] * (parameters['D'] + (1.0 - mean_c))


# ---------------------------------------------------------------------------
# Failure probability and collapse
# ---------------------------------------------------------------------------

def calculate_failure_probability(parameters):
    """
    P_f = sigmoid(kappa_pf · (D - D_crit))

    Crosses 0.5 when divergence reaches D_crit.
    """
    score = parameters['kappa_pf'] * (parameters['D'] - parameters['D_crit'])
    P_f = 1.0 / (1.0 + np.exp(-score))
    parameters['P_f'] = P_f
    return P_f


def apply_collapse(parameters):
    """
    Stochastic partial collapse. Fires at rate P_f · 0.02 when D > D_crit.
    Severity s ~ Uniform(0.1, 0.5) governs loss of agencies and economic damage.
    Returns True if a collapse event occurred.
    """
    if parameters['D'] <= parameters['D_crit']:
        return False
    if np.random.random() >= parameters['P_f'] * 0.02:
        return False

    severity = np.random.uniform(0.1, 0.5)
    parameters['G'] = max(1.0, parameters['G'] * (1.0 - severity * 0.2))
    parameters['mu_0'] -= severity * 0.5
    parameters['Lambda'] = max(0.0, parameters['Lambda'] - severity * 0.3)
    parameters['collapse_events'] += 1
    update_hierarchical_depth(parameters)
    return True


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------

def draw_population_wealth(parameters):
    """
    Draw base wealth W_base before tax suppression. Stochastic: called
    once per cycle. Pass to apply_laffer() to get actual productive wealth.
    """
    expected_mean = np.exp(parameters['mu_0'] + 0.5 * parameters['sigma'] ** 2)
    noise = np.random.normal(loc=1.0, scale=0.05)
    return parameters['N'] * expected_mean * noise


def apply_laffer(base_wealth, tau, kappa_tau):
    """
    W(τ) = W_base · exp(-κ_τ · τ)

    Tax suppression of productive activity. Revenue R = τ · W(τ) is
    maximised at τ* = 1 / κ_τ (the Laffer peak). Beyond this raising
    taxes shrinks revenue — a fiscal trap.
    """
    return base_wealth * np.exp(-kappa_tau * tau)


def levy_tax(W, tau):
    return tau * W


def spend_on_public_goods(parameters, surplus, W):
    """
    Invests in public goods. Two effects:

    1. Barro (1990) productivity multiplier on mu_0:
       Δμ₀ = lambda_g · (g_share)^phi_g · (1 - g_share)
       Peaks at g_share* = phi_g / (1 + phi_g) ≈ 23% of GDP with defaults.

    2. Redistribution: higher public goods share reduces the Gini coefficient.
    """
    g_share = surplus / max(W, 1e-10)
    phi_g = parameters['phi_g']

    barro_effect = (parameters['lambda_g']
                    * (g_share ** phi_g)
                    * max(0.0, 1.0 - g_share))
    parameters['mu_0'] += barro_effect

    redist = parameters['gini_redist'] * (surplus / max(W, 1e-10))
    parameters['gini'] = max(0.0, parameters['gini'] - redist)


def update_inequality(parameters):
    """Natural upward drift of the Gini coefficient each cycle."""
    parameters['gini'] = min(1.0, parameters['gini'] + parameters['gini_drift'])


def reduce_enforcement_cost(parameters, surplus):
    parameters['epsilon'] = max(
        1e-5,
        parameters['epsilon'] - np.exp(-0.00005 * (surplus / 3)) / parameters['H']
    )
    parameters['delta'] = max(
        1.0,
        parameters['delta'] - 0.00005 * np.log1p(surplus / 3) / parameters['H']
    )
    parameters['zeta'] = max(
        0.01,
        parameters['zeta'] - 0.00005 * np.log1p(surplus / 3) / parameters['H']
    )


def save_funds_for_later(parameters, surplus):
    parameters['savings'] += surplus


# ---------------------------------------------------------------------------
# Government expansion
# ---------------------------------------------------------------------------

def calculate_growth_cost(parameters):
    G = parameters['G']
    H = parameters['H']
    rho = parameters['rho']
    theta = parameters['theta']
    return parameters['C_B'] * (H + rho * G ** theta + rho * theta * G ** theta)


def expand_government(parameters, surplus):
    c_growth = calculate_growth_cost(parameters)
    DG = surplus / c_growth
    parameters['G'] += min(DG, parameters['N'] / 2)
    update_hierarchical_depth(parameters)


# ---------------------------------------------------------------------------
# Surplus allocation
# ---------------------------------------------------------------------------

def _random_weights(n, noise=0.1):
    weights = np.full(n, 1.0 / n)
    weights += np.random.normal(0, noise, n)
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()
    return weights


def spend_surplus(parameters, surplus, W):
    """
    Allocates surplus across four activities with randomised weights:
      1. Government expansion
      2. Enforcement efficiency gains
      3. Public goods investment (Barro productivity + inequality reduction)
      4. Savings reserve
    """
    w = _random_weights(4)
    expand_government(parameters, surplus * w[0])
    reduce_enforcement_cost(parameters, surplus * w[1])
    spend_on_public_goods(parameters, surplus * w[2], W)
    save_funds_for_later(parameters, surplus * w[3])


def grow_population(parameters):
    parameters['N'] += parameters['N'] * parameters['growth_rate']


# ---------------------------------------------------------------------------
# Democratic correction
# ---------------------------------------------------------------------------

def apply_democratic_correction(parameters, cycle):
    """
    Every T_election cycles, if divergence exceeds D_ref, a reform event
    partially realigns government policy with population preferences.

    Effects:
      - D is reduced by reform_strength · (D - D_ref)
      - G is trimmed by 3% (bureaucratic reform)
      - Legitimacy gets a boost of 0.1
      - A one-cycle transition tax cost is applied

    Returns True if a reform occurred.
    """
    if cycle % parameters['T_election'] != 0:
        return False
    if parameters['D'] <= parameters['D_ref']:
        return False

    D_excess = parameters['D'] - parameters['D_ref']
    parameters['D'] = max(parameters['D_ref'],
                          parameters['D'] - parameters['reform_strength'] * D_excess)

    parameters['tau'] += parameters['reform_cost_tau']
    parameters['G'] = max(1.0, parameters['G'] * 0.97)
    update_hierarchical_depth(parameters)

    parameters['Lambda'] = min(1.0, parameters['Lambda'] + 0.1)
    parameters['reform_events'] += 1
    return True


# ---------------------------------------------------------------------------
# Fiscal insolvency
# ---------------------------------------------------------------------------

def handle_insolvency(parameters):
    """
    Called when costs cannot be covered even at the Laffer-peak tax rate.
    Government is forced to downsize: loses 5% of agencies and takes a
    legitimacy hit. Economic damage is not applied here — that is reserved
    for stochastic collapse events, which carry heavier consequences.
    """
    parameters['G'] = max(1.0, parameters['G'] * 0.95)
    parameters['Lambda'] = max(0.0, parameters['Lambda'] - 0.05)
    parameters['insolvent_count'] += 1
    update_hierarchical_depth(parameters)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_government(parameters):
    """Scale G up until information capacity meets enforcement load."""
    I = calculate_interactions(parameters)
    I_E = calculate_enforced_interactions(parameters, I)
    B = calculate_information_capacity(parameters)
    while B < I_E:
        parameters['G'] += 1
        B = calculate_information_capacity(parameters)
    return parameters


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(parameters, n_cycles):
    parameters = initialize_government(parameters)
    history = [copy.deepcopy(parameters)]

    for cycle in tqdm(range(n_cycles)):

        # --- Enforcement load and capacity ---
        I = calculate_interactions(parameters)
        I_E = calculate_enforced_interactions(parameters, I)
        B = calculate_information_capacity(parameters)   # also updates H
        L = calculate_government_latency(parameters)

        # --- Surveillance cost ---
        E_C = calculate_enforcement_cost(parameters)
        C_surveillance = E_C * I_E + L

        # --- Divergence (depends on current H) ---
        calculate_divergence(parameters)

        # --- Base wealth drawn once; Laffer applied inside loop ---
        base_wealth = draw_population_wealth(parameters)
        tau_laffer_peak = 0.99 / max(parameters['kappa_tau'], 1e-10)

        # --- Combined cost loop ---
        # Recompute W, R, compliance, and C_social as tau is raised.
        # Loop terminates at the Laffer peak if costs are still unmet.
        W = apply_laffer(base_wealth, parameters['tau'], parameters['kappa_tau'])
        R = levy_tax(W, parameters['tau'])
        mean_c = calculate_compliance(parameters, parameters['tau'])
        C_social = calculate_social_cost(parameters, mean_c)
        C_total = C_surveillance + C_social

        insolvent = False
        while R < C_total:
            if parameters['tau'] >= tau_laffer_peak:
                insolvent = True
                break
            parameters['tau'] = min(parameters['tau'] + 0.0001, tau_laffer_peak)
            W = apply_laffer(base_wealth, parameters['tau'], parameters['kappa_tau'])
            R = levy_tax(W, parameters['tau'])
            mean_c = calculate_compliance(parameters, parameters['tau'])
            C_social = calculate_social_cost(parameters, mean_c)
            C_total = C_surveillance + C_social

        if insolvent:
            handle_insolvency(parameters)

        # --- Surplus distribution ---
        surplus = max(0.0, R - C_total)
        stolen = surplus * (1.0 - parameters['omega'])
        parameters['stolen_funds'] += stolen
        surplus -= stolen

        spend_surplus(parameters, surplus, W)

        # --- Natural inequality drift ---
        update_inequality(parameters)

        # --- Legitimacy update (based on this cycle's governance outcomes) ---
        update_legitimacy(parameters, B, I_E)

        # --- Failure probability and stochastic collapse ---
        P_f = calculate_failure_probability(parameters)
        collapsed = apply_collapse(parameters)

        # --- Periodic democratic correction ---
        reformed = apply_democratic_correction(parameters, cycle)

        # --- Record computed values for history snapshot ---
        parameters['C_surveillance'] = C_surveillance
        parameters['C_social'] = C_social
        parameters['C_total_computed'] = C_total
        parameters['R'] = R
        parameters['W'] = W
        parameters['collapse'] = int(collapsed)
        parameters['reform'] = int(reformed)
        parameters['insolvent'] = int(insolvent)
        parameters['Lambda_current'] = parameters['Lambda']

        grow_population(parameters)
        history.append(copy.deepcopy(parameters))

        # Reset tau to equilibrium for next cycle
        parameters['tau'] = parameters['tau_ref']

    return history
