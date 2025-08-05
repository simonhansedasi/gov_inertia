import numpy as np
import copy
from tqdm import tqdm
initial_conditions = {
    'N':1000,
    'm':0.01,
    'k':50,
    'G':1,
    'H_0':1,
    'H':0,
    'rho':1.5,
    'theta':3.5,
    'F':0.1,
    'alpha':0.5,
    'beta':0.5,
    'gamma':0.5,
    'delta':20,
    'mu_0':0,
    'sigma':0.1,
    'tau':0.05,
    'lambda':0.001,
    'psi':0.5,
    'epsilon':10,
    'zeta':0.1,
    'eta':0.7,
    'C_B':5,
    'omega':0.95,
    'stolen_funds':0,
    'savings':0,
    'growth_rate':0.015
}

def calculate_interactions(parameters):
    return parameters['m'] * parameters['k'] * parameters['N']


def calculate_enforced_interactions(parameters, I):
    return I * parameters['F']


def calculate_information_capacity(parameters):
    eta = parameters['eta']
    G = parameters['G']
    H = parameters['H_0'] + (parameters['rho'] * G**parameters['theta'])
    parameters['H'] = H
    return eta * G * H

def update_hierarchical_depth(parameters):
    H_0 = parameters['H_0']
    rho = parameters['rho']
    G = parameters['G']
    theta = parameters['theta']

    H = parameters['H_0'] + (parameters['rho'] * G**theta)
    parameters['H'] = H    


def calculate_government_latency(parameters):
    alpha = parameters['alpha']
    N = parameters['N']
    beta = parameters['beta']
    H = parameters['H']
    gamma = parameters['gamma']
    return alpha * (N**beta) * (H**gamma)


def calculate_enforcement_cost(parameters):
    eps = parameters['epsilon']
    k = parameters['k']
    m = parameters['m']
    delta = parameters['delta']
    N = parameters['N']
    zeta = parameters['zeta']
    
    return eps * ((k * m)**delta) * (N**zeta)


def draw_population_wealth(parameters):
    mu = parameters['mu_0']
    sigma = parameters['sigma']
    N = parameters['N']
    
    # estimate average wealth from a lognormal
    expected_mean = np.exp(mu + 0.5 * sigma**2)
    # print(expected_mean)
    base_wealth = N * expected_mean

    # estimate std of wealth distribution
    expected_std = np.sqrt(N) * expected_mean * 0.1
    
    
    noise_factor = np.random.normal(loc=1.0, scale=0.05)
    wealth = base_wealth * noise_factor
    return wealth


def levy_tax(parameters, wealth):
    revenue = wealth * parameters['tau']
    return (revenue)


def redistribute_wealth(parameters, surplus):
    mu_0 = parameters['mu_0']
    B = calculate_information_capacity(parameters)
    L = calculate_government_latency(parameters)

    scaled_surplus = np.log1p(surplus) * parameters['lambda'] * np.log(1 + B)
    scaled_surplus /= (1 + mu_0)  # dampen as mu_0 rises   
    penalty = parameters['psi'] * L
    net_gain = scaled_surplus - penalty

    mu = max(mu_0 + net_gain, mu_0)
    parameters['mu_0'] = mu
    # print(mu)



def calculate_growth_cost(parameters, surplus):
    C_B = parameters['C_B']
    H_0 = parameters['H']
    rho = parameters['rho']
    G = parameters['G']
    theta = parameters['theta']

    c_growth = C_B * ( H_0 + (rho* (G**theta) ) + (rho*theta*(G**theta) ) )
    # print(f'cost to grow: {c_growth}')
    return c_growth


def calculate_growth_potential(parameters, allocated_surplus, c_growth):
    G = parameters['G']
    S = allocated_surplus
    savings = parameters['savings']
    
    DG = (S / c_growth)
    # print(f'DG for this round: {DG}')
    # if DG < 1:
    #     S = S + savings
    #     DG = round(S / c_growth)
    #     parameters['savings'] -= savings
    return DG

def grow_government(parameters, DG):
    parameters['G'] += min(DG, parameters['N'] / 2)


    

def expand_government(parameters, surplus):
    c_growth = calculate_growth_cost(parameters, surplus)
    DG = calculate_growth_potential(parameters, surplus, c_growth)
    grow_government(parameters, DG)
    update_hierarchical_depth(parameters)
    




def reduce_enforcement_cost(parameters, surplus):
    parameters['epsilon'] = (parameters['epsilon'] - (np.exp(-0.00005 * (surplus/3))) / parameters['H'])
    parameters['delta'] = (parameters['delta'] - (0.00005 * np.log1p((surplus/3))) / parameters['H'])
    parameters['zeta'] = (parameters['zeta'] - (0.00005 * np.log1p((surplus/3))) / parameters['H'])
    return parameters



def save_funds_for_later(parameters, surplus):
    parameters['savings'] += surplus



def loss_to_corruption(parameters, surplus):
    parameters['stolen_funds'] += surplus
    return



def determine_surplus_weights(n_activities, noise_level):
    base_weight = 1/n_activities
    weights = np.full(n_activities, base_weight)

    noise = np.random.normal(loc = 0, scale = noise_level, size = n_activities)

    weights += noise
    
    weights = np.clip(weights, a_min = 0, a_max = None)

    weights /= weights.sum()

    return weights





def assign_surplus_weights(surplus, weights):
    surpluses = []
    for weight in weights:
        surpluses.append(surplus * weight)

    return surpluses
    

def spend_surplus(parameters, surpluses):
    # print(f'surpluses: {surpluses}')
    # loss_to_corruption(parameters,surpluses[0])

    expand_government(parameters,surpluses[0])

    reduce_enforcement_cost(parameters,surpluses[1])

    redistribute_wealth(parameters, surpluses[2])
    
    save_funds_for_later(parameters,surpluses[3])




def grow_population(parameters):
    parameters['N'] += parameters['N'] * parameters['growth_rate']











def initialize_government(parameters):
    I = calculate_interactions(parameters)
    I_E = calculate_enforced_interactions(parameters, I)
    B = calculate_information_capacity(parameters)
    while B < I_E:
        parameters['G'] += 1
        B = calculate_information_capacity(parameters)

    return parameters

def run_simulation(parameters, n_cycles):
    parameters = initialize_government(parameters)
    history = []
    snapshot = copy.deepcopy(parameters)
    history.append(snapshot)
    # history.append(parameters)
    for cycle in tqdm(range(n_cycles)):
    
        # calculate cost to run government for given population
        I = calculate_interactions(parameters)
        I_E = calculate_enforced_interactions(parameters, I)
        B = calculate_information_capacity(parameters)
        L = calculate_government_latency(parameters)
        E_C = calculate_enforcement_cost(parameters)
        # print(E_C)
        # calculate total cost
        C_E = E_C * I_E
        C_total = C_E + L
        # print(C_total)
    
        # determine weath of population
        wealth = draw_population_wealth(parameters)
    
    
        # print(C_total, sum(wealth))
        # determine tax revenue to spend
        R = levy_tax(parameters, wealth)
        # print(R)
        
        
                # find tax surplus
        # print(
        #     f'Revenue: {R}\n'
        #     f'Total Cost: {C_total}\n'
        #     f'Surplus: {surplus}\n'
        # )
        if R < C_total:
            # print('gov failed')
            
            while R < C_total:
                # print('raising taxes')
                parameters['tau'] += 0.0001
                R = levy_tax(parameters, wealth)
        surplus = R - C_total

        stolen_funds = surplus * (1-parameters['omega'])
        
        parameters['stolen_funds'] += stolen_funds
        
        surplus -= stolen_funds 
        
        # print(surplus)
        # print()
    
        # print(surplus)
    
        # weight decisions on how to spend surplus
        weights = determine_surplus_weights(4, 0.1)
    
        # assign weights to surplus
        surpluses = assign_surplus_weights(surplus, weights)
    
        # spend surplus
        spend_surplus(parameters, surpluses)
    
        # grow population
        grow_population(parameters)
        snapshot = copy.deepcopy(parameters)
        history.append(snapshot)
        parameters['tau'] = 0.01

    return history


















    