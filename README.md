# Government Inertia Simulation

A mathematical model of government as an evolving dynamical system. Models bureaucratic growth, legitimacy, public goods provision, wealth inequality, and eventual collapse driven by the divergence between government behaviour and population preferences.

---

## Instructions

### Requirements

```
numpy
matplotlib
pandas
tqdm
```

Install with:

```bash
pip install numpy matplotlib pandas tqdm
```

### Running the simulation

```bash
python run_simulation.py
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--cycles` | `1500` | Number of simulation cycles to run |
| `--seed` | `None` | Random seed for reproducibility |
| `--output` | `output` | Prefix for all output files |

Examples:

```bash
# Standard run
python run_simulation.py --cycles 1500 --seed 42

# Longer run to observe full collapse spiral
python run_simulation.py --cycles 3000 --seed 42 --output results/long_run

# Vary parameters by editing initial_conditions in gov_inertia.py, then:
python run_simulation.py --cycles 2000 --seed 1 --output results/experiment_1
```

### Outputs

Each run produces four plots and a CSV:

| File | Contents |
|---|---|
| `<output>.csv` | Full history DataFrame, one row per cycle |
| `<output>_overview.png` | 4×3 panel grid: structure, political economy, failure mechanics, budget |
| `<output>_dynamics.png` | Phase portrait (D vs compliance) + revenue vs cost |
| `<output>_political_economy.png` | Legitimacy, Gini, and cumulative event counts |
| `<output>_collapse.png` | P_f over time with cumulative collapse counter |

Red vertical lines mark collapse events; green lines mark democratic reform events.

### Using the module directly

```python
import copy
import gov_inertia as gi

params = copy.deepcopy(gi.initial_conditions)

# Modify any parameter before running
params['growth_rate'] = 0.02
params['lambda_build'] = 0.005   # faster legitimacy accumulation
params['T_election'] = 25        # more frequent elections

history = gi.run_simulation(params, n_cycles=2000)

import pandas as pd
df = pd.DataFrame(history)
print(df[['N', 'G', 'Lambda_current', 'D', 'mean_c', 'P_f', 'gini', 'mu_0']].tail(10))
```

### Key tuning parameters

**Legitimacy dynamics**

| Parameter | Effect | Default |
|---|---|---|
| `lambda_build` | How fast legitimacy accumulates from effective governance | `0.002` |
| `lambda_erode` | Erosion rate per unit of divergence D | `0.01` |
| `lambda_corr` | Erosion from corruption per cycle | `0.02` |
| `Lambda` | Initial legitimacy | `0.5` |

**Public goods and inequality**

| Parameter | Effect | Default |
|---|---|---|
| `lambda_g` | Productivity multiplier on public goods spending | `0.05` |
| `phi_g` | Barro exponent — controls optimal government spending share | `0.3` |
| `gini_drift` | Natural per-cycle increase in inequality | `0.0001` |
| `phi_ineq` | Compliance penalty per unit of Gini | `0.2` |

**Laffer curve**

| Parameter | Effect | Default |
|---|---|---|
| `kappa_tau` | Tax suppression intensity; revenue peak at τ* = 1/kappa_tau | `1.0` |

**Democratic correction**

| Parameter | Effect | Default |
|---|---|---|
| `T_election` | Cycles between reform opportunities | `50` |
| `reform_strength` | Fraction of excess divergence removed per reform | `0.3` |
| `D_ref` | Target divergence level after reform | `0.05` |

**Collapse timing**

| Parameter | Effect | Default |
|---|---|---|
| `lambda_D` | How fast divergence D grows with H | `0.00245` |
| `D_crit` | Divergence at which P_f = 0.5 | `0.3` |
| `kappa_pf` | Steepness of the P_f sigmoid | `10.0` |

---

## Documentation

### Thesis

Government is modelled as an information-processing system that must monitor a fraction of all meaningful social interactions to maintain order. To keep pace with a growing population, the government must expand — adding agencies, deepening hierarchy, and spending tax revenue on enforcement. The model explores three competing forces:

1. **The case for government**: Public goods spending creates economic productivity that would not otherwise exist. A legitimate, well-functioning government generates broad compliance without surveillance.

2. **The scaling problem**: Bureaucratic depth grows superlinearly with the number of agencies. As the hierarchy deepens, government policy drifts from what the population actually wants — even without any malicious intent.

3. **The feedback trap**: Declining legitimacy forces governments to rely more on coercion. Tax pressure to fund coercion reduces compliance further. Reforms occur periodically but cannot permanently reverse structural divergence.

The simulation does not assume government inevitably collapses. With strong enough legitimacy-building, effective public goods provision, and timely democratic corrections, the system can remain stable. Collapse is a failure mode, not a foregone conclusion.

---

### Core Equations

#### Population interactions

Total meaningful interactions:

$$I = m \cdot k \cdot N$$

Government enforcement load (the fraction it must monitor):

$$I_E = F \cdot I$$

#### Government structure

`G` agencies require `H` layers of hierarchy to function:

$$H(G) = H_0 + \rho \cdot G^{\theta}$$

With θ = 3.5, H grows steeply with G. This is the central non-linearity: government cannot grow without becoming dramatically more layered.

#### Information capacity

$$B = \eta \cdot G \cdot H$$

The government is initialised with enough G to ensure B ≥ I_E.

#### Government latency

Internal friction of a large, deep bureaucracy:

$$L = \alpha \cdot N^{\beta} \cdot H^{\gamma}$$

#### Surveillance cost

Cost of monitoring the enforcement load:

$$C_{\text{surv}} = \varepsilon \cdot (k \cdot m)^{\delta} \cdot N^{\zeta} \cdot I_E + L$$

---

### Divergence

$$D = \min\!\left(1,\ D_0 + \lambda_D \cdot \ln(1+H)^{\gamma_D}\right)$$

As bureaucratic depth H grows, government policy necessarily drifts from population preferences — not through corruption or malice, but because large hierarchical systems move slowly and optimise for their own internal logic. With default parameters, D crosses the critical threshold of 0.3 around cycle 1050 when G ≈ 180.

---

### Legitimacy

Legitimacy Λ is a stock variable representing the degree to which the population consents to being governed — as distinct from merely complying under enforcement pressure. This follows the Weber–Easton model of political legitimacy.

$$\Delta\Lambda = \underbrace{\lambda_{\text{build}} \cdot \min\!\left(1, \frac{B}{I_E}\right) \cdot (1 - D)}_{\text{builds when effective and aligned}} - \underbrace{\lambda_{\text{erode}} \cdot D - \lambda_{\text{corr}} \cdot (1 - \omega)}_{\text{erodes with divergence and corruption}}$$

**Dynamics:** With default parameters, legitimacy builds from 0.5 to ~0.6 across the first 750 cycles as the government is functional and divergence is low. It then declines as D grows, reaching zero around cycle 1200. A government can arrest this decline by investing in public goods (reducing D's growth) or through democratic correction.

---

### Compliance

Compliance blends two components weighted by legitimacy:

$$\bar{c} = \underbrace{\Lambda \cdot (1 - D)}_{\text{voluntary}} + \underbrace{(1 - \Lambda) \cdot \max\!\left(0,\ 1 - D - \phi_\tau \cdot \max(0, \tau - \tau_{\text{ref}})\right)}_{\text{coerced}}$$

**The key distinction from a purely enforcement-based model:** a high-legitimacy government retains broad compliance even under tax pressure, because citizens trust the revenue is being used well. When legitimacy is zero, all compliance is coercion-based and erodes sharply with tax pressure — the government enters a fiscal trap where raising taxes destroys the base it is taxing.

Both components are further penalised by inequality:

$$\bar{c} \leftarrow \bar{c} \cdot \max(0,\ 1 - \phi_{\text{ineq}} \cdot \text{Gini})$$

#### Social enforcement cost

The per-cycle cost of a non-compliant population:

$$C_{\text{social}} = \phi_e \cdot N \cdot \left(D + (1 - \bar{c})\right)$$

`D` captures enforcement against laws the population regards as illegitimate. `(1 - c̄)` captures the cost of processing active violators.

Total governance cost:

$$C_{\text{total}} = C_{\text{surv}} + C_{\text{social}}$$

---

### Economics

#### Wealth and the Laffer curve

Expected aggregate wealth incorporates tax suppression of productive activity:

$$W(\tau) = N \cdot e^{\mu_0 + \frac{\sigma^2}{2}} \cdot e^{-\kappa_\tau \cdot \tau} \cdot \varepsilon_{\text{noise}}$$

Revenue is:

$$R = \tau \cdot W(\tau)$$

This peaks at τ* = 1/κ_τ — the Laffer maximum. Beyond this point, raising taxes shrinks the tax base faster than it raises revenue. When C_total cannot be covered even at τ*, the government is fiscally insolvent and must downsize.

This creates a **fiscal trap**: a government that has lost legitimacy must raise taxes to fund coercive enforcement; high taxes reduce productive activity and compliance; lower revenue requires even higher taxes. The trap is escapable only by rebuilding legitimacy or reducing costs through structural reform.

#### Public goods and the Barro multiplier

A fraction of each cycle's surplus is invested in public goods — infrastructure, rule of law, education. Following Barro (1990), this investment has a non-linear effect on economic productivity:

$$\Delta\mu_0 = \lambda_g \cdot \left(\frac{G_{\text{spend}}}{W}\right)^{\phi_g} \cdot \left(1 - \frac{G_{\text{spend}}}{W}\right)$$

The effect is maximised at a spending share of g* = φ_g / (1 + φ_g) ≈ 23% of GDP with default parameters. Too little investment fails to develop the economy; too much crowds out private activity. This gives government a genuine economic rationale: it is not merely an extraction mechanism but a producer of the conditions under which private wealth is possible.

Public goods investment also reduces the Gini coefficient, reflecting the distributional benefits of broadly shared infrastructure.

#### Inequality

The Gini coefficient drifts upward naturally (reflecting concentration dynamics in growing economies) and is partially offset by public goods investment:

$$\Delta\text{Gini} = \text{gini\_drift} - \text{gini\_redist} \cdot \frac{G_{\text{spend}}}{W}$$

High inequality reduces compliance through `phi_ineq`, independent of divergence or legitimacy. This captures the empirical finding (Acemoglu & Robinson) that unequal societies have lower institutional legitimacy and higher rates of defection.

---

### Democratic Correction

Every T_election cycles, if D > D_ref, a reform event occurs representing electoral accountability, constitutional revision, or administrative restructuring:

- D is partially reset: D ← D − reform_strength · (D − D_ref)
- G is trimmed by 3% (bureaucratic rationalisation)
- Legitimacy receives a +0.1 boost from the democratic process
- A one-cycle transition tax cost is applied

**What reforms can and cannot do:** With default parameters, 22 reforms occur across a 1500-cycle run. They slow the growth of D but cannot reverse the underlying structural dynamic — as long as H keeps growing, D will keep growing. Reforms buy time but do not substitute for the legitimacy-building that comes from effective governance and low divergence.

---

### Collapse

#### Failure probability

$$P_f = \sigma\!\left(\kappa \cdot (D - D_{\text{crit}})\right) = \frac{1}{1 + e^{-\kappa(D - D_{\text{crit}})}}$$

Crosses 0.5 when D reaches D_crit. With default parameters this occurs around cycle 1050.

#### Stochastic partial collapse

Each cycle when D > D_crit, a collapse event fires with probability P_f × 0.02. A random severity s ~ Uniform(0.1, 0.5) drives:

- G loss: G ← G × (1 − 0.2s)
- Legitimacy loss: Λ ← Λ − 0.3s
- Economic damage: μ₀ ← μ₀ − 0.5s

Collapse events are partial, not terminal. However, each collapse reduces capacity, which forces higher taxes, which further erodes compliance and legitimacy — a compounding spiral.

#### Fiscal insolvency

If C_total cannot be met even at the Laffer-peak tax rate, the government is fiscally insolvent. It is forced to shrink (G × 0.95) and takes a legitimacy penalty. This can occur when legitimacy has collapsed (all compliance is coerced) and divergence is very high.

---

### Surplus Allocation

Each cycle, post-corruption surplus is split randomly across four activities:

| Allocation | Effect |
|---|---|
| Government expansion | Increases G → increases H and B, but also accelerates D growth |
| Enforcement efficiency | Gradually reduces ε, δ, ζ |
| Public goods | Increases μ₀ via Barro curve; reduces Gini |
| Savings | Accumulated reserve |

Corruption drains (1 − ω) of gross surplus each cycle before allocation.

Note the tension in surplus allocation: spending on government expansion grows capacity but also deepens H, which grows D. Spending on public goods grows the economy and reduces inequality but also funds more government expansion through larger future surpluses. There is no allocation strategy that escapes the structural divergence problem indefinitely.

---

### Simulation Loop (per cycle)

```
1.  Compute I, I_E, B, H, L
2.  Compute surveillance cost C_surv = E_C × I_E + L
3.  Compute divergence D from current H
4.  Draw base wealth W_base (stochastic, once per cycle)
5.  Compute compliance c̄ and social cost C_social at current τ
6.  While R < C_total and τ < Laffer peak:
        raise τ by 0.0001; recompute W(τ), R, c̄, C_social
7.  If still R < C_total: handle fiscal insolvency
8.  Deduct corruption from surplus
9.  Spend surplus: expand gov, reduce costs, public goods, savings
10. Natural inequality drift (Gini += gini_drift)
11. Update legitimacy Λ based on capacity ratio and divergence
12. Compute failure probability P_f
13. Stochastic collapse check
14. Democratic correction check (every T_election cycles)
15. Grow population
16. Snapshot all state; reset τ to τ_ref
```

---

### Parameter Reference

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `N` | N | 1000 | Initial population |
| `growth_rate` | — | 0.015 | Per-cycle population growth rate |
| `m` | m | 0.01 | Fraction of interactions that are meaningful |
| `k` | k | 50 | Average interactions per person |
| `G` | G | 1 | Initial number of governing agencies |
| `H_0` | H₀ | 1 | Base hierarchical depth |
| `rho` | ρ | 1.5 | Hierarchy scaling coefficient |
| `theta` | θ | 3.5 | Hierarchy scaling exponent |
| `F` | F | 0.1 | Fraction of interactions to enforce |
| `eta` | η | 0.7 | Information throughput per agency per layer |
| `alpha` | α | 0.5 | Latency base coefficient |
| `beta` | β | 0.5 | Latency population exponent |
| `gamma` | γ | 0.5 | Latency hierarchy exponent |
| `epsilon` | ε | 10 | Base surveillance cost coefficient |
| `delta` | δ | 20 | Interaction complexity exponent |
| `zeta` | ζ | 0.1 | Population scaling in surveillance cost |
| `mu_0` | μ₀ | 0 | Initial log-mean of wealth distribution |
| `sigma` | σ | 0.1 | Log-std of wealth distribution |
| `tau` | τ | 0.05 | Initial tax rate (reset to `tau_ref` each cycle) |
| `kappa_tau` | κ_τ | 1.0 | Laffer suppression; revenue peaks at τ* = 1/κ_τ |
| `C_B` | C_B | 5 | Per-unit cost of government expansion |
| `omega` | ω | 0.95 | Fraction of surplus retained after corruption |
| `Lambda` | Λ | 0.5 | Initial legitimacy |
| `lambda_build` | λ_build | 0.002 | Legitimacy accumulation rate |
| `lambda_erode` | λ_erode | 0.01 | Legitimacy erosion per unit of D |
| `lambda_corr` | λ_corr | 0.02 | Legitimacy erosion from corruption |
| `phi_tau` | φ_τ | 0.5 | Tax resistance coefficient (coerced compliance only) |
| `tau_ref` | τ_ref | 0.01 | Equilibrium tax rate |
| `D_0` | D₀ | 0.001 | Base divergence |
| `lambda_D` | λ_D | 0.00245 | Divergence growth coefficient |
| `gamma_D` | γ_D | 1.70 | Divergence growth exponent |
| `phi_e` | φ_e | 0.002 | Per-capita social enforcement cost coefficient |
| `kappa_pf` | κ | 10.0 | Failure probability sigmoid steepness |
| `D_crit` | D_crit | 0.3 | Divergence at which P_f = 0.5 |
| `phi_g` | φ_g | 0.3 | Barro exponent for public goods productivity |
| `lambda_g` | λ_g | 0.05 | Public goods productivity multiplier |
| `gini` | — | 0.35 | Initial Gini coefficient |
| `gini_drift` | — | 0.0001 | Natural per-cycle increase in Gini |
| `gini_redist` | — | 0.002 | Gini reduction per unit of public goods share |
| `phi_ineq` | φ_ineq | 0.2 | Compliance penalty per unit of Gini |
| `T_election` | T_E | 50 | Cycles between democratic reform opportunities |
| `D_ref` | D_ref | 0.05 | Target divergence after democratic reform |
| `reform_strength` | — | 0.3 | Fraction of excess D removed per reform |
| `reform_cost_tau` | — | 0.005 | One-cycle tax increase during reform |

---

### Output DataFrame Columns

| Column | Description |
|---|---|
| `N` | Population |
| `G` | Governing agencies |
| `H` | Bureaucratic depth |
| `D` | Policy divergence |
| `Lambda_current` | Legitimacy (Λ) |
| `mean_c` | Mean compliance |
| `P_f` | Failure probability |
| `gini` | Gini coefficient |
| `mu_0` | Population wealth log-mean (grows via Barro public goods) |
| `C_surveillance` | Surveillance + latency cost |
| `C_social` | Social enforcement cost |
| `C_total_computed` | Total cost (C_surv + C_social) |
| `R` | Tax revenue collected |
| `W` | Productive wealth this cycle (after Laffer suppression) |
| `tau` | Tax rate used this cycle |
| `stolen_funds` | Cumulative corruption losses |
| `savings` | Cumulative saved surplus |
| `collapse` | 1 if a stochastic collapse event occurred this cycle |
| `reform` | 1 if a democratic reform event occurred this cycle |
| `insolvent` | 1 if fiscal insolvency was triggered this cycle |
| `collapse_events` | Cumulative collapse count |
| `reform_events` | Cumulative reform count |
| `insolvent_count` | Cumulative insolvency count |
| `epsilon`, `delta`, `zeta` | Enforcement cost parameters (decay via efficiency spending) |

---

### Theoretical Grounding

| Mechanism | Source |
|---|---|
| Legitimacy distinct from enforcement compliance | Weber (1922), Easton (1965), Lipset (1960) |
| Public goods productivity multiplier | Barro (1990) |
| Revenue-maximising tax rate (Laffer) | Standard public finance |
| Inequality and political instability | Acemoglu & Robinson (2006) |
| Democratic correction of policy divergence | Electoral accountability literature |
| Bureaucratic depth and coordination overhead | Williamson (1967), Parkinson (1958) |
