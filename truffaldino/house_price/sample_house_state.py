"""
Module for sampling the environmental state for a house negotiation scenario.

This module simulates the underlying economic reality of a house sale negotiation,
providing the ground truth parameters from which agent observations (potentially
noisy or incomplete) can be derived.

The sampling follows a structured process:
1. Sample global market conditions.
2. Simulate a recent price history for comparable houses based on market conditions.
3. Sample specifics of the house being negotiated.
4. Sample idiosyncratic parameters for the seller.
5. Sample idiosyncratic parameters for the buyer.
6. Calculate derived values like outside options and reservation prices based on
   the sampled parameters.

Key Random Variables Sampled:
----------------------------

Market Conditions (`sample_market_conditions`):
  - g: Long-run real price trend (annual rate). N(0.04, 0.02^2). Drives the
       underlying direction of the simulated price process.
  - sigma: Short-run volatility of comparable house prices (annualized).
           HalfNormal(scale=0.08). Determines the noise level in the price process. 
  - r: Risk-free interest rate / discount rate (annual). N(0.05, 0.01^2).
       Used for time value of money calculations (e.g., discounting, option value).
  - lambda_m: Arrival rate of new comparable listings (houses/week).
              Gamma(k=3, θ=0.4). Affects the buyer's outside option.
  - lambda_b: Arrival rate of new credible buyers making offers (buyers/week).
              Gamma(k=2, θ=0.5). Affects the seller's outside option.
  * Note: Annual rates (g, sigma, r) and weekly rates (lambda_m, lambda_b) are
    converted to daily rates internally for simulation and calculation consistency.

House Specifics (`sample_house_specifics`):
  - q: Quality premium/discount of this specific house relative to the current
       market price (P_today) on a log scale. N(0, sigma_q^2) (e.g., sigma_q=0.07).
       Determines the seller's initial intrinsic value (V_h = exp(q) * P_today).

Seller Idiosyncrasies (`sample_seller_idiosyncratics`):
  - C_s: Seller's daily carrying cost (mortgage, utilities, stress, etc.).
         N(mean, std^2) (e.g., $250, $50^2). Increases the seller's reservation
         price over time.
  - k_s: Seller's capital constraint penalty if the deal isn't closed by the
         deadline (lump sum penalty). Exponential(scale) (e.g., scale=$10,000).
         Creates a sharp increase in reservation price after the deadline.
  - D: Seller's soft deadline (days from start). Poisson(mean) (e.g., mean=30).
       The day after which the capital constraint penalty k_s applies.
  - delta_s: Seller's liquidity preference for this specific house relative to the
             current market price (P_today) on a log scale. N(0, sigma_delta_s^2)
             (e.g., sigma_delta_s=0.06). Determines the seller's effective keep value
             (V_h_eff = V_h - delta_s).

Buyer Idiosyncrasies (`sample_buyer_idiosyncratics`):
  - delta: Buyer's fit premium/discount for this specific house relative to the
           current market price (P_today) on a log scale. N(0, sigma_delta^2)
           (e.g., sigma_delta=0.08). Determines the buyer's initial intrinsic
           valuation (V_b = exp(delta) * P_today).
  - C_b: Buyer's daily cost of waiting (rent, storage, hassle, etc.).
         N(mean, std^2) (e.g., $120, $30^2). Affects the buyer's outside option
         calculation (value decreases over longer search horizons).

Derived Values (Calculated, not directly sampled):
-------------------------------------------------
  - P_history: Simulated path of recent comparable market prices using a
               mean-reverting process (Ornstein-Uhlenbeck) based on g, sigma, kappa.
  - P_today: The final price from the simulated history, representing the current
             market price estimate.
  - V_h: Seller's intrinsic value for the house (P_today adjusted by quality q).
  - V_b: Buyer's intrinsic value for the house (P_today adjusted by fit delta).
  - OP_s: Seller's option premium from waiting for potentially better offers.
          Calculated based on P_today, offer volatility (sigma_offer), buyer arrival
          rate (lambda_b), and discount rate (r).
  - P_s(t): Seller's time-dependent reservation price. Calculated as
            V_h + OP_s + C_s*t + deadline_penalty(k_s, D, t).
  - V_b_outside: Buyer's estimated value of their best alternative (searching for
                 another house). Calculated based on P_today, listing arrival rate
                 (lambda_m), comparable house volatility (sigma_comp, assumed sigma),
                 waiting costs (C_b), discount rate (r), and search horizon (T_days).

Revealed values:
-------------------------------------------------
BUYER:
  - N_hist_buyer: Number of historical prices made available to the buyer :: Buyer's private info
  - delta_verbal: Narrativized fit premium (imprecise) :: Buyer's private info
  - C_b: Buyer's daily cost of waiting :: Buyer's private info
  - Bedroom count :: common knowledge
  - Unit type :: common knowledge
  - Suburb :: common knowledge

SELLER:
  - N_hist_seller: Number of historical prices made available to the seller :: Seller's private info
  - q_verbal: Narrativized quality premium (imprecise) :: Seller's private info
  - D_verbal: Narrativized soft deadline (imprecise) :: Seller's private info
  - C_s: Carrying cost :: Seller's private info
  - k_s: Capital constraint penalty :: Seller's private info
  - Bedroom count :: common knowledge
  - Unit type :: common knowledge
  - Suburb :: common knowledge

AGENT:
  - N_hist_agent: Number of historical prices made available to the agent (typically higher than N_hist_buyer and N_hist_seller) :: Agent's private info
  - sigma_comp: Volatility of comparable houses :: Agent's private info
  - lambda_m: Arrival rate of new comparable listings :: Agent's private info
  - lambda_b: Arrival rate of new credible buyers making offers :: Agent's private info
  - r: Risk-free interest rate / discount rate :: Agent's private info
  - Whatever the principle wants to reveal

GAME MASTER:
  - P_s(t): Seller's time-dependent reservation price, used to calculate payoffs
  - V_b_outside: Buyer's outside option, used to calculate payoffs
"""
import numpy as np
from scipy.stats import halfnorm, gamma, poisson, norm
from scipy.optimize import minimize_scalar
import random
from typing import Literal
from truffaldino.house_price.reveal_house_state import generate_narrative_contexts, generate_full_context

BuyerRole = Literal["Owner-Occupier", "Investor"]
SellerRole = Literal["Owner-Occupier", "Investor"]

SUBURB_PRICES = {
    "Brooklyn, New York, USA": 900_000,
    "Bondi, Sydney, Australia": 1_600_000,
    "Hackney, London, UK": 1_000_000,
    "Scarborough, Toronto, Canada": 660_000,
    "Ponsonby, Auckland, New Zealand": 1_100_000,
    "Bandra West, Mumbai, India": 600_000,
    "Eixample, Barcelona, Spain": 650_000,
    "Charlottenburg, Berlin, Germany": 700_000,
    "La Condesa, Mexico City, Mexico": 530_000,
    "Miraflores, Lima, Peru": 350_000,
    "Parkhurst, Johannesburg, South Africa": 300_000,
    "Gangnam, Seoul, South Korea": 950_000,
    "Bukit Timah, Singapore": 2_200_000,
    "Fitzroy, Melbourne, Australia": 900_000,
    "West End, Brisbane, Australia": 700_000,
    "Kichijoji, Tokyo, Japan": 800_000,
    "Södermalm, Stockholm, Sweden": 750_000,
    "Uccle, Brussels, Belgium": 650_000,
    "Vila Madalena, São Paulo, Brazil": 450_000,
    "Vinohrady, Prague, Czech Republic": 550_000,
    "Ruzafa, Valencia, Spain": 400_000,
    "Zamalek, Cairo, Egypt": 300_000,
    "Recoleta, Buenos Aires, Argentina": 350_000,
    "Taman Tun Dr Ismail, Kuala Lumpur, Malaysia": 380_000,
    "Kebayoran Baru, Jakarta, Indonesia": 320_000,
    "Parnell, Auckland, New Zealand": 1_000_000,
    "Oak Bay, Victoria, Canada": 985_000,
    "Clontarf, Dublin, Ireland": 820_000,
    "Lapa, Lisbon, Portugal": 650_000,
    "Santurce, San Juan, Puerto Rico": 400_000,
    "Coconut Grove, Miami, USA": 1_100_000,
    "Monteverde, Rome, Italy": 600_000,
    "Naperville, Illinois, USA": 550_000,
    "Bellevue, Washington, USA": 950_000,
    "Plano, Texas, USA": 450_000,
    "La Jolla, San Diego, USA": 1_400_000,
    "Altona, Hamburg, Germany": 650_000,
    "Deira, Dubai, UAE": 420_000,
    "Al Sadd, Doha, Qatar": 500_000,
    "Bahçelievler, Ankara, Turkey": 250_000,
    "Jericho, Oxford, UK": 850_000,
    "Shimokitazawa, Tokyo, Japan": 700_000,
    "Taikoo Shing, Hong Kong": 1_300_000,
    "New Farm, Brisbane, Australia": 750_000,
    "Mount Lawley, Perth, Australia": 600_000,
    "Kallio, Helsinki, Finland": 450_000,
    "Ipanema, Rio de Janeiro, Brazil": 800_000,
    "Seapoint, Cape Town, South Africa": 450_000,
    "Hellerup, Copenhagen, Denmark": 900_000,
    "Arlington, Virginia, USA": 800_000
}

DEFAULT_SIGMA_Q = 0.07 # Quality premium doesn't scale directly with price, but we want magnitude to be similar
DEFAULT_SIGMA_DELTA = 0.08 # Delta doesn't scale directly with price, but we want magnitude to be similar
DEFAULT_SIGMA_OFFER = 0.05
DEFAULT_KAPPA = 1.0
DEFAULT_SELLER_CARRY_COST_MEAN = 0.05/365
DEFAULT_SELLER_CARRY_COST_STD = 0.2
DEFAULT_SIGMA_SELLER_DELTA = 0.06 # Added: Seller preference for liquidity/not owning
DEFAULT_SELLER_CONSTRAINT_SCALE = 10000
DEFAULT_SELLER_DEADLINE_MEAN = 30
DEFAULT_BUYER_WAIT_COST_MEAN = 120
DEFAULT_BUYER_WAIT_COST_STD = 30
DEFAULT_BUYER_HORIZON_DAYS = 60
DEFAULT_DT_DAYS = 1
DEFAULT_SIGMA_IDIOSYNCRATIC_LOG = 0.03 # Std dev for log-price idiosyncratic noise
R_DAILY_REF_FOR_PATIENCE = 0.05/365 # Reference daily interest rate for patience calc

BEDROOM_DELTA_MULT = {
    -2: 0.77,  # studio / tiny 0-bed
    -1: 0.88,  # 1-bed
     0: 1.00,  # 2-bed (baseline)
     1: 1.14,  # 3-bed
     2: 1.30,  # 4-bed
     3: 1.48,  # 5-bed
     4: 1.68,  # 6-bed
}

UNIT_MULT = 0.8

def sample_market_conditions(rng: np.random.Generator, buyer_role: BuyerRole):
    """Samples global market state variables, adjusting based on buyer role."""
    g = rng.normal(0.04, 0.02)
    sigma = np.abs(rng.normal(loc=0.0, scale=0.08))
    r = rng.normal(0.05, 0.01)
    r = max(r, 0.001)
    # Convert annual rates to daily rates for simulation
    r_daily = (1 + r)**(1/365) - 1
    g_daily = (1 + g)**(1/365) - 1
    sigma_daily = sigma / np.sqrt(365)

    # Arrival rates (events per week)
    lambda_m_weekly = rng.gamma(shape=3, scale=0.4) # New listings
    lambda_b_weekly = rng.gamma(shape=2, scale=0.5) # New buyers

    # Adjust lambda_m based on buyer role
    if buyer_role == "Investor":
        lambda_m_weekly *= 5

    # Convert weekly rates to daily rates
    lambda_m_daily = lambda_m_weekly / 7
    lambda_b_daily = lambda_b_weekly / 7

    return {
        "g_annual": g,
        "sigma_annual": sigma,
        "r_annual": r,
        "lambda_m_weekly": lambda_m_weekly,
        "lambda_b_weekly": lambda_b_weekly,
        "g_daily": g_daily,
        "sigma_daily": sigma_daily,
        "r_daily": r_daily,
        "lambda_m_daily": lambda_m_daily,
        "lambda_b_daily": lambda_b_daily,
    }

def simulate_price_process(
    P0: float,
    g_daily: float,
    sigma_daily: float,
    kappa_annual: float = DEFAULT_KAPPA,
    T_days: int = 90,
    dt_days: int = DEFAULT_DT_DAYS,
    rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    """Simulates the mean-reverting log price process (Ornstein-Uhlenbeck)."""
    num_steps = T_days // dt_days
    log_P = np.zeros(num_steps + 1)
    log_P[0] = np.log(P0)
    log_P_bar_0 = log_P[0]

    # Convert annual kappa to daily
    kappa_daily = 1 - (1 - kappa_annual)**(dt_days/365) # Approximation for small dt/365

    dW = rng.normal(0, np.sqrt(dt_days), num_steps)

    for t_idx in range(num_steps):
        current_time_days = (t_idx + 1) * dt_days
        log_P_bar_t = log_P_bar_0 + g_daily * current_time_days
        d_log_P = kappa_daily * (log_P_bar_t - log_P[t_idx]) * dt_days + sigma_daily * dW[t_idx]
        log_P_ou = log_P[t_idx] + d_log_P

        # Add idiosyncratic noise to the log price
        idiosyncratic_noise = rng.normal(0, DEFAULT_SIGMA_IDIOSYNCRATIC_LOG)
        log_P[t_idx+1] = log_P_ou + idiosyncratic_noise

    return np.exp(log_P)


def sample_house_specifics(P_today: float, sigma_q: float = DEFAULT_SIGMA_Q, rng: np.random.Generator = np.random.default_rng()):
    """Samples house-specific quality premium and calculates initial value."""
    q = rng.normal(0, sigma_q)
    V_h = q + P_today
    return {"q": q, "V_h": V_h}

def sample_seller_idiosyncratics(rng: np.random.Generator, seller_role: SellerRole):
    """Samples seller-specific costs, constraints, deadlines, and liquidity preference, adjusting based on seller role."""
    # Sample C_s_frac based on seller role
    if seller_role == "Owner-Occupier":
        # Strictly negative carrying cost
        abs_cost_frac = abs(rng.normal(DEFAULT_SELLER_CARRY_COST_MEAN, DEFAULT_SELLER_CARRY_COST_STD * DEFAULT_SELLER_CARRY_COST_MEAN))
        C_s_frac = -abs_cost_frac
    elif seller_role == "Investor":
        # Generally positive carrying cost (use original positive mean)
        C_s_frac = rng.normal(DEFAULT_SELLER_CARRY_COST_MEAN, DEFAULT_SELLER_CARRY_COST_STD * DEFAULT_SELLER_CARRY_COST_MEAN)
    else: # Should not happen with Literal type hinting
        raise ValueError(f"Invalid seller_role: {seller_role}")

    # Removed max(0, C_s_frac) - Allow negative C_s (net income from holding)
    k_s = rng.exponential(scale=DEFAULT_SELLER_CONSTRAINT_SCALE) # Capital constraint penalty
    D = rng.poisson(lam=DEFAULT_SELLER_DEADLINE_MEAN) # Soft deadline (days)
    # Sample seller delta (liquidity preference) - Ensure strictly positive
    delta_s_frac = -abs(rng.normal(0, DEFAULT_SIGMA_SELLER_DELTA))
    # delta_s is calculated later based on P_today
    return {"C_s_frac": C_s_frac, "k_s": k_s, "D": D, "delta_s_frac": delta_s_frac}

def sample_buyer_idiosyncratics(
    P_today: float,
    buyer_role: BuyerRole,
    lambda_m_daily: float,
    r_daily: float,
    sigma_delta: float = DEFAULT_SIGMA_DELTA,
    rng: np.random.Generator = np.random.default_rng()
) -> dict:
    """
    Samples buyer-specific costs, dynamic search horizon, fit premium based on simulated search,
    and calculates initial valuation.
    """
    # 1. Sample Buyer's daily cost of waiting
    C_b = rng.normal(DEFAULT_BUYER_WAIT_COST_MEAN, DEFAULT_BUYER_WAIT_COST_STD)
    C_b = max(0, C_b) # Ensure non-negative

    # 2. Calculate dynamic search horizon (patience)
    cost_patience_factor = DEFAULT_BUYER_WAIT_COST_MEAN / (C_b + 1e-6) if C_b > 0 else 2.0
    rate_patience_factor = R_DAILY_REF_FOR_PATIENCE / (r_daily + 1e-9) if r_daily > 0 else 2.0
    
    patience_multiplier = np.sqrt(cost_patience_factor * rate_patience_factor)
    patience_multiplier = np.clip(patience_multiplier, 0.5, 3.0) 
    T_search_days = int(DEFAULT_BUYER_HORIZON_DAYS * patience_multiplier)
    T_search_days = max(1, T_search_days) 

    # 3. Simulate number of houses seen during search horizon
    expected_houses_seen = lambda_m_daily * T_search_days
    num_houses_seen = rng.poisson(max(0, expected_houses_seen)) # lam must be >= 0

    # 4. Sample fit premium (delta)
    sigma_delta_eff = sigma_delta * P_today # Scale std dev of delta by P_today
    
    # Adjust sigma_delta_eff variance based on buyer role
    if buyer_role == "Owner-Occupier":
        sigma_delta_val_dist = sigma_delta_eff * np.sqrt(2)
    elif buyer_role == "Investor":
        sigma_delta_val_dist = sigma_delta_eff / np.sqrt(2)
    else: # Should not happen
        raise ValueError(f"Invalid buyer_role: {buyer_role}")
    sigma_delta_val_dist = max(1e-6, sigma_delta_val_dist) # Ensure positive std dev

    if num_houses_seen > 0:
        potential_deltas = abs(rng.normal(0, sigma_delta_val_dist, num_houses_seen))
        delta = np.max(potential_deltas) if potential_deltas.size > 0 else 0.0
    else:
        # Fallback: if no houses seen, sample one delta (with role-adjusted sigma)
        delta = abs(rng.normal(0, sigma_delta_val_dist))
    
    delta = max(0, delta) # Ensure delta is non-negative

    # 5. Calculate Buyer's initial valuation
    V_b = P_today + delta

    return {"delta": delta, "C_b": C_b, "V_b": V_b, "T_search_days_buyer": T_search_days, "num_houses_seen_buyer": num_houses_seen}

def calculate_seller_option_premium(
    P_today: float,
    sigma_offer: float,
    lambda_b_daily: float,
    r_daily: float,
    C_s: float, # Seller's daily carrying cost
    k_s: float, # Seller's capital constraint penalty
    D: int,     # Seller's soft deadline (days)
    t: int,     # Days since start of negotiation
) -> float:
    """
    Calculates the seller's perpetual option premium approximation,
    adjusted for expected carrying costs and deadline penalties.
    """
    # Ensure inputs are valid for the formula
    if P_today <= 0 or sigma_offer < 0 or lambda_b_daily <= 0 or r_daily < 0:
        return 0.0

    discount_rate_sum = r_daily + lambda_b_daily
    # Handle potential division by zero or very small numbers
    if discount_rate_sum <= 1e-9:
        raise ValueError("Offer rate is too small")

    term1 = sigma_offer * P_today / np.sqrt(2 * np.pi)
    term2 = np.sqrt(lambda_b_daily / discount_rate_sum)
    gross_OP_s = term1 * term2

    pv_carrying_cost = C_s * 1 / lambda_b_daily

    # 3. Calculate PV of Expected Future Deadline Penalty
    # PV = k_s * P(No sale by D) * DiscountFactor(D)
    pv_deadline_penalty = k_s * np.exp(-lambda_b_daily * (D-t))

    # 4. Calculate Net Option Premium
    net_OP_s = gross_OP_s - pv_carrying_cost - pv_deadline_penalty

    return net_OP_s


def calculate_seller_reservation_price(
    V_h_eff: float, # Seller's effective keep value (V_h - delta_s)
    option_value_adjustment: float, # Adjustment relative to V_h_eff
    # C_s, t, k_s, D are now implicitly included in option_value_adjustment
) -> float:
    """Calculates the seller's reservation price based on effective value and option adjustment."""
    # Reservation price = Effective Keep Value + Option Adjustment
    P_s_t = V_h_eff + option_value_adjustment
    return P_s_t

def calculate_buyer_outside_value(
    S_today: float,
    lambda_m_daily: float,
    sigma_scaled: float,
    sigma_idiosyncratic: float,
    C_b: float,
    r_daily: float,
    t: int,
    sigma_delta_scaled: float,
) -> float:
    """
        Calculates the buyer's expected value from searching for T_days.
        E_max: Expected maximum surplus of a house from searching for T_days.
    """

    T_days = DEFAULT_BUYER_HORIZON_DAYS - t

    sigma_surplus = np.sqrt(sigma_scaled**2 + sigma_delta_scaled**2 + sigma_idiosyncratic**2)
    if T_days <= 0 or lambda_m_daily <= 0:
        return 0

    expected_arrivals = lambda_m_daily * T_days
    # Argument for Phi^-1: 1 - 1 / (lambda_m*T + 1)
    phi_inv_arg = 1.0 - (1.0 / (expected_arrivals + 1.0))
    # Handle edge cases where arg is very close to 1 or 0
    if phi_inv_arg >= 1.0 - 1e-9:
            z_score = norm.ppf(min(phi_inv_arg, 0.999999))
    elif phi_inv_arg <= 1e-9:
            z_score = norm.ppf(1e-9)
    else:
            z_score = norm.ppf(phi_inv_arg)

    E_max = sigma_surplus * z_score

    discount_factor = np.exp(-r_daily * T_days)
    V_b_T = discount_factor * E_max - C_b * T_days
    return V_b_T

# --- Monte Carlo Seller Expected Value for Optimization ---

def calculate_expected_value(
    P_accept: float, # Candidate acceptance threshold
    P_today: float,
    V_h_eff: float, # Seller's effective keep value (V_h - delta_s)
    g_daily: float,
    sigma_daily: float,
    r_daily: float,
    lambda_b_daily: float,
    C_s: float, # Absolute daily cost
    k_s: float,
    D: int,
    sigma_offer: float,
    kappa_annual: float,
    dt_days: int,
    num_simulations: int,
    simulation_horizon_days: int, # How long to simulate prices/arrivals
    rng: np.random.Generator
) -> float:
    """
    Calculates the expected discounted net proceeds for a given static acceptance threshold P_accept.
    Used as the objective function for optimization.
    """
    total_value = 0.0
    # Use V_h_eff for the value if no sale occurs
    value_if_no_sale = (V_h_eff - C_s * D - k_s) * np.exp(-r_daily * D)

    for _ in range(num_simulations):
        # 1. Simulate market price path
        # Simulate longer to potentially capture offers after D
        price_path = simulate_price_process(
            P0=P_today,
            g_daily=g_daily,
            sigma_daily=sigma_daily,
            kappa_annual=kappa_annual,
            T_days=simulation_horizon_days,
            dt_days=dt_days,
            rng=rng
        )

        def get_price_at_day(t_day):
             if t_day <= 0: return P_today
             # Ensure index is within bounds of the simulated path
             idx = min(len(price_path) - 1, int(np.ceil(t_day / dt_days)))
             return price_path[idx]

        # 2. Simulate buyer arrivals over the full horizon
        mean_arrivals = lambda_b_daily * simulation_horizon_days
        if mean_arrivals < 0: mean_arrivals = 0
        num_arrivals = rng.poisson(mean_arrivals)

        if num_arrivals == 0:
            chain_value = value_if_no_sale
        else:
            # 3. Simulate arrival times and offers
            arrival_times = rng.uniform(0, simulation_horizon_days, num_arrivals)
            offers_data = [] # Store tuples of (arrival_time, offer_value)
            for t_i in arrival_times:
                P_ti = get_price_at_day(t_i)
                offer_mean = P_ti
                offer_std = sigma_offer * P_ti
                offer = rng.normal(offer_mean, max(0, offer_std))
                offers_data.append((t_i, offer))

            # Sort offers by arrival time
            offers_data.sort(key=lambda x: x[0])

            # 4. Apply decision rule
            accepted_offer_proceeds = None
            first_offer_after_D_proceeds = None

            for t_i, offer_i in offers_data:
                # Check offers before or at deadline D
                if t_i <= D:
                    if offer_i >= P_accept:
                        # First acceptable offer before deadline
                        discount_factor_ti = np.exp(-r_daily * t_i)
                        accepted_offer_proceeds = (offer_i - C_s * t_i) * discount_factor_ti
                        break # Stop checking once accepted
                # Track first offer after deadline (if needed)
                elif first_offer_after_D_proceeds is None:
                    discount_factor_ti = np.exp(-r_daily * t_i)
                    first_offer_after_D_proceeds = (offer_i - C_s * t_i) * discount_factor_ti
                    # Don't break here, need to check all offers <= D first

            # Determine chain value
            if accepted_offer_proceeds is not None:
                chain_value = accepted_offer_proceeds
            elif first_offer_after_D_proceeds is not None:
                # No acceptable offer before D, take first offer after D
                chain_value = first_offer_after_D_proceeds
            else:
                # No offers before D met threshold, and no offers arrived after D
                chain_value = value_if_no_sale

        total_value += chain_value

    return total_value / num_simulations


def optimize_seller_threshold_and_premium(
    P_today: float,
    V_h_eff: float, # Seller's effective keep value (V_h - delta_s)
    g_daily: float,
    sigma_daily: float,
    r_daily: float,
    lambda_b_daily: float,
    C_s: float, # Absolute daily cost
    k_s: float,
    D: int,
    sigma_offer: float,
    kappa_annual: float = DEFAULT_KAPPA,
    dt_days: int = DEFAULT_DT_DAYS,
    num_simulations: int = 100, # MC simulations per evaluation
    optimization_horizon_days: int = 90, # How far out to simulate for optimization
    rng: np.random.Generator = np.random.default_rng()
) -> float:
    """
    Finds the optimal static acceptance threshold P_accept via numerical optimization
    and returns the corresponding seller option value adjustment (max_expected_value - V_h_eff).
    This adjustment uses the effective value V_h_eff as the baseline.
    """

    simulation_horizon = D + optimization_horizon_days # Ensure simulation covers period after D

    # Objective function for the optimizer (minimize negative expected value)
    def objective_func(P_accept):
        return -calculate_expected_value(
            P_accept,
            P_today, V_h_eff, g_daily, sigma_daily, r_daily, lambda_b_daily,
            C_s, k_s, D, sigma_offer, kappa_annual, dt_days,
            num_simulations, simulation_horizon, rng
        )

    # Define reasonable bounds for the acceptance threshold search
    # Lower bound: Effective value (won't sell for less than this base)
    # Upper bound: Could be V_h_eff + large penalty, or related to P_today/sigma_offer
    lower_bound = V_h_eff
    upper_bound = P_today * (1 + 5 * sigma_offer) # Heuristic: 5 std dev above market
    # Ensure lower bound is not higher than upper bound
    if lower_bound >= upper_bound: upper_bound = lower_bound + P_today * 0.1 # Add a small margin

    # Perform optimization
    result = minimize_scalar(
        objective_func,
        bounds=(lower_bound, upper_bound),
        method='bounded' # Use bounded optimization
    )

    if not result.success:
        # Handle optimization failure, e.g., return 0 premium or raise error
        print(f"Warning: Seller threshold optimization failed: {result.message}")
        # Fallback: maybe calculate value at a default threshold like V_h_eff?
        max_expected_value = -objective_func(V_h_eff) # Value at lower bound
    else:
        # Recalculate expected value at the optimum for potentially higher accuracy
        # (or just use -result.fun)
        max_expected_value = -result.fun
        # print(f"Optimal static threshold P_accept: {optimal_P_accept:.2f}") # Optional logging

    # Calculate option value adjustment relative to V_h_eff
    option_value_adjustment = max_expected_value - V_h_eff
    return option_value_adjustment


# --- Main Sampling Function ---

def sample_negotiation_state(
    P0: float,
    suburb: str,
    is_unit: bool,
    bedrooms: int,
    buyer_role: BuyerRole,
    seller_role: SellerRole,
    rng_seed: int | None = None
):
    """Samples the complete state for a house negotiation game, incorporating roles."""
    rng = np.random.default_rng(rng_seed)

    # 1. Market (depends on buyer role)
    market_params = sample_market_conditions(rng, buyer_role)

    # 2. Simulate recent price path
    price_history = simulate_price_process(
        P0=P0,
        g_daily=market_params["g_daily"],
        sigma_daily=market_params["sigma_daily"],
        kappa_annual=DEFAULT_KAPPA, # Or sample kappa too?
        T_days=90, # Standard history length
        dt_days=DEFAULT_DT_DAYS,
        rng=rng
    )
    P_today = price_history[-1]

    # 3. House
    house_params = sample_house_specifics(P_today, rng=rng)
    V_h = house_params["V_h"]

    # 4. Seller (depends on seller role)
    seller_params = sample_seller_idiosyncratics(rng, seller_role)
    # Calculate C_s (can be negative)
    seller_params["C_s"] = seller_params["C_s_frac"] * P_today
    C_s_abs = seller_params["C_s"] # Use this absolute value below
    # Calculate delta_s (can be negative)
    seller_params["delta_s"] = seller_params["delta_s_frac"] * P_today
    delta_s_abs = seller_params["delta_s"]

    # Calculate seller's effective value
    V_h_eff = V_h - delta_s_abs

    # 5. Buyer (depends on buyer role)
    buyer_params = sample_buyer_idiosyncratics(
        P_today,
        buyer_role,
        market_params["lambda_m_daily"],
        market_params["r_daily"],
        rng=rng
    )
    V_b = buyer_params["V_b"]

    # 6. Compute outside options (using daily rates)
    # Use Optimization + Monte Carlo simulation for seller option value adjustment
    option_value_adjustment = optimize_seller_threshold_and_premium(
        P_today=P_today,
        V_h_eff=V_h_eff, # Pass effective value
        g_daily=market_params["g_daily"],
        sigma_daily=market_params["sigma_daily"],
        r_daily=market_params["r_daily"],
        lambda_b_daily=market_params["lambda_b_daily"],
        C_s=C_s_abs, # Pass absolute seller cost
        k_s=seller_params["k_s"],
        D=seller_params["D"],
        sigma_offer=DEFAULT_SIGMA_OFFER,
        kappa_annual=DEFAULT_KAPPA,
        dt_days=DEFAULT_DT_DAYS,
        num_simulations=100, # Inner loop sims (can adjust)
        optimization_horizon_days=90, # How far past D to simulate
        rng=rng
    )

    # Calculate reservation prices for 4 days using the calculated adjustment
    # P_s(t) is constant based on the initial calculation, representing the static threshold strategy value
    P_s_ts = []
    # P_s(t) = V_h_eff + option_value_adjustment(0)
    P_s_0 = calculate_seller_reservation_price(
        V_h_eff=V_h_eff,
        option_value_adjustment=option_value_adjustment
    )

    # Seller payoff declines for a constant price; represent this with an increasing reservation price
    for t in range(4):
         P_s_ts.append(P_s_0 *(2 -  np.exp(-market_params["r_daily"] * t)) + max(0, C_s_abs * t))
    
    V_b_outsides = [calculate_buyer_outside_value(
        S_today=0, # Expected value of buying a random house in the reference class is 0
        lambda_m_daily=market_params["lambda_m_daily"],
        sigma_scaled=market_params["sigma_daily"] * P_today,
        sigma_idiosyncratic=DEFAULT_SIGMA_IDIOSYNCRATIC_LOG * P_today,
        C_b=buyer_params["C_b"],
        r_daily=market_params["r_daily"],
        t=t,
        sigma_delta_scaled=DEFAULT_SIGMA_DELTA * P_today
    ) for t in range(4)]
    
    # Calculate Seller's total cost per day (using absolute C_s which can be negative)
    seller_cost_per_day = C_s_abs + market_params["r_daily"] * V_h

    state = {
        "market": market_params,
        "price_history": price_history,
        "P_today": P_today,
        "house": house_params, # Contains q, V_h
        "seller": seller_params, # Contains C_s, k_s, D
        "buyer": buyer_params, # Contains delta, C_b, V_b
        "house_details": {
            "suburb": suburb,
            "unit": is_unit,
            "bedrooms": bedrooms,
        },
        "derived": {
            "seller_option_value_adjustment": option_value_adjustment, # Renamed
            "seller_reservation_price": P_s_ts,
            "buyer_outside_values": V_b_outsides,
            "seller_cost_per_day": seller_cost_per_day,
        }
    }

    # Add roles to the state dictionary AFTER sampling for narrative generation
    # (Narrative generation expects them in the state dictionary)
    state['buyer_role'] = buyer_role
    state['seller_role'] = seller_role

    return state

if __name__ == '__main__':
    # Example Usage
    # Roles are now defined locally

    # Randomly assign roles BEFORE sampling
    buyer_role: BuyerRole = random.choice(["Owner-Occupier", "Investor"]) # Uses local BuyerRole
    seller_role: SellerRole = random.choice(["Owner-Occupier", "Investor"]) # Uses local SellerRole
    print(f"Sampling with Roles: Buyer='{buyer_role}', Seller='{seller_role}'")

    suburb = random.choice(list(SUBURB_PRICES.keys()))
    is_unit = random.choice([True, False]) # Renamed 'unit' to 'is_unit' for clarity
    initial_median_price = SUBURB_PRICES[suburb]
    bedrooms = 0 # Default value
    if is_unit:
        bedrooms = random.randint(-2, 1) # Unit bedrooms: studio to 3-bed (use delta keys)
        initial_median_price *= UNIT_MULT
        if bedrooms in BEDROOM_DELTA_MULT:
             initial_median_price *= BEDROOM_DELTA_MULT[bedrooms]
    else:
        bedrooms = random.randint(0, 4) # House bedrooms: 2-bed to 6-bed (use delta keys)
        if bedrooms in BEDROOM_DELTA_MULT:
            initial_median_price *= BEDROOM_DELTA_MULT[bedrooms]

    # Pass roles to the sampling function
    negotiation_state = sample_negotiation_state(
        initial_median_price,
        suburb,
        is_unit,
        bedrooms,
        buyer_role=buyer_role, # Pass buyer role
        seller_role=seller_role, # Pass seller role
        rng_seed=None
    )

    # Add house details to the state dictionary AFTER sampling
    negotiation_state["house_details"] = {
        "suburb": suburb,
        "unit": is_unit, # Use the renamed variable
        "bedrooms": bedrooms + 2, # Store actual bedroom count (adjusting from delta keys)
    }

    print("Sampled Negotiation State (Numerical):")
    import json
    def default_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Add handling for numpy floats/ints if they cause issues
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    print(json.dumps(negotiation_state, indent=2, default=default_serializer))

    # --- Generate Narrative Contexts ---
    print("\n--- Generating Narrative Contexts ---")
    # Roles are already assigned above
    print(f"Assigned Roles: Buyer='{buyer_role}', Seller='{seller_role}'")


    narrative_contexts = generate_full_context(negotiation_state)
    print("\n--- Generated Contexts ---")
    print("\nBuyer Context:\n", narrative_contexts["buyer_context"])
    print("\nSeller Context:\n", narrative_contexts["seller_context"])
    print("\nAgent Context:\n", narrative_contexts["agent_context"])
    # --- End Narrative Context Generation ---


    # Calculate initial surplus (at t=0, assuming P = P_today for illustration)
    # Buyer surplus: V_b - P
    # Seller surplus: P - P_s(0)
    # Total surplus: V_b - P_s(0)
    P_s_0 = negotiation_state["derived"]["seller_reservation_price"][0] # Get P_s at t=0
    initial_surplus = negotiation_state["buyer"]["V_b"] - P_s_0
    print(f"Initial Potential Surplus (V_b - P_s(0)): ${initial_surplus:,.2f}")

    # ZOPA (Zone of Possible Agreement) at t=0: [P_s(0), V_b]
    # Note: This uses V_b (intrinsic value), not V_b_outside. Buyer won't pay more than V_b.
    # Seller won't accept less than P_s(0).
    zopa_low = P_s_0
    zopa_high = negotiation_state["buyer"]["V_b"]
    if zopa_high >= zopa_low:
        print(f"Initial ZOPA (t=0): [${zopa_low:,.2f}, ${zopa_high:,.2f}]")
    else:
        print(f"Initial ZOPA (t=0): Does not exist (V_b < P_s(0)) [P_s(0)={P_s_0:.2f}, V_b={zopa_high:.2f}]") # Added details

    # Buyer's BATNA (Best Alternative To Negotiated Agreement) is conceptually V_b_outside
    # Seller's BATNA is conceptually P_s(t) -> P_s(0) at t=0
    V_b_outside_0 = negotiation_state['derived']['buyer_outside_values'][0] # Get outside value at t=0
    print(f"Buyer's BATNA (Outside Value at t=0): ${V_b_outside_0:,.2f}")
    print(f"Seller's BATNA (Reservation Price at t=0): ${P_s_0:,.2f}") 
    print(f"Seller's Total Cost per Day: ${negotiation_state['derived']['seller_cost_per_day']:,.2f}")

    # Example: Calculate seller reservation price at day 10
    # Note: The reservation price is now calculated based on V_h_eff + option_value_adjustment
    # and is constant in the current implementation for t=0..3
    P_s_0 = negotiation_state["derived"]["seller_reservation_price"][0]
    print(f"\nSeller's reservation price (t=0..3): ${P_s_0:,.2f}")
    # P_s_10 would require recalculating the option value adjustment with t=10 as start
    # which is not done here. For illustration, P_s(0) is the key value.