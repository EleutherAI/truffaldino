import numpy as np
from truffaldino.house_price.sample_house_state import sample_negotiation_state
from truffaldino.scenarios import HousePriceScenario

def test_market_price_payoffs():
    """
    Test that payoffs at market price are reasonable:
    - Seller payoff should be close to 0 (within 10% of market price)
    - Buyer payoff should be positive
    """
    n_samples = 50
    seller_payoffs = []
    buyer_payoffs = []
    market_prices = []

    for _ in range(n_samples):
        # Sample a random state
        scenario = HousePriceScenario(seed=np.random.randint(0, 10000))
        state = scenario.negotiation_state
        
        # Get market price and calculate payoffs
        market_price = state['P_today']
        market_prices.append(market_price)
        
        # Create an offer at market price
        offer = {"price": market_price}
        
        # Calculate payoffs
        seller_payoff, buyer_payoff = scenario.get_payoffs(offer, steps=0)
        seller_payoffs.append(seller_payoff)
        buyer_payoffs.append(buyer_payoff)

    # Convert to numpy arrays for analysis
    seller_payoffs = np.array(seller_payoffs)
    buyer_payoffs = np.array(buyer_payoffs)
    market_prices = np.array(market_prices)

    # Calculate statistics
    seller_payoff_ratio = seller_payoffs / market_prices
    mean_seller_ratio = np.mean(seller_payoff_ratio)
    mean_buyer_payoff = np.mean(buyer_payoffs)
    
    # Print results
    print(f"\nTest Results (n={n_samples}):")
    print(f"Mean seller payoff ratio (|payoff|/market_price): {mean_seller_ratio:.3f}")
    print(f"Mean seller payoff: ${seller_payoffs.mean():,.2f}")
    print(f"Max seller payoff: ${seller_payoffs.max():,.2f}")
    print(f"Seller payoff std dev: ${np.std(seller_payoffs):,.2f}")
    print(f"Mean buyer payoff: ${mean_buyer_payoff:,.2f}")
    print(f"Seller payoff ratio std dev: {np.std(seller_payoff_ratio):.3f}")
    print(f"Buyer payoff std dev: ${np.std(buyer_payoffs):,.2f}")

    
    # Assertions
    assert mean_seller_ratio < 0.1 and mean_seller_ratio > -0.1, f"Seller payoff ratio {mean_seller_ratio:.3f} exceeds 10% of market price"
    assert mean_buyer_payoff > 0, f"Mean buyer payoff ${mean_buyer_payoff:,.2f} is not positive"
    assert np.all(buyer_payoffs > -market_prices), "Buyer payoffs should not be more negative than the market price"

if __name__ == "__main__":
    test_market_price_payoffs() 