import pulp
import pandas as pd

# Complete Storage Configuration
STORAGE_CONFIG = {
    # Storage level thresholds (in GJ)
    'thresholds': {
        'low': 500000,    # First threshold for rate changes
        'high': 750000,   # Second threshold for rate changes
        'max': 1000000    # Maximum storage capacity
    },
    
    # Injection rates for different storage levels (GJ/day)
    'injection_rates': {
        'below_low': 7500,      # Rate when storage < low threshold
        'mid_range': 5000,      # Rate when low threshold <= storage < high threshold
        'above_high': 2500      # Rate when storage >= high threshold
    },
    
    # Withdrawal rates for different storage levels (GJ/day)
    'withdrawal_rates': {
        'above_high': 7500,     # Rate when storage > high threshold
        'mid_range': 5000,      # Rate when high threshold >= storage > low threshold
        'below_low': 2500       # Rate when storage <= low threshold
    },
    
    # Monthly schedule configuration
    'schedule': {
        # Format: 'Month-Year': (days_in_month, price_per_gj)
        'Apr-2025': (30, 2.0),
        'May-2025': (31, 1.8),
        'Jun-2025': (30, 1.9),
        'Jul-2025': (31, 2.0),
        'Aug-2025': (31, 2.1),
        'Sep-2025': (30, 2.2),
        'Oct-2025': (31, 2.5),
        'Nov-2025': (30, 2.6),
        'Dec-2025': (31, 2.7),
        'Jan-2026': (31, 2.8),
        'Feb-2026': (28, 2.9),
        'Mar-2026': (31, 2.8)
    },
    
    # Seasonal configuration
    'seasons': {
        # List of months where injection is allowed
        'injection_months': ['Apr-2025', 'May-2025', 'Jun-2025', 
                           'Jul-2025', 'Aug-2025', 'Sep-2025'],
        # List of months where withdrawal is allowed
        'withdrawal_months': ['Oct-2025', 'Nov-2025', 'Dec-2025', 
                            'Jan-2026', 'Feb-2026', 'Mar-2026'],
        'require_full_cycle': True  # Whether to require complete fill and empty cycle
    }
}

def optimize_storage(config=STORAGE_CONFIG):
    """
    Optimize natural gas storage operations based on provided configuration.
    
    Args:
        config (dict): Configuration dictionary containing all parameters for optimization
                      Including storage thresholds, rates, schedule, and seasonal constraints
    
    Returns:
        pandas.DataFrame: Results containing daily rates, storage levels, and profits
    """
    # Initialize the optimization model
    model = pulp.LpProblem("NatGasStorage", pulp.LpMaximize)
    
    # Extract schedule and seasonal information from config
    schedule = config['schedule']
    months = list(schedule.keys())
    injection_months = config['seasons']['injection_months']
    withdrawal_months = config['seasons']['withdrawal_months']
    
    # Validate that all months are properly categorized
    all_months = set(months)
    injection_set = set(injection_months)
    withdrawal_set = set(withdrawal_months)
    if not (injection_set | withdrawal_set) == all_months:
        raise ValueError("All months must be categorized as either injection or withdrawal months")
    if injection_set & withdrawal_set:
        raise ValueError("A month cannot be both an injection and withdrawal month")
    
    # Extract storage threshold values
    low_threshold = config['thresholds']['low']
    high_threshold = config['thresholds']['high']
    max_storage = config['thresholds']['max']
    
    # Determine maximum rates
    max_injection_rate = max(config['injection_rates'].values())
    max_withdrawal_rate = max(config['withdrawal_rates'].values())
    
    # Initialize decision variables
    rates = {}       # Daily injection/withdrawal rates
    storage = {}     # Storage levels at end of each month
    
    # Binary variables to track storage level ranges
    in_range_low = {}    # 0 to low_threshold
    in_range_mid = {}    # low_threshold to high_threshold
    in_range_high = {}   # high_threshold to max_storage
    
    # Create variables for each month
    for m in months:
        # Rate variables: positive for injection, negative for withdrawal
        if m in injection_months:
            rates[m] = pulp.LpVariable(f"rate_{m}", 0, max_injection_rate)
        else:
            rates[m] = pulp.LpVariable(f"rate_{m}", -max_withdrawal_rate, 0)
        
        # Storage level variables
        storage[m] = pulp.LpVariable(f"storage_{m}", 0, max_storage)
        
        # Binary variables for storage ranges
        in_range_low[m] = pulp.LpVariable(f"range1_{m}", 0, 1, pulp.LpBinary)
        in_range_mid[m] = pulp.LpVariable(f"range2_{m}", 0, 1, pulp.LpBinary)
        in_range_high[m] = pulp.LpVariable(f"range3_{m}", 0, 1, pulp.LpBinary)
    
    # Objective function: Maximize profit
    # Negative rate (withdrawal) * price = revenue
    # Positive rate (injection) * price = cost
    model += pulp.lpSum(-rates[m] * schedule[m][1] * schedule[m][0] for m in months)
    
    # Storage evolution constraints
    # Initial storage calculation
    model += storage[months[0]] == rates[months[0]] * schedule[months[0]][0]
    
    # Storage evolution for subsequent months
    for i in range(1, len(months)):
        prev_month = months[i-1]
        curr_month = months[i]
        model += storage[curr_month] == storage[prev_month] + rates[curr_month] * schedule[curr_month][0]
    
    # Storage cycle constraints (if required)
    if config['seasons']['require_full_cycle']:
        # Must reach max capacity by end of injection season
        model += storage[injection_months[-1]] == max_storage
        # Must empty by end of cycle
        model += storage[months[-1]] == 0
    
    # Storage range constraints for each month
    for m in months:
        # Ensure exactly one range is active
        model += in_range_low[m] + in_range_mid[m] + in_range_high[m] == 1
        
        # Define storage range boundaries
        # Low range: 0 to low_threshold
        model += storage[m] <= low_threshold + max_storage * (1 - in_range_low[m])
        model += storage[m] >= 0 * in_range_low[m]
        
        # Mid range: low_threshold to high_threshold
        model += storage[m] <= high_threshold + max_storage * (1 - in_range_mid[m])
        model += storage[m] >= low_threshold * in_range_mid[m]
        
        # High range: high_threshold to max_storage
        model += storage[m] <= max_storage + max_storage * (1 - in_range_high[m])
        model += storage[m] >= high_threshold * in_range_high[m]
        
        # Rate constraints based on storage levels
        if m in injection_months:
            # Injection rates decrease as storage fills
            model += rates[m] <= (
                config['injection_rates']['below_low'] * in_range_low[m] +
                config['injection_rates']['mid_range'] * in_range_mid[m] +
                config['injection_rates']['above_high'] * in_range_high[m]
            )
        else:
            # Withdrawal rates decrease as storage empties
            model += rates[m] >= -(
                config['withdrawal_rates']['below_low'] * in_range_low[m] +
                config['withdrawal_rates']['mid_range'] * in_range_mid[m] +
                config['withdrawal_rates']['above_high'] * in_range_high[m]
            )
    
    # Enforce seasonal constraints
    for m in withdrawal_months:
        model += rates[m] <= 0  # Only withdrawal allowed
    for m in injection_months:
        model += rates[m] >= 0  # Only injection allowed
    
    # Solve the optimization model
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = model.solve(solver)
    
    if status != 1:
        print("Warning: Optimal solution not found!")
    
    # Compile results
    results = []
    for m in months:
        results.append({
            'Month': m,
            'Daily_Rate': rates[m].value(),
            'Days': schedule[m][0],
            'Price': schedule[m][1],
            'Storage_Level': storage[m].value(),
            'Monthly_Profit': -rates[m].value() * schedule[m][1] * schedule[m][0]
        })
    
    return pd.DataFrame(results)

# Example of how to use custom configuration with different seasonal definitions
custom_config = STORAGE_CONFIG.copy()
custom_config = {
    # Keep the original storage thresholds
    'thresholds': {
        'low': 500000,
        'high': 750000,
        'max': 1000000
    },
    
    # Keep the original injection rates
    'injection_rates': {
        'below_low': 7500,
        'mid_range': 5000,
        'above_high': 2500
    },
    
    # Keep the original withdrawal rates
    'withdrawal_rates': {
        'above_high': 7500,
        'mid_range': 5000,
        'below_low': 2500
    },
    
    # Modified schedule with custom prices
    'schedule': {
        'Apr-2025': (30, 1.5),  # Lower summer prices
        'May-2025': (31, 1.4),
        'Jun-2025': (30, 1.6),
        'Jul-2025': (31, 1.8),
        'Aug-2025': (31, 2.0),
        'Sep-2025': (30, 2.2),
        'Oct-2025': (31, 3.0),  # Higher winter prices
        'Nov-2025': (30, 3.2),
        'Dec-2025': (31, 3.5),
        'Jan-2026': (31, 3.8),
        'Feb-2026': (28, 4.0),
        'Mar-2026': (31, 3.7)
    },
    
    # Modified seasonal configuration
    'seasons': {
        'injection_months': ['Apr-2025', 'May-2025', 'Jun-2025', 'Jul-2025'],  # Changed injection period
        'withdrawal_months': ['Aug-2025', 'Sep-2025', 'Oct-2025', 'Nov-2025', 
                            'Dec-2025', 'Jan-2026', 'Feb-2026', 'Mar-2026'],
        'require_full_cycle': True
    }
}

# Run optimization with custom configuration
results = optimize_storage(custom_config)
print("\nOptimization Results with Custom Configuration:")
print(results[['Month', 'Daily_Rate', 'Days', 'Price', 'Storage_Level', 'Monthly_Profit']].to_string(index=False))
print(f"\nTotal Profit: ${results['Monthly_Profit'].sum():,.2f}")

# You can also run with default configuration:
# results = optimize_storage()

