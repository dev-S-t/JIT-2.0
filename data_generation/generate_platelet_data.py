"""
Synthetic Platelet Demand Dataset Generator

Generates realistic platelet demand data for 2 years across 2 hospital scenarios:
- Hamilton (Medium Hospital): Mean 17.9, SD 7.05
- Stanford (Large Hospital): Mean 35.4, SD 9.2

Incorporates temporal patterns:
- Weekday vs Weekend effects
- Friday spike (prophylactic transfusions)
- Tuesday spike (post-weekend surgeries)
- Holiday dips with rebound
- December/January seasonal slump
- Random trauma events
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============== CONFIGURATION ==============

# Hospital Scenarios
SCENARIOS = {
    "Hamilton_Medium_Hospital": {
        "mean_demand": 17.9,
        "std_demand": 7.05,
        "description": "Medium/Large Hospital - Hamilton, Ontario"
    },
    "Stanford_Large_Hospital": {
        "mean_demand": 35.4,
        "std_demand": 9.2,
        "description": "Large Tertiary Care - Stanford, USA"
    }
}

# Date range: 2 years (2024-2025)
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)

# Temporal multipliers based on research data
WEEKDAY_MULTIPLIERS = {
    0: 1.10,  # Monday - slightly elevated (catch-up)
    1: 1.15,  # Tuesday - spike (post-weekend surgeries)
    2: 1.05,  # Wednesday - normal
    3: 1.05,  # Thursday - normal
    4: 1.20,  # Friday - highest (prophylactic transfusions)
    5: 0.70,  # Saturday - low
    6: 0.70,  # Sunday - low
}

# US Holidays (simplified - major ones)
HOLIDAYS_2024 = [
    datetime(2024, 1, 1),   # New Year's Day
    datetime(2024, 1, 15),  # MLK Day
    datetime(2024, 2, 19),  # Presidents Day
    datetime(2024, 5, 27),  # Memorial Day
    datetime(2024, 7, 4),   # Independence Day
    datetime(2024, 9, 2),   # Labor Day
    datetime(2024, 11, 28), # Thanksgiving
    datetime(2024, 11, 29), # Day after Thanksgiving
    datetime(2024, 12, 24), # Christmas Eve
    datetime(2024, 12, 25), # Christmas
    datetime(2024, 12, 31), # New Year's Eve
]

HOLIDAYS_2025 = [
    datetime(2025, 1, 1),   # New Year's Day
    datetime(2025, 1, 20),  # MLK Day
    datetime(2025, 2, 17),  # Presidents Day
    datetime(2025, 5, 26),  # Memorial Day
    datetime(2025, 7, 4),   # Independence Day
    datetime(2025, 9, 1),   # Labor Day
    datetime(2025, 11, 27), # Thanksgiving
    datetime(2025, 11, 28), # Day after Thanksgiving
    datetime(2025, 12, 24), # Christmas Eve
    datetime(2025, 12, 25), # Christmas
    datetime(2025, 12, 31), # New Year's Eve
]

ALL_HOLIDAYS = set(HOLIDAYS_2024 + HOLIDAYS_2025)

# Seasonal multipliers by month
MONTH_MULTIPLIERS = {
    1: 0.90,   # January - holiday/weather slump
    2: 0.95,   # February - slight recovery
    3: 1.00,   # March - normal
    4: 1.00,   # April - normal
    5: 1.02,   # May - slight increase
    6: 1.05,   # June - summer surgeries
    7: 1.08,   # July - peak (year-end fiscal, organ transplants)
    8: 1.05,   # August - elevated
    9: 1.02,   # September - back to school season (less elective)
    10: 1.00,  # October - normal
    11: 0.98,  # November - pre-holiday slowdown
    12: 0.88,  # December - holiday slump
}

# ============== HELPER FUNCTIONS ==============

def calculate_negative_binomial_params(mean, std):
    """
    Calculate n and p parameters for negative binomial distribution
    from mean and standard deviation.
    
    For negative binomial:
    - mean = n * (1-p) / p
    - var = n * (1-p) / p^2
    """
    var = std ** 2
    # Solving for p and n
    p = mean / var
    n = mean * p / (1 - p)
    return n, p


def generate_demand_for_scenario(scenario_name, scenario_config, start_date, end_date):
    """Generate daily demand data for a single hospital scenario."""
    
    mean = scenario_config["mean_demand"]
    std = scenario_config["std_demand"]
    
    # Calculate negative binomial parameters
    n, p = calculate_negative_binomial_params(mean, std)
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    
    for date in dates:
        # Base demand from negative binomial
        base_demand = stats.nbinom.rvs(n, p)
        
        # Apply weekday multiplier
        weekday = date.weekday()
        weekday_mult = WEEKDAY_MULTIPLIERS[weekday]
        
        # Apply seasonal/monthly multiplier
        month_mult = MONTH_MULTIPLIERS[date.month]
        
        # Check if holiday
        is_holiday = date.date() in [h.date() for h in ALL_HOLIDAYS]
        holiday_mult = 0.80 if is_holiday else 1.0
        
        # Check if day after holiday (rebound effect)
        yesterday = date - timedelta(days=1)
        is_rebound = yesterday.date() in [h.date() for h in ALL_HOLIDAYS]
        rebound_mult = 1.10 if is_rebound else 1.0
        
        # Random trauma event (5% chance, +30-50% spike)
        trauma_event = random.random() < 0.05
        trauma_mult = random.uniform(1.30, 1.50) if trauma_event else 1.0
        
        # Calculate final demand
        final_demand = base_demand * weekday_mult * month_mult * holiday_mult * rebound_mult * trauma_mult
        final_demand = max(0, int(round(final_demand)))  # Ensure non-negative integer
        
        # Generate timestamp (demand typically happens during working hours 6am-10pm)
        # Peak demand hours: 8am-6pm
        hour = np.random.choice(
            range(6, 23),
            p=[0.02, 0.03, 0.08, 0.10, 0.10, 0.10, 0.10, 0.10, 0.08, 0.06, 
               0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02]
        )
        minute = random.randint(0, 59)
        timestamp = datetime(date.year, date.month, date.day, hour, minute)
        
        data.append({
            "timestamp": timestamp,
            "date": date.date(),
            "units_demanded": final_demand,
            "day_of_week": weekday,
            "day_name": date.strftime("%A"),
            "month": date.month,
            "month_name": date.strftime("%B"),
            "is_weekend": weekday >= 5,
            "is_holiday": is_holiday,
            "is_trauma_event": trauma_event,
            "scenario": scenario_name
        })
    
    return pd.DataFrame(data)


def validate_dataset(df, scenario_name, expected_mean):
    """Validate the generated dataset against expected patterns."""
    print(f"\n{'='*60}")
    print(f"VALIDATION: {scenario_name}")
    print(f"{'='*60}")
    
    # Overall statistics
    actual_mean = df["units_demanded"].mean()
    actual_std = df["units_demanded"].std()
    print(f"\n📊 Overall Statistics:")
    print(f"   Expected Mean: {expected_mean:.1f}")
    print(f"   Actual Mean:   {actual_mean:.1f}")
    print(f"   Actual Std:    {actual_std:.1f}")
    print(f"   Min:           {df['units_demanded'].min()}")
    print(f"   Max:           {df['units_demanded'].max()}")
    
    # Weekday vs Weekend
    weekday_mean = df[~df["is_weekend"]]["units_demanded"].mean()
    weekend_mean = df[df["is_weekend"]]["units_demanded"].mean()
    ratio = weekday_mean / weekend_mean if weekend_mean > 0 else 0
    print(f"\n📅 Weekday vs Weekend:")
    print(f"   Weekday Mean: {weekday_mean:.1f}")
    print(f"   Weekend Mean: {weekend_mean:.1f}")
    print(f"   Ratio:        {ratio:.2f}x (expected ~1.5-1.7x)")
    
    # Day of week breakdown
    print(f"\n📆 Daily Breakdown:")
    for day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        day_mean = df[df["day_name"] == day_name]["units_demanded"].mean()
        print(f"   {day_name:12s}: {day_mean:.1f}")
    
    # Monthly breakdown
    print(f"\n📈 Monthly Breakdown:")
    for month in range(1, 13):
        month_mean = df[df["month"] == month]["units_demanded"].mean()
        month_name = datetime(2024, month, 1).strftime("%B")
        print(f"   {month_name:12s}: {month_mean:.1f}")
    
    # Holiday analysis
    holiday_mean = df[df["is_holiday"]]["units_demanded"].mean()
    non_holiday_mean = df[~df["is_holiday"]]["units_demanded"].mean()
    print(f"\n🎄 Holiday Effect:")
    print(f"   Holiday Mean:     {holiday_mean:.1f}")
    print(f"   Non-Holiday Mean: {non_holiday_mean:.1f}")
    
    # Trauma events
    trauma_count = df["is_trauma_event"].sum()
    trauma_pct = (trauma_count / len(df)) * 100
    print(f"\n🚑 Trauma Events:")
    print(f"   Count: {trauma_count} ({trauma_pct:.1f}%)")
    
    return True


def main():
    """Main function to generate all datasets."""
    print("=" * 60)
    print("SYNTHETIC PLATELET DEMAND DATASET GENERATOR")
    print("=" * 60)
    print(f"\nGenerating data from {START_DATE.date()} to {END_DATE.date()}")
    print(f"Total days: {(END_DATE - START_DATE).days + 1}")
    
    all_data = []
    
    for scenario_name, scenario_config in SCENARIOS.items():
        print(f"\n🏥 Generating: {scenario_config['description']}")
        
        df = generate_demand_for_scenario(
            scenario_name, 
            scenario_config, 
            START_DATE, 
            END_DATE
        )
        
        # Validate
        validate_dataset(df, scenario_name, scenario_config["mean_demand"])
        
        # Save individual scenario
        filename = f"platelet_demand_{scenario_name.lower()}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Saved: {filename}")
        
        all_data.append(df)
    
    # Combine all scenarios
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv("platelet_demand_all_scenarios.csv", index=False)
    print(f"\n💾 Saved: platelet_demand_all_scenarios.csv")
    
    print("\n" + "=" * 60)
    print("✅ DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. platelet_demand_hamilton_medium_hospital.csv")
    print("  2. platelet_demand_stanford_large_hospital.csv")
    print("  3. platelet_demand_all_scenarios.csv")
    
    return combined_df


if __name__ == "__main__":
    df = main()
