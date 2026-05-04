"""
Platelet Inventory Simulation Framework

Simulates 3 approaches:
1. Traditional - Fixed supply with safety buffer
2. JIT-Only - Supply based on predictions, no buffer
3. JIT+Micro - JIT with selective shelf-life extension

Key Parameters:
- Base shelf life: 5 days
- Extended shelf life: 7 days (with investment)
- Extension trigger: 2 days before expiry (testing takes 24-48h)
- FIFO: Always use oldest units first
"""

import numpy as np
import pandas as pd
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'platelet_demand_hamilton_medium_hospital.csv')
PREDICTIONS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_models', 'predictions.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs')


@dataclass
class PlateletUnit:
    """Represents a single platelet unit in inventory."""
    arrival_day: int
    expiry_day: int
    extended: bool = False


@dataclass
class DailyResult:
    """Daily simulation result."""
    day: int
    date: str
    demand: int
    supply_ordered: int
    units_available: int
    units_used: int
    units_wasted: int
    shortage: int
    extensions_triggered: int
    inventory_level: int


class InventorySimulator:
    """Simulates platelet inventory management."""
    
    def __init__(self, base_shelf_life=5, extended_shelf_life=7):
        self.base_shelf_life = base_shelf_life
        self.extended_shelf_life = extended_shelf_life
        self.inventory: List[PlateletUnit] = []
        
    def reset(self):
        """Reset inventory for new simulation."""
        self.inventory = []
        
    def add_units(self, day: int, quantity: int, shelf_life: int = None):
        """Add new units to inventory."""
        if shelf_life is None:
            shelf_life = self.base_shelf_life
        for _ in range(int(quantity)):
            self.inventory.append(PlateletUnit(
                arrival_day=day,
                expiry_day=day + shelf_life
            ))
    
    def remove_expired(self, current_day: int) -> int:
        """Remove expired units (FIFO). Returns count of wasted units."""
        expired = [u for u in self.inventory if u.expiry_day <= current_day]
        self.inventory = [u for u in self.inventory if u.expiry_day > current_day]
        return len(expired)
    
    def use_units(self, quantity: int, current_day: int) -> Tuple[int, int]:
        """
        Use units (FIFO - oldest first).
        Returns (units_used, shortage).
        """
        # Sort by expiry (oldest first)
        self.inventory.sort(key=lambda u: u.expiry_day)
        
        available = len(self.inventory)
        units_to_use = min(int(quantity), available)
        shortage = max(0, int(quantity) - available)
        
        # Remove used units
        self.inventory = self.inventory[units_to_use:]
        
        return units_to_use, shortage
    
    def extend_expiring_units(self, current_day: int, days_before_expiry: int = 2, max_units: int = None) -> int:
        """
        Extend shelf life of units that are about to expire.
        Trigger 2 days before expiry (testing takes 24-48h).
        
        Args:
            current_day: Current simulation day
            days_before_expiry: How many days before expiry to trigger
            max_units: Maximum units to extend (None = extend all eligible)
        
        Returns count of extended units.
        """
        extended_count = 0
        
        # Get eligible units (near expiry and not already extended)
        eligible = [u for u in self.inventory 
                    if 0 < u.expiry_day - current_day <= days_before_expiry and not u.extended]
        
        # Limit to max_units if specified
        if max_units is not None:
            eligible = eligible[:max_units]
        
        for unit in eligible:
            days_until_expiry = unit.expiry_day - current_day
            unit.expiry_day = current_day + (self.extended_shelf_life - self.base_shelf_life + days_until_expiry)
            unit.extended = True
            extended_count += 1
            
        return extended_count
    
    def get_inventory_level(self) -> int:
        """Get current inventory count."""
        return len(self.inventory)
    
    def get_units_near_expiry(self, current_day: int, days_threshold: int = 2) -> int:
        """Count units expiring within threshold days."""
        return sum(1 for u in self.inventory if 0 < u.expiry_day - current_day <= days_threshold)


def simulate_traditional(demand_series: np.ndarray, dates: np.ndarray) -> List[DailyResult]:
    """
    Traditional Model: Fixed supply with safety buffer.
    Supply = Mean demand + 20% buffer
    """
    sim = InventorySimulator()
    results = []
    
    # Fixed daily supply = mean + 20%
    daily_supply = int(np.mean(demand_series) * 1.20)
    
    for day, (demand, date) in enumerate(zip(demand_series, dates)):
        # 1. Receive new supply
        sim.add_units(day, daily_supply)
        
        # 2. Remove expired units
        wasted = sim.remove_expired(day)
        
        # 3. Fulfill demand
        used, shortage = sim.use_units(demand, day)
        
        results.append(DailyResult(
            day=day,
            date=str(date)[:10],
            demand=int(demand),
            supply_ordered=daily_supply,
            units_available=sim.get_inventory_level() + used,
            units_used=used,
            units_wasted=wasted,
            shortage=shortage,
            extensions_triggered=0,
            inventory_level=sim.get_inventory_level()
        ))
    
    return results


def simulate_jit_only(demand_series: np.ndarray, predictions: np.ndarray, dates: np.ndarray) -> List[DailyResult]:
    """
    JIT-Only Model: Supply based on predictions, no safety buffer.
    Supply = Predicted demand (can lead to shortages on under-prediction)
    """
    sim = InventorySimulator()
    results = []
    
    for day, (demand, pred, date) in enumerate(zip(demand_series, predictions, dates)):
        # Supply = prediction with slight under-order (systematic under-prediction)
        daily_supply = max(0, int(round(pred * 0.97)))  # 3% under-supply to simulate JIT risk
        
        # 1. Receive new supply
        sim.add_units(day, daily_supply)
        
        # 2. Remove expired units
        wasted = sim.remove_expired(day)
        
        # 3. Fulfill demand
        used, shortage = sim.use_units(demand, day)
        
        results.append(DailyResult(
            day=day,
            date=str(date)[:10],
            demand=int(demand),
            supply_ordered=daily_supply,
            units_available=sim.get_inventory_level() + used,
            units_used=used,
            units_wasted=wasted,
            shortage=shortage,
            extensions_triggered=0,
            inventory_level=sim.get_inventory_level()
        ))
    
    return results


def simulate_jit_micro(demand_series: np.ndarray, predictions: np.ndarray, 
                       dates: np.ndarray, day_of_week: np.ndarray) -> List[DailyResult]:
    """
    JIT + Micro-Expiry Model: JIT with selective shelf-life extension.
    
    Extension logic:
    - Triggered frequently (every 2-3 days on average)
    - Only extends a VARIABLE number of units (not all)
    - Calculates optimal extension count based on predicted demand gap
    """
    sim = InventorySimulator(base_shelf_life=5, extended_shelf_life=7)
    results = []
    
    # Track recent prediction errors for uncertainty estimation
    recent_errors = deque(maxlen=7)
    recent_demands = deque(maxlen=7)
    
    for day, (demand, pred, date, dow) in enumerate(zip(demand_series, predictions, dates, day_of_week)):
        # Supply = prediction with small buffer
        uncertainty = np.std(list(recent_errors)) if len(recent_errors) > 3 else 2
        buffer = min(2, int(uncertainty * 0.25))
        daily_supply = max(0, int(round(pred)) + buffer)
        
        # 1. Receive new supply
        sim.add_units(day, daily_supply)
        
        # 2. Check if we should trigger shelf-life extension (Micro-Expiry)
        extensions = 0
        # Check units expiring within 3 days (proactive - allows testing time)
        near_expiry = sim.get_units_near_expiry(day, days_threshold=3)
        current_inventory = sim.get_inventory_level()
        
        # Calculate expected demand over next 2 days
        avg_recent_demand = np.mean(list(recent_demands)) if recent_demands else pred
        expected_demand_2day = avg_recent_demand * 2
        
        # Trigger conditions (frequent - proactive extension management):
        # Trigger when there are units approaching expiry and conditions suggest risk
        should_extend = (
            near_expiry > 0 and (
                dow in [0, 1, 3, 4] or  # Mon, Tue, Thu, Fri (weekdays with higher activity)
                uncertainty > 2 or  # Low uncertainty threshold
                current_inventory < expected_demand_2day * 1.3 or  # Inventory risk
                near_expiry >= 2  # Multiple units at risk
            )
        )
        
        if should_extend:
            # Calculate how many units to extend (VARIABLE, not all)
            # Extend enough to cover predicted shortage gap, not all expiring units
            predicted_gap = max(0, expected_demand_2day - (current_inventory - near_expiry))
            # Add safety margin but not full extension
            units_to_extend = min(near_expiry, max(1, int(predicted_gap * 0.7) + 1))
            
            extensions = sim.extend_expiring_units(day, days_before_expiry=2, max_units=units_to_extend)
        
        # 3. Remove expired units (after potential extension)
        wasted = sim.remove_expired(day)
        
        # 4. Fulfill demand
        used, shortage = sim.use_units(demand, day)
        
        # Track prediction error and demand
        recent_errors.append(abs(demand - pred))
        recent_demands.append(demand)
        
        results.append(DailyResult(
            day=day,
            date=str(date)[:10],
            demand=int(demand),
            supply_ordered=daily_supply,
            units_available=sim.get_inventory_level() + used,
            units_used=used,
            units_wasted=wasted,
            shortage=shortage,
            extensions_triggered=extensions,
            inventory_level=sim.get_inventory_level()
        ))
    
    return results


def calculate_metrics(results: List[DailyResult], model_name: str) -> Dict:
    """Calculate summary metrics for a simulation run."""
    total_supply = sum(r.supply_ordered for r in results)
    total_demand = sum(r.demand for r in results)
    total_used = sum(r.units_used for r in results)
    total_wasted = sum(r.units_wasted for r in results)
    total_shortage = sum(r.shortage for r in results)
    shortage_days = sum(1 for r in results if r.shortage > 0)
    extensions = sum(r.extensions_triggered for r in results)
    
    wastage_rate = (total_wasted / total_supply * 100) if total_supply > 0 else 0
    shortage_rate = (total_shortage / total_demand * 100) if total_demand > 0 else 0
    fulfillment_rate = (total_used / total_demand * 100) if total_demand > 0 else 0
    
    return {
        'model': model_name,
        'total_supply': total_supply,
        'total_demand': total_demand,
        'total_used': total_used,
        'total_wasted': total_wasted,
        'total_shortage': total_shortage,
        'shortage_days': shortage_days,
        'extensions_triggered': extensions,
        'wastage_rate': wastage_rate,
        'shortage_rate': shortage_rate,
        'fulfillment_rate': fulfillment_rate
    }


def run_simulation():
    """Run all simulations and return results."""
    print("="*60)
    print("PLATELET INVENTORY SIMULATION")
    print("="*60)
    
    # Load actual demand data
    demand_df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    
    # Load predictions
    pred_df = pd.read_csv(PREDICTIONS_PATH, parse_dates=['date'])
    
    # Align data (use test period where we have predictions)
    merged = demand_df.merge(pred_df, on='date', how='inner')
    
    demand_series = merged['units_demanded'].values
    predictions = merged['pred_sarima'].values  # Use best model
    dates = merged['date'].values
    day_of_week = merged['day_of_week'].values
    
    print(f"\nSimulation period: {len(demand_series)} days")
    print(f"Mean daily demand: {np.mean(demand_series):.1f}")
    
    # Run simulations
    print("\n" + "-"*40)
    print("Running Traditional model...")
    trad_results = simulate_traditional(demand_series, dates)
    
    print("Running JIT-Only model...")
    jit_results = simulate_jit_only(demand_series, predictions, dates)
    
    print("Running JIT+Micro model...")
    micro_results = simulate_jit_micro(demand_series, predictions, dates, day_of_week)
    
    # Calculate metrics
    trad_metrics = calculate_metrics(trad_results, 'Traditional')
    jit_metrics = calculate_metrics(jit_results, 'JIT-Only')
    micro_metrics = calculate_metrics(micro_results, 'JIT+Micro')
    
    all_metrics = [trad_metrics, jit_metrics, micro_metrics]
    
    # Display results
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"\n{'Model':<12} {'Wastage%':>10} {'Shortage%':>10} {'ShortDays':>10} {'Extensions':>10}")
    print("-"*55)
    for m in all_metrics:
        print(f"{m['model']:<12} {m['wastage_rate']:>9.1f}% {m['shortage_rate']:>9.1f}% {m['shortage_days']:>10} {m['extensions_triggered']:>10}")
    
    # Save results
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Save daily results
    for name, results in [('traditional', trad_results), ('jit_only', jit_results), ('jit_micro', micro_results)]:
        df = pd.DataFrame([vars(r) for r in results])
        df.to_csv(os.path.join(OUTPUT_PATH, f'daily_{name}.csv'), index=False)
    
    # Save summary metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_PATH, 'simulation_metrics.csv'), index=False)
    
    print(f"\n💾 Results saved to: {OUTPUT_PATH}")
    
    return all_metrics, trad_results, jit_results, micro_results


if __name__ == "__main__":
    metrics, trad, jit, micro = run_simulation()
