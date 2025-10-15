"""
CORRECT Data Parser for Rust (1987) based on zurcher-data repository.

From https://github.com/OpenSourceEconomics/zurcher-data

Data structure (from their code):
- Column 1: Bus_ID
- Column 2-3: Month/Year purchased
- Column 4-5: Month/Year of 1st replacement
- Column 6: Odo_1 (odometer reading at 1st replacement)
- Column 7-8: Month/Year of 2nd replacement  
- Column 9: Odo_2 (odometer reading at 2nd replacement)
- Column 10-11: Month/Year begin
- Columns 12+: Monthly odometer readings

Note: Odo_1 and Odo_2 tell us WHEN replacements occurred!
When we cross these thresholds, we subtract them to get "mileage since last replacement".

Author: Pierre-Luc Bacon
Course: IFT6162
"""

import numpy as np
import os


def load_rust_group(filepath, n_buses, n_periods, binsize=5000):
    """
    Load one group following zurcher-data approach.
    
    This correctly handles:
    - Odo_1, Odo_2: odometer readings at 1st and 2nd replacement
    - Subtracts these to get "mileage since last replacement"
    - Marks replacement periods correctly
    
    Parameters
    ----------
    filepath : str
        Path to .asc file
    n_buses : int
        Number of rows in matrix
    n_periods : int
        Number of columns in matrix
    binsize : int
        Mileage bin size (default 5000)
        
    Returns
    -------
    states : np.ndarray
        State observations (mileage bins)
    decisions : np.ndarray
        Replacement decisions (0 or 1)
    """
    # Read file
    with open(filepath, 'r') as f:
        file_cols = f.read().split('\n')
    file_cols = [float(x.strip()) for x in file_cols if x.strip()]
    
    # Reshape: data is column-stacked
    # Column j has rows [j*n_buses : (j+1)*n_buses]
    matrix = np.zeros((n_buses, n_periods))
    for j in range(n_periods):
        for k in range(n_buses):
            matrix[k, j] = file_cols[j * n_buses + k]
    
    all_states = []
    all_decisions = []
    
    num_observations = n_periods - 11  # First 11 cols are metadata
    
    # Process each bus
    for bus_idx in range(n_buses):
        bus_data = matrix[bus_idx, :]
        
        # Extract metadata (columns 0-10 are indexed 1-11 in documentation)
        # Column indices in Python (0-based):
        # 5: Odo_1 (odometer at 1st replacement)
        # 8: Odo_2 (odometer at 2nd replacement)
        odo_1 = bus_data[5]
        odo_2 = bus_data[8]
        
        # Odometer readings start at column 11 (index 11)
        bus_odo = bus_data[11:].copy()
        
        # Find replacement periods and adjust odometer
        replacement_periods = []
        
        if odo_1 > 0:
            # Find when odometer first reaches or exceeds Odo_1
            idx = np.where(bus_odo >= odo_1)[0]
            if len(idx) > 0:
                repl_period = idx[0]
                replacement_periods.append(repl_period)
                # Subtract Odo_1 from all subsequent readings
                bus_odo[repl_period:] -= odo_1
        
        if odo_2 > 0:
            # Find when odometer first reaches or exceeds Odo_2 (after first adjustment)
            idx = np.where(bus_odo >= odo_2)[0]
            if len(idx) > 0:
                repl_period = idx[0]
                replacement_periods.append(repl_period)
                # Subtract Odo_2 from all subsequent readings
                bus_odo[repl_period:] -= odo_2
        
        # Now bus_odo is "mileage since last replacement"
        # Discretize
        bus_states = (bus_odo / binsize).astype(int)
        
        # Create decision vector
        bus_decisions = np.zeros(len(bus_odo), dtype=int)
        for rp in replacement_periods:
            if rp < len(bus_decisions):
                bus_decisions[rp] = 1
        
        # Store
        all_states.extend(bus_states)
        all_decisions.extend(bus_decisions)
    
    return np.array(all_states), np.array(all_decisions)


def load_all_rust_data(binsize=5000):
    """
    Load all bus groups and combine into single dataset.
    
    Returns
    -------
    states : np.ndarray
        All state observations
    decisions : np.ndarray
        All replacement decisions
    """
    # Group dimensions (from Rust's documentation)
    groups_info = {
        'g870.asc': (36, 15),
        't8h203.asc': (81, 48),
        'a530875.asc': (128, 37),
    }
    
    all_states = []
    all_decisions = []
    
    for filename, (n_buses, n_periods) in groups_info.items():
        filepath = f'bus_engine_replacement/data/nfxp/dat/{filename}'
        print(f"\nLoading {filename}...")
        states, decisions = load_rust_group(filepath, n_buses, n_periods, binsize)
        all_states.extend(states)
        all_decisions.extend(decisions)
        print(f"  → {len(states)} observations, {np.sum(decisions)} replacements")
    
    return np.array(all_states), np.array(all_decisions)


if __name__ == "__main__":
    print("="*70)
    print("LOADING RUST DATA - CORRECT IMPLEMENTATION")
    print("Based on OpenSourceEconomics/zurcher-data")
    print("="*70)
    
    states, decisions = load_all_rust_data()
    
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total observations: {len(states)}")
    print(f"Total replacements: {np.sum(decisions)}")
    print(f"Replacement rate: {100*np.mean(decisions):.2f}%")
    
    print(f"\nState distribution:")
    print(f"  Min: {np.min(states)}, Max: {np.max(states)}")
    print(f"  Mean: {np.mean(states):.1f}, Std: {np.std(states):.1f}")
    
    print(f"\nEmpirical replacement frequency by state:")
    for state in [0, 10, 20, 30, 40, 50, 60, 70, 80]:
        in_state = states == state
        if np.sum(in_state) > 10:
            freq = np.mean(decisions[in_state])
            n = np.sum(in_state)
            print(f"  State {state} ({state*5}k miles): {freq:6.2%} (n={n:4d})")
    
    # Check monotonicity
    unique_states = np.unique(states)
    state_freqs = []
    state_counts = []
    
    for s in unique_states:
        in_state = states == s
        if np.sum(in_state) > 20:
            state_freqs.append(np.mean(decisions[in_state]))
            state_counts.append(s)
    
    state_freqs = np.array(state_freqs)
    state_counts = np.array(state_counts)
    
    diffs = np.diff(state_freqs)
    n_increasing = np.sum(diffs > -0.01)
    frac_increasing = n_increasing / len(diffs) if len(diffs) > 0 else 0
    
    print(f"\nMonotonicity check (states with >20 obs):")
    print(f"  Fraction increasing: {frac_increasing:.1%}")
    
    if frac_increasing > 0.7:
        print(f"  ✓ Mostly monotone increasing - data looks good!")
    else:
        print(f"  ~ Some non-monotonicity (expected in real noisy data)")
    
    # Check mileage after replacement
    if np.sum(decisions) > 0:
        repl_indices = np.where(decisions == 1)[0]
        # Check next observation after each replacement
        next_states = []
        for idx in repl_indices:
            if idx + 1 < len(states):
                next_states.append(states[idx + 1])
        
        if len(next_states) > 0:
            next_states = np.array(next_states)
            print(f"\nState immediately AFTER replacement:")
            print(f"  Mean: {np.mean(next_states):.1f} (should be near 0!)")
            print(f"  Median: {np.median(next_states):.1f}")
            print(f"  % at state 0: {100*np.mean(next_states == 0):.1f}%")
            
            if np.mean(next_states) < 2:
                print(f"  ✓✓✓ EXCELLENT! Replacements reset mileage properly!")
            else:
                print(f"  ~ Mileage not fully reset (may need parser adjustment)")
    
    # Save
    np.savez('rust_data_processed.npz', states=states, decisions=decisions)
    print(f"\n✓ Saved: rust_data_processed.npz")

