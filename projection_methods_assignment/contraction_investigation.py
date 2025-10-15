"""
Investigation of Contraction Properties in Discretized Dynamic Programming

This module investigates whether discretization preserves the contraction mapping
property of the Bellman operator. In the infinite-dimensional setting, the Bellman
operator T is a gamma-contraction:
    
    ||T(v1) - T(v2)||_inf <= gamma ||v1 - v2||_inf

However, when we discretize using projection methods (collocation, Galerkin, etc.),
the discretized operator may NOT preserve this property!

Theoretical Background:
-----------------------
The question is whether the "approximate" operator T_hat (which operates on
coefficient vectors) is still a contraction. This depends on:

1. **The basis functions**: Linear interpolation tends to preserve contraction,
   while high-order polynomials may not.

2. **The projection operator**: Different projections (L2, L-infinity, etc.)
   affect whether contraction is preserved.

3. **Gordon's Averagers (1995)**: Gordon showed that certain function approximation
   schemes ("averagers") preserve contraction if they are:
   - Non-expansive: ||Pi(v1) - Pi(v2)|| <= ||v1 - v2||
   - Related to the sampling points appropriately

This module empirically tests these theoretical predictions on the timber problem.

References:
-----------
- Gordon, G. (1995). "Stable Function Approximation in Dynamic Programming"
- Tsitsiklis, J. & Van Roy, B. (1996). "Feature-Based Methods for Large Scale DP"
- Gordon, G. (1999). "Approximate Solutions to Markov Decision Processes"

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List
from timber_projection import BellmanEquation, solve_timber_with_projection
from projection_framework import (
    LinearSplineBasis, ChebyshevBasis, LagrangeBasis,
    chebyshev_nodes, CollocationTest
)


class ContractionAnalyzer:
    """Analyze contraction properties of discretized Bellman operators"""
    
    def __init__(self, bellman_eq: BellmanEquation, basis, nodes: np.ndarray):
        """Initialize analyzer
        
        Args:
            bellman_eq: Bellman equation object
            basis: basis functions used for approximation
            nodes: collocation/interpolation nodes
        """
        self.bellman = bellman_eq
        self.basis = basis
        self.nodes = nodes
        self.domain = bellman_eq.get_domain()
        
        # For measuring distances, we'll use:
        # 1. L-infinity norm (sup norm)
        # 2. L2 norm on fine grid
        self.eval_grid = np.linspace(self.domain[0], self.domain[1], 500)
    
    def coeffs_to_function(self, coeffs: np.ndarray) -> Callable:
        """Convert coefficient vector to function"""
        def v_func(s):
            return self.basis.evaluate_approximation(s, coeffs)
        return v_func
    
    def apply_discretized_bellman(self, coeffs: np.ndarray) -> np.ndarray:
        """Apply discretized Bellman operator: T_hat(a) = a'
        
        This is the composition: Apply Bellman operator + Refit coefficients
        
        This follows the projection-based value iteration scheme:
        1. Convert coefficients to value function
        2. Apply Bellman operator at collocation nodes
        3. Fit new coefficients
        
        Args:
            coeffs: input coefficient vector
        
        Returns:
            new_coeffs: output coefficient vector (same dimension)
        """
        
        # TODO: Apply the discretized Bellman operator to coefficient vector
        #
        # This is ONE step of parametric value iteration: a → T_hat(a)
        #
        # Steps:
        # 1. Convert coefficients to a callable function:
        #    v_func = self.coeffs_to_function(coeffs)
        #
        # 2. Apply Bellman operator at nodes:
        #    v_new = self.bellman.bellman_operator(self.nodes, v_func)
        #
        # 3. Fit new coefficients so v_hat(nodes[i]) = v_new[i]:
        #    - Build basis matrix Phi where Phi[i,j] = phi_j(nodes[i])
        #    - Solve: new_coeffs = np.linalg.solve(Phi, v_new)
        #
        # This is the discretized operator we're testing for contraction!
        
        raise NotImplementedError("Implement discretized Bellman operator")
    
    def compute_distance_linf(self, coeffs1: np.ndarray, coeffs2: np.ndarray) -> float:
        """Compute L-infinity distance between two value functions
        
        ||v1 - v2||_inf = sup_s |v1(s) - v2(s)|
        
        Approximated by evaluating on a fine grid.
        """
        v1 = self.basis.evaluate_approximation(self.eval_grid, coeffs1)
        v2 = self.basis.evaluate_approximation(self.eval_grid, coeffs2)
        return np.max(np.abs(v1 - v2))
    
    def compute_distance_l2(self, coeffs1: np.ndarray, coeffs2: np.ndarray) -> float:
        """Compute L2 distance between two value functions
        
        ||v1 - v2||_2 = sqrt(integral (v1(s) - v2(s))^2 ds)
        
        Approximated using trapezoidal rule on a fine grid.
        """
        v1 = self.basis.evaluate_approximation(self.eval_grid, coeffs1)
        v2 = self.basis.evaluate_approximation(self.eval_grid, coeffs2)
        integrand = (v1 - v2) ** 2
        return np.sqrt(np.trapz(integrand, self.eval_grid))
    
    def estimate_lipschitz_constant(self, n_samples: int = 100, 
                                   norm: str = 'linf',
                                   coeffs_range: Tuple[float, float] = (-5, 5),
                                   seed: int = 42) -> Tuple[float, List[float]]:
        """Estimate Lipschitz constant of discretized Bellman operator
        
        The operator T is a gamma-contraction if:
            ||T(v1) - T(v2)|| <= gamma ||v1 - v2|| for all v1, v2
        
        We estimate the Lipschitz constant L = sup_{v1!=v2} ||T(v1) - T(v2)|| / ||v1 - v2||
        
        If L <= gamma, the discretized operator is a contraction!
        If L > gamma, contraction is not preserved (though convergence may still occur).
        
        Args:
            n_samples: number of random pairs to sample
            norm: 'linf' or 'l2'
            coeffs_range: range for random coefficient generation
            seed: random seed for reproducibility
        
        Returns:
            lipschitz_estimate: max observed ||T(v1)-T(v2)|| / ||v1-v2||
            all_ratios: list of all computed ratios
        """
        np.random.seed(seed)
        
        distance_fn = self.compute_distance_linf if norm == 'linf' else self.compute_distance_l2
        
        
        # TODO: Estimate the Lipschitz constant empirically via sampling
        #
        # Goal: Estimate L = sup ||T(v1) - T(v2)|| / ||v1 - v2||
        #
        # Algorithm:
        # 1. Sample n_samples random pairs of coefficient vectors (a1, a2)
        #    Use: np.random.uniform(coeffs_range[0], coeffs_range[1], n_coeffs)
        #
        # 2. For each pair:
        #    a. Compute input distance: dist_in = distance_fn(a1, a2)
        #    b. Apply discretized operator: Ta1 = self.apply_discretized_bellman(a1)
        #    c. Compute output distance: dist_out = distance_fn(Ta1, Ta2)
        #    d. Compute ratio: dist_out / dist_in
        #    e. Track all ratios
        #
        # 3. Return max(ratios) as Lipschitz estimate
        #
        # If L <= gamma: discretization preserves contraction!
        # If L > gamma: contraction is destroyed!
        
        raise NotImplementedError("Implement Lipschitz constant estimation")
    
    def visualize_operator_behavior(self, coeffs1: np.ndarray, coeffs2: np.ndarray):
        """Visualize how the discretized Bellman operator transforms two functions
        
        Shows:
        1. Input functions v1, v2
        2. Output functions T(v1), T(v2)
        3. Differences |v1 - v2| and |T(v1) - T(v2)|
        """
        # Apply operator
        Ta1 = self.apply_discretized_bellman(coeffs1)
        Ta2 = self.apply_discretized_bellman(coeffs2)
        
        # Evaluate on grid
        v1 = self.basis.evaluate_approximation(self.eval_grid, coeffs1)
        v2 = self.basis.evaluate_approximation(self.eval_grid, coeffs2)
        Tv1 = self.basis.evaluate_approximation(self.eval_grid, Ta1)
        Tv2 = self.basis.evaluate_approximation(self.eval_grid, Ta2)
        
        # Compute distances
        dist_input_linf = np.max(np.abs(v1 - v2))
        dist_output_linf = np.max(np.abs(Tv1 - Tv2))
        ratio_linf = dist_output_linf / dist_input_linf if dist_input_linf > 0 else 0
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Top left: input functions
        ax = axes[0, 0]
        ax.plot(self.eval_grid, v1, 'b-', linewidth=2, label='v1')
        ax.plot(self.eval_grid, v2, 'r-', linewidth=2, label='v2')
        ax.plot(self.nodes, self.basis.evaluate_approximation(self.nodes, coeffs1), 
                'bo', markersize=4, alpha=0.5)
        ax.plot(self.nodes, self.basis.evaluate_approximation(self.nodes, coeffs2),
                'ro', markersize=4, alpha=0.5)
        ax.set_xlabel('State (s)')
        ax.set_ylabel('Value')
        ax.set_title('Input Functions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Top right: output functions
        ax = axes[0, 1]
        ax.plot(self.eval_grid, Tv1, 'b-', linewidth=2, label='T(v1)')
        ax.plot(self.eval_grid, Tv2, 'r-', linewidth=2, label='T(v2)')
        ax.plot(self.nodes, self.basis.evaluate_approximation(self.nodes, Ta1),
                'bo', markersize=4, alpha=0.5)
        ax.plot(self.nodes, self.basis.evaluate_approximation(self.nodes, Ta2),
                'ro', markersize=4, alpha=0.5)
        ax.set_xlabel('State (s)')
        ax.set_ylabel('Value')
        ax.set_title('Output Functions (After Bellman Operator)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom left: input difference
        ax = axes[1, 0]
        ax.plot(self.eval_grid, np.abs(v1 - v2), 'g-', linewidth=2)
        ax.axhline(dist_input_linf, color='k', linestyle='--', alpha=0.5,
                  label=f'||v1 - v2||∞ = {dist_input_linf:.3f}')
        ax.set_xlabel('State (s)')
        ax.set_ylabel('|v1(s) - v2(s)|')
        ax.set_title('Input Difference')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom right: output difference
        ax = axes[1, 1]
        ax.plot(self.eval_grid, np.abs(Tv1 - Tv2), 'purple', linewidth=2)
        ax.axhline(dist_output_linf, color='k', linestyle='--', alpha=0.5,
                  label=f'||T(v1) - T(v2)||∞ = {dist_output_linf:.3f}')
        ax.axhline(self.bellman.delta * dist_input_linf, color='r', linestyle='--',
                  alpha=0.5, label=f'γ·||v1 - v2||∞ = {self.bellman.delta * dist_input_linf:.3f}')
        ax.set_xlabel('State (s)')
        ax.set_ylabel('|T(v1)(s) - T(v2)(s)|')
        ax.set_title(f'Output Difference (Ratio = {ratio_linf:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Contraction Analysis: {type(self.basis).__name__}\n' + 
                    f'{"CONTRACTION PRESERVED" if ratio_linf <= self.bellman.delta else "CONTRACTION NOT PRESERVED"}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


def compare_basis_functions():
    """Compare contraction properties across different basis functions"""
    
    print("\n" + "="*70)
    print("CONTRACTION PROPERTY INVESTIGATION")
    print("="*70)
    
    # Setup
    bellman_eq = BellmanEquation(delta=0.95, price=1.0, C=0.2, K=0.5, alpha=0.1)
    domain = bellman_eq.get_domain()
    n_nodes = 20
    
    # Test configurations
    configs = [
        ("Linear Splines (Uniform)", 
         LinearSplineBasis(np.linspace(domain[0], domain[1], n_nodes)),
         np.linspace(domain[0], domain[1], n_nodes)),
        
        ("Linear Splines (Chebyshev)", 
         LinearSplineBasis(chebyshev_nodes(n_nodes, domain)),
         chebyshev_nodes(n_nodes, domain)),
        
        ("Chebyshev Polynomials",
         ChebyshevBasis(domain),
         chebyshev_nodes(n_nodes, domain)),
        
        ("Lagrange Polynomials (Uniform)",
         LagrangeBasis(np.linspace(domain[0], domain[1], n_nodes)),
         np.linspace(domain[0], domain[1], n_nodes)),
        
        ("Lagrange Polynomials (Chebyshev)",
         LagrangeBasis(chebyshev_nodes(n_nodes, domain)),
         chebyshev_nodes(n_nodes, domain)),
    ]
    
    results = {}
    print("\nTesting 5 basis functions with 200 samples each...")
    
    for name, basis, nodes in configs:
        analyzer = ContractionAnalyzer(bellman_eq, basis, nodes)
        L_linf, ratios_linf = analyzer.estimate_lipschitz_constant(
            n_samples=200, norm='linf', coeffs_range=(-3, 3)
        )
        results[name] = {
            'lipschitz_linf': L_linf,
            'ratios_linf': ratios_linf,
        }
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Basis':<35} {'Lipschitz L':<12} {'Contraction?'}")
    print("-"*70)
    
    gamma = bellman_eq.delta
    for name, result in results.items():
        L = result['lipschitz_linf']
        # Use small tolerance for floating point comparison
        status = "✓ YES" if L <= gamma + 1e-6 else "✗ NO"
        print(f"{name:<35} {L:<12.4f} {status}")
    
    print(f"\nDiscount factor γ = {gamma:.4f}")
    print("Contraction preserved if L ≤ γ")
    
    # Visualization: distribution of ratios
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        if idx >= 6:
            break
        
        ax = axes[idx]
        ratios = result['ratios_linf']
        
        ax.hist(ratios, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(gamma, color='red', linestyle='--', linewidth=2, 
                  label=f'γ = {gamma:.2f}')
        ax.axvline(result['lipschitz_linf'], color='green', linestyle='-', linewidth=2,
                  label=f'Max = {result["lipschitz_linf"]:.3f}')
        ax.axvline(np.mean(ratios), color='orange', linestyle=':', linewidth=2,
                  label=f'Mean = {np.mean(ratios):.3f}')
        
        ax.set_xlabel('||T(v1) - T(v2)|| / ||v1 - v2||')
        ax.set_ylabel('Frequency')
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplot if any
    for idx in range(len(results), 6):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribution of Lipschitz Ratios Across Different Bases', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('contraction_comparison_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: contraction_comparison_histograms.png")
    
    return results


if __name__ == "__main__":
    compare_basis_functions()


