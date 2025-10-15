"""
Timber Cutting Problem using Generic Projection Framework

This demonstrates how to apply the projection framework to dynamic programming.
The Bellman equation is an operator equation N(v) = Lv - v = 0.

Problem:
    - State: s = biomass
    - Actions: x ∈ {0, 1} (don't cut, cut)
    - Dynamics: s' = K + exp(-alpha)*(s - K) if x=0, s'=0 if x=1
    - Reward: r(s, x) = (price*s - C)*x
    - Bellman equation: v(s) = max_x {r(s,x) + delta*v(s')}

Operator formulation:
    - N(v)(s) = Lv(s) - v(s)
    - L is the Bellman operator
    - We seek v such that N(v) = 0

Projection methods:
    - Collocation: v(s_i) = Lv(s_i) at nodes s_i
    - Galerkin: <Lv - v, phi_i> = 0 (uses quadrature)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no popup windows)
import matplotlib.pyplot as plt
from projection_framework import (
    BasisFunction, TestFunction, OperatorEquation, ProjectionMethod,
    LinearSplineBasis, ChebyshevBasis, CollocationTest, GalerkinTest,
    chebyshev_nodes
)


class BellmanEquation(OperatorEquation):
    """Bellman optimality equation for timber cutting
    
    N(v)(s) = Lv(s) - v(s) where L is the Bellman operator:
        Lv(s) = max_x {r(s,x) + delta*v(g(s,x))}
    """
    
    def __init__(self, delta: float = 0.95, price: float = 1.0, 
                 C: float = 0.2, K: float = 0.5, alpha: float = 0.1):
        """Initialize timber cutting problem
        
        Args:
            delta: discount factor
            price: price per unit biomass
            C: replanting cost
            K: carrying capacity (maximum biomass)
            alpha: growth rate parameter
        """
        self.delta = delta
        self.price = price
        self.C = C
        self.K = K
        self.alpha = alpha
    
    def get_domain(self):
        """State space is [0, K]"""
        return (0.0, self.K)
    
    def growth_function(self, s: np.ndarray) -> np.ndarray:
        """Biomass growth: s' = K + exp(-alpha)*(s - K)"""
        return self.K + np.exp(-self.alpha) * (s - self.K)
    
    def reward_function(self, s: np.ndarray, x: int) -> np.ndarray:
        """Reward for action x ∈ {0, 1}
        
        x=0: don't cut, reward = 0
        x=1: cut, reward = price*s - C
        """
        return (self.price * s - self.C) * x
    
    def bellman_operator(self, s: np.ndarray, v_func: callable) -> np.ndarray:
        """Apply Bellman operator: Lv(s) = max_x Q(s,x)
        
        Args:
            s: states to evaluate
            v_func: value function v(s)
        
        Returns:
            Lv(s): optimal value at each state
        """
        # Action 0: don't cut
        s_next_0 = self.growth_function(s)
        Q_0 = 0.0 + self.delta * v_func(s_next_0)
        
        # Action 1: cut
        s_next_1 = np.zeros_like(s)  # After cutting, biomass = 0
        Q_1 = self.price * s - self.C + self.delta * v_func(s_next_1)
        
        # Take maximum
        return np.maximum(Q_0, Q_1)
    
    def residual(self, s: np.ndarray, coeffs: np.ndarray,
                basis: BasisFunction) -> np.ndarray:
        """Compute residual: R(s; a) = Lv_hat(s) - v_hat(s)
        
        This is the KEY connection between the Bellman equation and projection methods!
        
        The residual measures how far our approximation v_hat is from satisfying
        the Bellman equation. For a true solution, R(s;a) = 0 for all s.
        
        Args:
            s: evaluation states
            coeffs: basis coefficients
            basis: basis functions
        
        Returns:
            R(s): residual at each state
        """
        
        # TODO: Compute the Bellman equation residual
        #
        # The residual is: R(s; a) = L(v_hat)(s) - v_hat(s)
        # where L is the Bellman operator and v_hat is your approximation
        #
        # Steps:
        # 1. Create a callable v_func(s_eval) that evaluates your approximation
        #    using basis.evaluate_approximation(s_eval, coeffs)
        # 
        # 2. Apply the Bellman operator: Lv = self.bellman_operator(s, v_func)
        #
        # 3. Evaluate the approximation: v_hat = v_func(s)
        #
        # 4. Return the residual: Lv - v_hat
        #
        # This measures how far v_hat is from satisfying the Bellman equation!
        
        raise NotImplementedError("Implement Bellman residual computation")
    
    def compute_policy(self, s: np.ndarray, coeffs: np.ndarray,
                       basis: BasisFunction) -> np.ndarray:
        """Compute optimal policy at states s
        
        Args:
            s: states
            coeffs: value function coefficients
            basis: basis functions
        
        Returns:
            policy: optimal action at each state (0 or 1)
        """
        def v_func(s_eval):
            return basis.evaluate_approximation(s_eval, coeffs)
        
        # Action 0: don't cut
        s_next_0 = self.growth_function(s)
        Q_0 = 0.0 + self.delta * v_func(s_next_0)
        
        # Action 1: cut
        s_next_1 = np.zeros_like(s)
        Q_1 = self.price * s - self.C + self.delta * v_func(s_next_1)
        
        # Choose action with higher Q-value
        return (Q_1 > Q_0).astype(int)


class BellmanProjectionMethod(ProjectionMethod):
    """Specialized projection method for Bellman equations
    
    Adds iterative solution via value function iteration.
    """
    
    def _solve_iterate(self, a0: np.ndarray, max_iter: int, tol: float,
                      verbose: bool) -> np.ndarray:
        """Solve using successive approximation (value iteration in coefficient space)
        
        THIS IS THE KEY INSIGHT: Parametric/Fitted Value Iteration!
        
        Traditional value iteration: v^(k+1)(s) = Lv^(k)(s) for all s
        Parametric value iteration: a^(k+1) = T_hat(a^(k)) where
            - We represent v using coefficients a
            - Apply L at collocation nodes
            - Refit coefficients to match
        
        This is exactly "fitted value iteration" from the DP literature!
        
        Algorithm:
            1. Start with coefficients a^(0)
            2. Compute v^(k+1) = Lv^(k) at collocation nodes  
            3. Fit new coefficients a^(k+1) so v_hat(s_i) = v^(k+1)_i
            4. Repeat until ||a^(k+1) - a^(k)|| < tol
        
        This only works with collocation test functions!
        """
        if not isinstance(self.test, CollocationTest):
            raise ValueError("Iterative method only works with collocation")
        
        
        # TODO: Implement parametric value iteration (THE KEY CONCEPTUAL INSIGHT!)
        #
        # This is fitted/parametric value iteration - value iteration in coefficient space!
        #
        # Algorithm (each iteration):
        # 1. Start with current coefficients a^(k) (which represent v_hat^(k))
        # 
        # 2. Create a callable value function from current coefficients:
        #    v_func(s) = basis.evaluate_approximation(s, coeffs)
        # 
        # 3. Apply Bellman operator at collocation nodes:
        #    v_new[i] = self.operator.bellman_operator(nodes, v_func)
        #    This gives you L(v_hat^(k)) evaluated at each node
        # 
        # 4. Fit new coefficients a^(k+1) such that:
        #    v_hat^(k+1)(s_i) = v_new[i] for all nodes s_i
        #    
        #    This is a linear system: Phi @ a^(k+1) = v_new
        #    where Phi[i,j] = phi_j(s_i) is the basis matrix
        #    
        #    Build Phi, then solve: coeffs_new = np.linalg.solve(Phi, v_new)
        # 
        # 5. Check convergence: if ||a^(k+1) - a^(k)|| < tol, stop
        # 
        # This IS value iteration, just in coefficient space instead of function space!
        # It's also called "fitted value iteration" in the DP literature.
        
        raise NotImplementedError("Implement parametric value iteration")


def solve_timber_with_projection(n_nodes: int = 50, basis_type: str = 'linear',
                                 test_type: str = 'collocation',
                                 solve_method: str = 'iterate',
                                 verbose: bool = True):
    """Solve timber cutting problem using projection methods
    
    Args:
        n_nodes: number of collocation nodes
        basis_type: 'linear' (splines) or 'chebyshev' (polynomials)
        test_type: 'collocation' or 'galerkin'
        solve_method: 'iterate', 'fsolve', or 'minimize'
        verbose: print convergence info
    
    Returns:
        coeffs: solution coefficients
        bellman_eq: Bellman equation object
        projection: projection method object
    """
    # Create Bellman equation
    bellman_eq = BellmanEquation(delta=0.95, price=1.0, C=0.2, K=0.5, alpha=0.1)
    domain = bellman_eq.get_domain()
    
    # Choose collocation nodes
    if basis_type == 'chebyshev':
        nodes = chebyshev_nodes(n_nodes, domain)
    else:
        nodes = np.linspace(domain[0], domain[1], n_nodes)
    
    # Create basis functions
    if basis_type == 'linear':
        basis = LinearSplineBasis(nodes)
        n_basis = len(nodes)
    elif basis_type == 'chebyshev':
        basis = ChebyshevBasis(domain)
        n_basis = n_nodes  # Use same number as nodes
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")
    
    # Create test functions
    if test_type == 'collocation':
        test = CollocationTest(nodes)
    elif test_type == 'galerkin':
        test = GalerkinTest(basis, n_quad=200)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Create projection method
    projection = BellmanProjectionMethod(bellman_eq, basis, test, n_basis)
    
    # Initial guess: v(s) = 0
    a0 = np.zeros(n_basis)
    
    # Solve
    if verbose:
        print("=" * 70)
        print("SOLVING TIMBER CUTTING WITH PROJECTION METHODS")
        print("=" * 70)
        print(f"Basis: {basis_type} (n={n_basis})")
        print(f"Test: {test_type}")
        print(f"Method: {solve_method}")
        print()
    
    coeffs = projection.solve(a0, method=solve_method, verbose=verbose)
    
    return coeffs, bellman_eq, projection


def compare_methods():
    """Compare different projection methods on timber problem"""
    
    print("\n" + "=" * 70)
    print("COMPARING PROJECTION METHODS FOR TIMBER CUTTING")
    print("=" * 70 + "\n")
    
    n_nodes = 30
    results = {}
    
    # Test configurations: compare collocation vs Galerkin
    configs = [
        ('linear', 'collocation', 'iterate', 'Linear + Collocation'),
        ('linear', 'galerkin', 'fsolve', 'Linear + Galerkin'),
        ('chebyshev', 'collocation', 'iterate', 'Chebyshev + Collocation'),
        ('chebyshev', 'galerkin', 'fsolve', 'Chebyshev + Galerkin'),
    ]
    
    for basis_type, test_type, solve_method, display_name in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {display_name}")
        print('='*70)
        
        try:
            coeffs, bellman_eq, projection = solve_timber_with_projection(
                n_nodes=n_nodes, basis_type=basis_type, test_type=test_type,
                solve_method=solve_method, verbose=True
            )
            results[display_name] = (coeffs, bellman_eq, projection)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Single plot comparing all methods
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    s_plot = np.linspace(0, 0.5, 200)
    
    for idx, (name, (coeffs, bellman_eq, projection)) in enumerate(results.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        
        # Plot value function
        v = projection.evaluate_solution(s_plot, coeffs)
        ax.plot(s_plot, v, 'b-', linewidth=2, label='Value function')
        
        # Plot policy (cutting threshold)
        policy = bellman_eq.compute_policy(s_plot, coeffs, projection.basis)
        
        # Find cutting threshold
        cut_indices = np.where(policy == 1)[0]
        if len(cut_indices) > 0:
            threshold = s_plot[cut_indices[0]]
            ax.axvline(threshold, color='r', linestyle='--', alpha=0.7,
                      label=f'Cut threshold: {threshold:.3f}')
        
        # Compute and show max residual
        residual = projection.compute_residual(s_plot, coeffs)
        max_res = np.max(np.abs(residual))
        
        ax.set_xlabel('Biomass (s)')
        ax.set_ylabel('Value')
        ax.set_title(f'{name}\nMax |residual| = {max_res:.2e}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timber_projection_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: timber_projection_comparison.png")
    plt.close()
    
    return results


if __name__ == "__main__":
    # Compare all projection methods
    compare_methods()

