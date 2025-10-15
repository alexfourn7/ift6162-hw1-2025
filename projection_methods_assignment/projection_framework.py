"""
Generic Projection Method Framework for Functional Equations

This module implements projection methods for discretizing functional equations.
Applied here to the Bellman equation for dynamic programming.

Key Idea:
    1. Start with operator equation N(f) = 0 (infinite dimensional)
    2. Choose basis functions: f_hat(x) = sum(a_i * phi_i(x))
    3. Compute residual: R(x; a) = N(f_hat)(x)
    4. Choose test functions p_i and require <R, p_i> = 0
    5. Solve finite dimensional system for coefficients a

Different choices of test functions give different methods:
    - Collocation: p_i = delta(x - x_i)  =>  R(x_i) = 0
    - Galerkin: p_i = phi_i  =>  <R, phi_i> = 0
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import fsolve
from typing import Callable, Tuple, Optional
import matplotlib.pyplot as plt


# =============================================================================
# 1. BASIS FUNCTIONS (for approximating the unknown function)
# =============================================================================

class BasisFunction(ABC):
    """Abstract base class for basis functions phi_i(x)"""
    
    @abstractmethod
    def __call__(self, x: np.ndarray, i: int) -> np.ndarray:
        """Evaluate i-th basis function at points x
        
        Args:
            x: evaluation points [shape: (n_points,)]
            i: basis function index (0-indexed)
        
        Returns:
            phi_i(x) [shape: (n_points,)]
        """
        pass
    
    @abstractmethod
    def derivative(self, x: np.ndarray, i: int) -> np.ndarray:
        """Evaluate derivative of i-th basis function at points x
        
        Args:
            x: evaluation points
            i: basis function index
        
        Returns:
            d/dx phi_i(x)
        """
        pass
    
    def evaluate_approximation(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate f_hat(x) = sum_i a_i * phi_i(x)
        
        Args:
            x: evaluation points [shape: (n_points,)]
            coeffs: basis coefficients [shape: (n_basis,)]
        
        Returns:
            f_hat(x) [shape: (n_points,)]
        """
        result = np.zeros_like(x, dtype=float)
        for i in range(len(coeffs)):
            result += coeffs[i] * self(x, i)
        return result
    
    def evaluate_derivative(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate d/dx f_hat(x) = sum_i a_i * d/dx phi_i(x)
        
        Args:
            x: evaluation points
            coeffs: basis coefficients
        
        Returns:
            d/dx f_hat(x)
        """
        result = np.zeros_like(x, dtype=float)
        for i in range(len(coeffs)):
            result += coeffs[i] * self.derivative(x, i)
        return result


class LinearSplineBasis(BasisFunction):
    """Piecewise linear "hat" functions (finite element basis)
    
    phi_i is 1 at node x_i, 0 at other nodes, linear in between.
    Shape-preserving: maintains monotonicity and concavity.
    """
    
    def __init__(self, nodes: np.ndarray):
        """Initialize with collocation nodes
        
        Args:
            nodes: collocation/interpolation nodes [shape: (n_nodes,)]
        """
        self.nodes = nodes
        self.n_nodes = len(nodes)
    
    def __call__(self, x: np.ndarray, i: int) -> np.ndarray:
        """Evaluate hat function centered at node i"""
        result = np.zeros_like(x, dtype=float)
        
        if i > 0:
            # Left piece: rises from 0 to 1
            mask = (x >= self.nodes[i-1]) & (x < self.nodes[i])
            result[mask] = (x[mask] - self.nodes[i-1]) / (self.nodes[i] - self.nodes[i-1])
        
        if i < self.n_nodes - 1:
            # Right piece: falls from 1 to 0
            mask = (x >= self.nodes[i]) & (x <= self.nodes[i+1])
            result[mask] = (self.nodes[i+1] - x[mask]) / (self.nodes[i+1] - self.nodes[i])
        
        # Handle exact match at node i
        result[x == self.nodes[i]] = 1.0
        
        return result
    
    def derivative(self, x: np.ndarray, i: int) -> np.ndarray:
        """Derivative of hat function (piecewise constant)"""
        result = np.zeros_like(x, dtype=float)
        
        if i > 0:
            # Left piece: positive slope
            left_slope = 1.0 / (self.nodes[i] - self.nodes[i-1])
            mask = (x > self.nodes[i-1]) & (x < self.nodes[i])
            result[mask] = left_slope
            if i < self.n_nodes - 1:
                result[x == self.nodes[i-1]] = left_slope
        
        if i < self.n_nodes - 1:
            # Right piece: negative slope
            right_slope = -1.0 / (self.nodes[i+1] - self.nodes[i])
            mask = (x > self.nodes[i]) & (x < self.nodes[i+1])
            result[mask] = right_slope
            result[x == self.nodes[i]] = right_slope
        elif i == self.n_nodes - 1 and i > 0:
            result[x == self.nodes[i]] = left_slope
        
        return result
    
    def evaluate_derivative(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate derivative of piecewise linear interpolant
        
        For piecewise linear functions, the derivative is piecewise constant.
        In interval [nodes[i], nodes[i+1]], derivative = (coeffs[i+1] - coeffs[i]) / (nodes[i+1] - nodes[i])
        """
        result = np.zeros_like(x, dtype=float)
        
        for i in range(self.n_nodes - 1):
            slope = (coeffs[i+1] - coeffs[i]) / (self.nodes[i+1] - self.nodes[i])
            mask = (x > self.nodes[i]) & (x < self.nodes[i+1])
            result[mask] = slope
            result[x == self.nodes[i]] = slope
        
        # At rightmost node, use left derivative
        if self.n_nodes > 1:
            last_slope = (coeffs[-1] - coeffs[-2]) / (self.nodes[-1] - self.nodes[-2])
            result[x == self.nodes[-1]] = last_slope
        
        return result


class ChebyshevBasis(BasisFunction):
    """Chebyshev polynomial basis: T_i(x)
    
    Much better numerical properties than monomials.
    Defined on [-1, 1], use coordinate transformation for other domains.
    
    Recurrence relation:
        T_0(x) = 1
        T_1(x) = x
        T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x)
    """
    
    def __init__(self, domain: Tuple[float, float] = (-1, 1)):
        """Initialize with domain [a, b]
        
        Args:
            domain: domain [a, b] for the basis functions
        """
        self.domain = domain
    
    def _transform_to_standard(self, x: np.ndarray) -> np.ndarray:
        """Transform x from [a, b] to [-1, 1]"""
        a, b = self.domain
        return 2 * (x - a) / (b - a) - 1
    
    def __call__(self, x: np.ndarray, i: int) -> np.ndarray:
        x_std = self._transform_to_standard(x)
        return np.polynomial.chebyshev.chebval(x_std, np.eye(i + 1)[i])
    
    def derivative(self, x: np.ndarray, i: int) -> np.ndarray:
        x_std = self._transform_to_standard(x)
        deriv_coeffs = np.polynomial.chebyshev.chebder(np.eye(i + 1)[i])
        # Chain rule: multiply by d/dx of coordinate transformation
        a, b = self.domain
        return (2 / (b - a)) * np.polynomial.chebyshev.chebval(x_std, deriv_coeffs)


class LagrangeBasis(BasisFunction):
    """Lagrange polynomial basis at given nodes
    
    phi_i(x) is the Lagrange polynomial that is 1 at node i and 0 at all other nodes.
    For n nodes, we get polynomials of degree n-1.
    """
    
    def __init__(self, nodes: np.ndarray):
        """Initialize with collocation nodes
        
        Args:
            nodes: collocation points [shape: (n_nodes,)]
        """
        self.nodes = nodes
        self.n_nodes = len(nodes)
    
    def __call__(self, x: np.ndarray, i: int) -> np.ndarray:
        """Evaluate i-th Lagrange polynomial
        
        L_i(x) = product_{j != i} (x - x_j) / (x_i - x_j)
        """
        result = np.ones_like(x, dtype=float)
        
        for j in range(self.n_nodes):
            if j != i:
                result *= (x - self.nodes[j]) / (self.nodes[i] - self.nodes[j])
        
        return result
    
    def derivative(self, x: np.ndarray, i: int) -> np.ndarray:
        """Evaluate derivative of i-th Lagrange polynomial
        
        Uses product rule: d/dx[product f_k] = sum_k [f'_k * product_{j!=k} f_j]
        """
        result = np.zeros_like(x, dtype=float)
        
        for k in range(self.n_nodes):
            if k == i:
                continue
            
            # Derivative of (x - x_k) / (x_i - x_k) is 1 / (x_i - x_k)
            term = np.ones_like(x, dtype=float) / (self.nodes[i] - self.nodes[k])
            
            # Multiply by all other factors
            for j in range(self.n_nodes):
                if j != i and j != k:
                    term *= (x - self.nodes[j]) / (self.nodes[i] - self.nodes[j])
            
            result += term
        
        return result


# =============================================================================
# 2. TEST FUNCTIONS (for measuring residual)
# =============================================================================

class TestFunction(ABC):
    """Abstract base class for test functions p_i
    
    Test functions define how we measure the residual:
        <R, p_i> = 0  for i = 1, ..., n
    """
    
    @abstractmethod
    def apply(self, residual_func: Callable, coeffs: np.ndarray, 
              domain: Tuple[float, float]) -> np.ndarray:
        """Apply test to residual function
        
        Args:
            residual_func: function R(x; a) that takes x and coeffs
            coeffs: current coefficient values
            domain: domain [a, b] for integration/evaluation
        
        Returns:
            test_values: <R, p_i> for i = 0, ..., n-1 [shape: (n_basis,)]
        """
        pass


class CollocationTest(TestFunction):
    """Collocation: p_i = delta(x - x_i)
    
    This gives pointwise conditions: R(x_i; a) = 0
    Most computationally efficient - no integration needed!
    """
    
    def __init__(self, nodes: np.ndarray):
        """Initialize with collocation nodes
        
        Args:
            nodes: collocation points [shape: (n_nodes,)]
        """
        self.nodes = nodes
    
    def apply(self, residual_func: Callable, coeffs: np.ndarray,
              domain: Tuple[float, float]) -> np.ndarray:
        """Evaluate residual at collocation points
        
        For collocation, we test against delta functions at nodes,
        which means we just evaluate the residual pointwise.
        """
        
        # TODO: Evaluate the residual function at the collocation nodes
        # Hint: Collocation uses delta functions, so <R, delta(x-xi)> = R(xi)
        raise NotImplementedError("Implement collocation test")


class GalerkinTest(TestFunction):
    """Galerkin: p_i = phi_i (same as basis functions)
    
    This gives: <R, phi_i> = integral R(x; a) * phi_i(x) w(x) dx = 0
    Theoretically optimal but requires integration.
    """
    
    def __init__(self, basis: BasisFunction, weight: Optional[Callable] = None,
                 n_quad: int = 100, quad_type: str = 'gauss'):
        """Initialize Galerkin test
        
        Args:
            basis: basis functions (used as test functions)
            weight: weight function w(x) for inner product (default: uniform)
            n_quad: number of quadrature points for integration
            quad_type: 'gauss' for Gaussian quadrature, 'uniform' for trapezoidal
        """
        self.basis = basis
        self.weight = weight if weight is not None else lambda x: np.ones_like(x)
        self.n_quad = n_quad
        self.quad_type = quad_type
    
    def apply(self, residual_func: Callable, coeffs: np.ndarray,
              domain: Tuple[float, float]) -> np.ndarray:
        """Compute <R, phi_i> for each basis function using quadrature
        
        Galerkin requires computing integrals:
            <R, phi_i> = integral R(x) * phi_i(x) * w(x) dx
        
        We use numerical quadrature (Gauss-Legendre or trapezoidal).
        """
        a, b = domain
        
        
        # TODO: Implement Galerkin projection using numerical quadrature
        # 
        # You need to compute: integral_a^b R(x) * phi_i(x) * w(x) dx = 0 for each i
        # 
        # Steps:
        # 1. Get quadrature nodes and weights:
        #    - For Gaussian: use np.polynomial.legendre.leggauss(self.n_quad)
        #    - Transform nodes from [-1,1] to [a,b]
        #    - Adjust weights for the domain transformation
        # 
        # 2. Evaluate the residual at quadrature points
        # 
        # 3. For each basis function i:
        #    - Evaluate phi_i at quadrature points
        #    - Compute weighted sum: sum(w_quad * R * phi_i * w_weight)
        # 
        # Hint: Gauss-Legendre quadrature approximates integral f(x)dx as sum w_j*f(x_j)
        
        raise NotImplementedError("Implement Galerkin test with quadrature")


# =============================================================================
# 3. OPERATOR EQUATIONS (define the problem)
# =============================================================================

class OperatorEquation(ABC):
    """Abstract base class for operator equations N(f) = 0
    
    Subclasses define specific problems (e.g., BellmanEquation for DP).
    """
    
    @abstractmethod
    def residual(self, x: np.ndarray, coeffs: np.ndarray,
                basis: BasisFunction) -> np.ndarray:
        """Compute residual R(x; a) = N(f_hat)(x)
        
        Args:
            x: evaluation points
            coeffs: basis coefficients defining f_hat
            basis: basis functions
        
        Returns:
            R(x; a): residual at each point x
        """
        pass
    
    @abstractmethod
    def get_domain(self) -> Tuple[float, float]:
        """Return domain [a, b] for the problem"""
        pass


# =============================================================================
# 4. PROJECTION METHOD (puts it all together)
# =============================================================================

class ProjectionMethod:
    """Generic projection method solver
    
    Solves operator equation N(f) = 0 by:
        1. Approximating f with basis functions
        2. Computing residual R(x; a)
        3. Enforcing test conditions <R, p_i> = 0
    """
    
    def __init__(self, operator: OperatorEquation, basis: BasisFunction,
                 test: TestFunction, n_basis: int):
        """Initialize projection method
        
        Args:
            operator: operator equation to solve
            basis: basis functions for approximation
            test: test functions for projection
            n_basis: number of basis functions
        """
        self.operator = operator
        self.basis = basis
        self.test = test
        self.n_basis = n_basis
        self.domain = operator.get_domain()
    
    def solve(self, a0: Optional[np.ndarray] = None, method: str = 'fsolve',
             max_iter: int = 100, tol: float = 1e-6, 
             verbose: bool = True) -> np.ndarray:
        """Solve for coefficients a
        
        Args:
            a0: initial guess for coefficients
            method: 'fsolve' for Newton's method, 'iterate' for fixed-point
            max_iter: maximum iterations
            tol: convergence tolerance
            verbose: print convergence info
        
        Returns:
            coeffs: solution coefficients [shape: (n_basis,)]
        """
        if a0 is None:
            a0 = np.zeros(self.n_basis)
        
        if method == 'fsolve':
            return self._solve_fsolve(a0, verbose)
        elif method == 'iterate':
            return self._solve_iterate(a0, max_iter, tol, verbose)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _solve_fsolve(self, a0: np.ndarray, verbose: bool) -> np.ndarray:
        """Solve using Newton's method (via scipy.optimize.fsolve)"""
        
        def equations(coeffs):
            """System of equations: <R, p_i> = 0"""
            residual_func = lambda x, a: self.operator.residual(x, a, self.basis)
            return self.test.apply(residual_func, coeffs, self.domain)
        
        if verbose:
            print("Solving with Newton's method (fsolve)...")
        
        solution = fsolve(equations, a0, full_output=True)
        coeffs = solution[0]
        info = solution[1]
        
        if verbose:
            print(f"  Iterations: {info['nfev']}")
            print(f"  Final residual norm: {np.linalg.norm(info['fvec']):.2e}")
        
        return coeffs
    
    def _solve_iterate(self, a0: np.ndarray, max_iter: int, tol: float,
                      verbose: bool) -> np.ndarray:
        """Solve using successive approximation (fixed-point iteration)
        
        This only works when the operator has a natural fixed-point structure,
        like the Bellman equation.
        """
        raise NotImplementedError("Subclass must implement iterative solution")
    
    def evaluate_solution(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate f_hat(x) using solution coefficients"""
        return self.basis.evaluate_approximation(x, coeffs)
    
    def compute_residual(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Compute residual R(x; a) for diagnostics"""
        return self.operator.residual(x, coeffs, self.basis)
    
    def plot_solution(self, coeffs: np.ndarray, n_plot: int = 200,
                     title: str = "Solution"):
        """Plot solution and residual
        
        Args:
            coeffs: solution coefficients
            n_plot: number of points for plotting
            title: plot title
        """
        a, b = self.domain
        x = np.linspace(a, b, n_plot)
        
        # Evaluate solution
        f_hat = self.evaluate_solution(x, coeffs)
        
        # Compute residual
        R = self.compute_residual(x, coeffs)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(x, f_hat, 'b-', linewidth=2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title(f'{title}: Approximate Solution')
        ax1.grid(True, alpha=0.3)
        
        # Mark collocation points if using collocation
        if isinstance(self.test, CollocationTest):
            nodes = self.test.nodes
            f_nodes = self.evaluate_solution(nodes, coeffs)
            ax1.plot(nodes, f_nodes, 'ro', markersize=6, 
                    label=f'Collocation nodes (n={len(nodes)})')
            ax1.legend()
        
        ax2.plot(x, R, 'r-', linewidth=2)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('x')
        ax2.set_ylabel('R(x)')
        ax2.set_title('Residual')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        
        plt.tight_layout()
        return fig


# =============================================================================
# 5. UTILITY FUNCTIONS
# =============================================================================

def chebyshev_nodes(n: int, domain: Tuple[float, float] = (-1, 1)) -> np.ndarray:
    """Generate Chebyshev nodes (zeros of T_n)
    
    These are optimal collocation points - they minimize interpolation error
    and produce well-conditioned systems.
    
    Args:
        n: number of nodes
        domain: domain [a, b]
    
    Returns:
        nodes: Chebyshev nodes [shape: (n,)]
    """
    a, b = domain
    # Standard Chebyshev nodes on [-1, 1]
    k = np.arange(1, n + 1)
    x_std = np.cos((2 * k - 1) * np.pi / (2 * n))
    
    # Transform to [a, b]
    x = 0.5 * (b - a) * x_std + 0.5 * (a + b)
    
    return np.sort(x)  # Return in increasing order


if __name__ == "__main__":
    print("Projection Framework Module")
    print("=" * 60)
    print("This module provides generic tools for projection methods.")
    print("")
    print("Main components:")
    print("  - BasisFunction: for approximating f_hat")
    print("  - TestFunction: for measuring residual")
    print("  - OperatorEquation: for defining problems")
    print("  - ProjectionMethod: for solving")
    print("")
    print("See timber_projection.py for the dynamic programming application.")
    print("=" * 60)
