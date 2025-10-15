"""
Bus Engine Replacement - Smooth Bellman Equation

Implements the smooth (entropy-regularized) Bellman equation:
    v(s) = (1/β) log Σ_a exp(β(r(s,a) + γ Σ_j p(j|s,a) v(j)))

where:
- v(s): value function
- β: inverse temperature (controls smoothness)
- α = 1/β: entropy regularization weight
- γ: discount factor  
- r(s,a): reward function (negative cost)
- p(j|s,a): transition probabilities
- π(a|s): softmax policy

Uses JAX with jaxopt for implicit differentiation through the fixed point.

Based on Rust (1987) "Optimal Replacement of GMC Bus Engines"
Author: Pierre-Luc Bacon
Course: IFT6162
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxopt import FixedPointIteration
import optax
import matplotlib.pyplot as plt
from data_processing import load_all_rust_data

print("="*70)
print("Bus Engine Replacement - Smooth Bellman Equation")
print("="*70)

# Load real data
print("\n[1] Loading Real Data")
states_data, decisions_data = load_all_rust_data(binsize=5000)
print(f"✓ {len(states_data)} observations, {np.sum(decisions_data)} replacements")

# Estimate transitions
usage = np.diff(states_data)
usage_valid = usage[(usage >= 0) & (usage < 10)]
counts = np.bincount(usage_valid.astype(int), minlength=max(int(np.max(usage_valid))+1, 6))
trans_probs = counts / counts.sum()

# Build transition matrix
n_states = 90
T = np.zeros((n_states, n_states))
for x in range(n_states):
    for j in range(len(trans_probs)):
        if x + j < n_states - 1:
            T[x, x + j] = trans_probs[j]
        elif x + j == n_states - 1:
            T[x, n_states - 1] = trans_probs[j:].sum()

print(f"✓ Transition matrix estimated")

# Model parameters following course notation  
gamma = 0.95  # Discount factor (γ) - enables fast convergence
cost_scale = 0.001  # Rust's scaling: costs in units of 0.001

print(f"\n[2] Model Specification (Course Notation)")
print(f"  γ (discount factor) = {gamma}")
print(f"  Cost scaling = {cost_scale}")
print(f"  Note: Rust's formulation = smooth Bellman with β=1 (implicit)")
print(f"        v(s) = log Σ_a exp(q(s,a)) where q = r + γ E[v]")
print(f"        Rust used γ=0.9999; we use 0.95 for computational efficiency")

# Define smooth Bellman operator following Rust (1987)
# In course notation: v(s) = log Σ_a exp(q(s,a)) where q(s,a) = r(s,a) + γ E[v(s')]
# (This is β=1 case of the general smooth Bellman equation)
def smooth_bellman_operator(v, theta):
    """
    Smooth Bellman operator (Rust's formulation with course notation).
    
    v(s) = log Σ_a exp(q(s,a))  [implicit β=1]
    
    where q(s,a) = r(s,a) + γ * E[v(s') | s, a]
    
    Parameters
    ----------
    v : jnp.ndarray
        Value function v(s)
    theta : jnp.ndarray
        Cost parameters [RC, C1]
    
    Returns
    -------
    v_new : jnp.ndarray
        Updated value function
    """
    RC, C1 = theta
    states = jnp.arange(n_states)
    
    # Costs
    cost_keep = C1 * states * cost_scale
    cost_replace = RC + cost_keep[0]
    
    
    # TODO: Implement the smooth Bellman operator
    # 
    # Compute Q-values: q(s,a) = r(s,a) + γ * E[v(s') | s, a]
    #   where r(s,a) = -cost(s,a) (negative cost)
    #   For keep: E[v(s')] = Σ_j p(j|s,keep) v(j) = T @ v
    #   For replace: E[v(s')] = v(0) (always go to state 0)
    #
    # Then apply smooth Bellman: v(s) = log Σ_a exp(q(s,a))
    #   Use jax.scipy.special.logsumexp for numerical stability
    
    q_keep = -cost_keep + gamma * jnp.dot(jnp.array(T), v)
    q_replace = -cost_replace + gamma * v[0]
    q_both = jnp.stack([q_keep, jnp.full(n_states, q_replace)], axis=1)
    v_new = logsumexp(q_both, axis=1)
    return v_new
    

def solve_smooth_bellman(theta):
    """
    Solve smooth Bellman equation using jaxopt with implicit differentiation.
    
    With γ=0.95, standard fixed-point iteration converges quickly.
    
    Parameters
    ----------
    theta : jnp.ndarray
        Cost parameters [RC, C1]
    
    Returns
    -------
    v : jnp.ndarray
        Converged value function v*(s)
    """
    fp_solver = FixedPointIteration(
        fixed_point_fun=lambda v, th: smooth_bellman_operator(v, th),
        maxiter=500,
        tol=1e-10,
        implicit_diff=True
    )
    
    v_init = jnp.zeros(n_states)
    result = fp_solver.run(v_init, theta)
    return result.params

def compute_policy(v, theta):
    """
    Compute softmax policy π(a|s) from value function.
    
    π(a|s) = exp(q(s,a)) / Σ_a' exp(q(s,a'))  [β=1 implicit]
    
    Parameters
    ----------
    v : jnp.ndarray
        Value function v(s)
    theta : jnp.ndarray
        Parameters [RC, C1]
    
    Returns
    -------
    pi : jnp.ndarray
        Policy π(a|s) of shape [n_states, 2]
    """
    RC, C1 = theta
    states = jnp.arange(n_states)
    
    # Costs
    cost_keep = C1 * states * cost_scale
    cost_replace = RC + cost_keep[0]
    
    
    # TODO: Compute softmax policy π(a|s) = exp(q(s,a)) / Σ exp(q(s,a'))
    # Hint: Same Q-value computation as in smooth_bellman_operator
    #       Then: log π(a|s) = q(s,a) - log Σ_a' exp(q(s,a'))
    
    q_keep = -cost_keep + gamma * jnp.dot(jnp.array(T), v)
    q_replace = -cost_replace + gamma * v[0]
    q_both = jnp.stack([q_keep, jnp.full(n_states, q_replace)], axis=1)
    log_pi = q_both - logsumexp(q_both, axis=1, keepdims=True)
    return jnp.exp(log_pi)
    

def log_likelihood(theta):
    """
    Compute log-likelihood of observed data.
    
    L(θ) = Σ_i log π(a_i | s_i; θ)
    
    Parameters
    ----------
    theta : jnp.ndarray
        Parameters [RC, C1]
        
    Returns
    -------
    ll : float
        Log-likelihood
    """
    
    # TODO: Implement log-likelihood L(θ) = Σ_i log π(a_i | s_i; θ)
    v = solve_smooth_bellman(theta)
    pi = compute_policy(v, theta)
    probs_observed = pi[states_data[:-1], decisions_data[:-1]]
    return jnp.sum(jnp.log(probs_observed + 1e-10))
    

if __name__ == "__main__":
    # Verify convergence
    print(f"\n[3] Verifying Convergence with γ={gamma}")
    theta_test = jnp.array([10.0, 2.5])
    v_test = solve_smooth_bellman(theta_test)
residual = v_test - smooth_bellman_operator(v_test, theta_test)
max_res = float(jnp.max(jnp.abs(residual)))
print(f"  Max residual ||v - L(v)||: {max_res:.2e}")
if max_res < 1e-6:
    print(f"  ✓ CONVERGED! Implicit differentiation will work correctly")
else:
    print(f"  ~ Residual: {max_res:.2e}")

# Estimation via MLE
print(f"\n[4] Maximum Likelihood Estimation")
print(f"  Using implicit differentiation through smooth Bellman operator")

theta_init = jnp.array([12.0, 1.5])  # [RC, C1]
print(f"  Initial guess: θ = [RC={theta_init[0]:.2f}, C1={theta_init[1]:.2f}]")
print(f"  Using Adam optimizer with implicit differentiation")

# Gradient-based optimization
grad_fn = jax.grad(lambda theta: -log_likelihood(theta))

optimizer = optax.chain(
    optax.clip_by_global_norm(10.0),
    optax.adam(0.01)
)

theta = theta_init
opt_state = optimizer.init(theta)

losses = []
theta_history = []
policy_history = []

n_steps = 250
print(f"\nOptimizing ({n_steps} steps)...")

for i in range(n_steps):
    loss = -log_likelihood(theta)
    grads = grad_fn(theta)
    
    updates, opt_state = optimizer.update(grads, opt_state)
    theta = optax.apply_updates(theta, updates)
    
    losses.append(float(loss))
    theta_history.append(np.array(theta))
    
    # Save policy every few steps for animation
    if i % 5 == 0 or i < 10 or i >= n_steps - 10:
        v_iter = solve_smooth_bellman(theta)
        pi_iter = compute_policy(v_iter, theta)
        policy_history.append((i, np.array(pi_iter), np.array(theta)))
    
    if i % 50 == 0:
        print(f"  Step {i:3d}: -log L(θ)={loss:10.2f}, θ=[{theta[0]:.2f}, {theta[1]:.2f}]")

# Show final
print(f"  Step {n_steps-1:3d}: -log L(θ)={losses[-1]:10.2f}, θ=[{theta_history[-1][0]:.2f}, {theta_history[-1][1]:.2f}]")

theta_hat = theta_history[-1]  # Final estimate

print(f"\n✓ Estimation complete!")
print(f"\n  Estimated: θ̂ = [RC={theta_hat[0]:.2f}, C1={theta_hat[1]:.3f}]")
print(f"  Published: θ = [RC=9.75, C1=2.633] (Rust 1987)")
print(f"  Error: ΔRC={abs(theta_hat[0]-9.75):.2f}, ΔC1={abs(theta_hat[1]-2.633):.3f}")

# Create final plots
print(f"\n[5] Creating Plots")

# Solve with BOTH methods to verify they match
print("  Computing with both implicit_diff and backprop...")

# Method 1: implicit_diff (jaxopt)
v_implicit = solve_smooth_bellman(jnp.array(theta_hat))
pi_implicit = compute_policy(v_implicit, jnp.array(theta_hat))

# Method 2: backprop through unrolled iterations  
def solve_backprop(theta):
    """Solve by unrolling iterations (for verification)."""
    def iterate(v, _):
        return smooth_bellman_operator(v, theta), None
    v_init = jnp.zeros(n_states)
    v_final, _ = jax.lax.scan(iterate, v_init, None, length=500)
    return v_final

v_backprop = solve_backprop(jnp.array(theta_hat))
pi_backprop = compute_policy(v_backprop, jnp.array(theta_hat))

# Check if they match
methods_match = jnp.allclose(pi_implicit, pi_backprop, atol=1e-6)
print(f"  Solutions match: {methods_match}")

# Verification: check Bellman residual
residual = v_implicit - smooth_bellman_operator(v_implicit, jnp.array(theta_hat))
max_residual = float(jnp.max(jnp.abs(residual)))
converged = max_residual < 1e-6

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

mileage_k = np.arange(n_states) * 5

# Plot 1: Replacement policy - overlay both methods
ax = axes[0]
ax.plot(mileage_k, pi_implicit[:, 1] * 100, 'b-', linewidth=3, 
        label='implicit_diff (jaxopt)', alpha=0.8)
ax.plot(mileage_k, pi_backprop[:, 1] * 100, 'r--', linewidth=2, 
        label='backprop (unrolled)', alpha=0.7)

# Set y-axis to show the actual range
max_prob = float(pi_implicit[:, 1].max() * 100)
min_prob = float(pi_implicit[:, 1].min() * 100)

if max_prob < 1.0:
    ax.set_ylim(min_prob - 0.001, max_prob * 1.2)
else:
    ax.set_ylim(0, max_prob * 1.1)

ax.set_xlabel('Mileage since last replacement (thousands)', fontsize=12)
ax.set_ylabel('π(replace | s)  (%)', fontsize=12)
ax.set_title(f'Softmax Replacement Policy\nθ̂ = [RC={theta_hat[0]:.2f}, C1={theta_hat[1]:.2f}]',
            fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

# Add validation info
info_text = f'γ={gamma}\nMethods match: {methods_match}\nConverged: {converged}'
ax.text(0.02, 0.65, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen' if (methods_match and converged) else 'yellow', alpha=0.5))

# Plot 2: Loss evolution
ax = axes[1]

final_loss = losses[-1]
initial_loss = losses[0]
improvement = initial_loss - final_loss

ax.plot(losses, 'g-', linewidth=2.5)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('-log L(θ)', fontsize=12)
ax.set_title('Optimization Progress', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Mark start and end
ax.scatter([0], [initial_loss], s=100, color='red', zorder=10, label='Start')
ax.scatter([len(losses)-1], [final_loss], s=100, color='green', zorder=10, label='End')
ax.legend(fontsize=10)

# Add info
rate = improvement / len(losses)
ax.text(0.02, 0.02, f'Improving at ~{rate:.2f} per step',
        transform=ax.transAxes, fontsize=10, ha='left', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('estimation_results.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: estimation_results.png")

# Create animation
print(f"\n[6] Creating Animation")
print(f"  Animating {len(policy_history)} policy snapshots...")

from matplotlib.animation import FuncAnimation, FFMpegWriter

fig_anim, ax_anim = plt.subplots(1, 1, figsize=(10, 6))

def init_anim():
    ax_anim.set_xlim(0, 450)
    ax_anim.set_ylim(-0.01, max(1.5, policy_history[-1][1][:, 1].max() * 110))
    ax_anim.set_xlabel('Mileage since last replacement (thousands)', fontsize=12)
    ax_anim.set_ylabel('π(replace | s)  (%)', fontsize=12)
    ax_anim.grid(True, alpha=0.3)
    return []

def update_anim(frame_idx):
    ax_anim.clear()
    ax_anim.set_xlim(0, 450)
    ax_anim.set_ylim(-0.01, max(1.5, policy_history[-1][1][:, 1].max() * 110))
    ax_anim.set_xlabel('Mileage since last replacement (thousands)', fontsize=12)
    ax_anim.set_ylabel('π(replace | s)  (%)', fontsize=12)
    ax_anim.grid(True, alpha=0.3)
    
    step_num, pi, theta_frame = policy_history[frame_idx]
    
    # Plot current policy
    ax_anim.plot(mileage_k, pi[:, 1] * 100, 'b-', linewidth=3, alpha=0.8)
    
    # Add final policy as reference (faint)
    if frame_idx < len(policy_history) - 1:
        ax_anim.plot(mileage_k, policy_history[-1][1][:, 1] * 100, 'gray', 
                    linewidth=1, linestyle='--', alpha=0.3, label='Final')
    
    # Add info
    ax_anim.set_title(f'Policy Evolution During Optimization\\nStep {step_num}: θ=[RC={theta_frame[0]:.2f}, C1={theta_frame[1]:.2f}]',
                     fontsize=13, fontweight='bold')
    
    loss_at_step = losses[step_num] if step_num < len(losses) else losses[-1]
    ax_anim.text(0.98, 0.98, f'Iteration: {step_num}\\n-log L: {loss_at_step:.1f}',
                transform=ax_anim.transAxes, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    return []

# Create animation
anim = FuncAnimation(fig_anim, update_anim, init_func=init_anim,
                    frames=len(policy_history), interval=100, blit=True)

# Try to save as video (requires ffmpeg)
try:
    writer = FFMpegWriter(fps=10, bitrate=1800)
    anim.save('policy_evolution.mp4', writer=writer)
    print(f"  ✓ Saved: policy_evolution.mp4")
except Exception as e:
    print(f"  ⚠ Could not save video (ffmpeg not available): {e}")
    print(f"  Saving as GIF instead...")
    try:
        anim.save('policy_evolution.gif', writer='pillow', fps=10)
        print(f"  ✓ Saved: policy_evolution.gif")
    except Exception as e2:
        print(f"  ⚠ Could not save animation: {e2}")
        print(f"  (Install ffmpeg or pillow for animation export)")

plt.close(fig_anim)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"✓ Real Rust (1987) data from Madison Metro (1974-1985)")
print(f"✓ Smooth Bellman operator with implicit differentiation (jaxopt)")
print(f"✓ Bellman residual: {max_residual:.2e} (converged: {converged})")
print(f"\nModel: v(s) = log Σ_a exp(q(s,a)) with γ={gamma}")
print(f"\nEstimated θ̂ = [RC={theta_hat[0]:.2f}, C1={theta_hat[1]:.2f}]")
print(f"Published θ = [RC=9.75, C1=2.63] (Rust 1987)")
print(f"Error: {abs(theta_hat[0]-9.75)/9.75*100:.1f}%")
print(f"\nLoss improved: {initial_loss:.1f} → {final_loss:.1f} (Δ = {improvement:.1f})")
print("="*70)

plt.show()

