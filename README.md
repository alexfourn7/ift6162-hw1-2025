# IFT6162 – Homework 1

This repository contains three programming assignments for **IFT6162** covering dynamic programming, optimal control, and numerical methods.

## 📋 Grading and Time Allocation

**Please read [`GRADING_AND_TIME_ALLOCATION.md`](GRADING_AND_TIME_ALLOCATION.md) first** for:
- Detailed grading rubric and point distribution
- Time allocation recommendations
- Submission requirements and expectations
- Debugging tips and common pitfalls

### Quick Overview

| Problem | Weight | Time | Files |
|---------|--------|------|-------|
| **Supermarket Refrigeration** (Trajectory Optimization MPC) | 45% | 12-15h | `supermarket_refrigeration/` |
| **Bus Engine Replacement** (Smooth Bellman & NFXP) | 35% | 8-10h | `bus_engine_replacement/` |
| **Projection Methods** (Collocation & Galerkin) | 20% | 5-7h | `projection_methods_assignment/` |

**Total estimated time: 25-32 hours**

## 🚀 Quick Start

### Environment Setup

Create a virtual environment at the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib jax jaxlib jaxopt optax chex absl-py
```

### Running the Assignments

Each problem has detailed instructions in its respective `question.md` file:

1. **Supermarket Refrigeration:**
   ```bash
   cd supermarket_refrigeration
   python3 trajectory_optimization.py          # Quick test (30 min)
   python3 trajectory_optimization.py --full   # Full evaluation (4 hours)
   ```

2. **Bus Engine Replacement:**
   ```bash
   cd bus_engine_replacement
   python3 bus_replacement.py                  # Runs in 2-3 minutes
   ```

3. **Projection Methods:**
   ```bash
   cd projection_methods_assignment
   python3 timber_projection.py                # Test implementations
   python3 contraction_investigation.py        # Verify contraction properties
   ```

## 📝 Submission Requirements

**You are NOT required to write formal multi-page reports.** For each problem:

- ✅ Submit your completed implementation files
- ✅ Include generated plots and output metrics
- ✅ For Problem 1 only: Add brief commentary (1-2 paragraphs) on results
- ❌ No lengthy derivations or literature reviews needed

## 📚 Problem Descriptions

### Problem 1: Supermarket Refrigeration (45%)
Implement multiple shooting Model Predictive Control for a hybrid refrigeration system. You'll coordinate compressors and valves to minimize energy while respecting temperature and pressure constraints. Uses JAX for autodiff and SLSQP for constrained optimization.

### Problem 2: Bus Engine Replacement (35%)
Implement Harold Zurcher's bus engine replacement model using smooth Bellman equations. Estimate cost parameters via maximum likelihood with the NFXP algorithm and implicit differentiation through fixed-point solvers.


### Problem 3: Projection Methods (20%)
Implement collocation and Galerkin methods for continuous-state dynamic programming. Solve a timber harvesting problem and investigate when parametric value iteration converges.

## 🔧 Debugging Tips

- **JAX issues:** Use `jnp` instead of `np` inside JIT-compiled functions
- **Constraint violations:** Check that temperature and pressure bounds are being satisfied
- **Slow convergence:** Verify warm-starting and check that gradients are being computed correctly
- **Numerical instability:** Use log-space operations (logsumexp) for probabilities

## 📖 Additional Resources

Each subdirectory contains:
- `question.md`: Detailed problem description and mathematical background
- Template files with `TODO` markers indicating what to implement
- Helper functions and simulation code (already implemented)

See [`GRADING_AND_TIME_ALLOCATION.md`](GRADING_AND_TIME_ALLOCATION.md) for comprehensive guidance on each problem.
