
import jax
import jax.numpy as jnp
from jax import random

@jax.jit
def calculate_local_field(spins, external_field, coupling_j):
    """Computes effective local field: h_i + J * sum(neighbors)."""
    neighbors = (
        jnp.roll(spins, 1, axis=0) + jnp.roll(spins, -1, axis=0) +
        jnp.roll(spins, 1, axis=1) + jnp.roll(spins, -1, axis=1)
    )
    return external_field + (coupling_j * neighbors)

@jax.jit
def metropolis_step_checkerboard(key, spins, external_field, coupling_j, temp):
    """Parallel Metropolis-Hastings update using Checkerboard decomposition."""
    rows, cols = spins.shape
    i_idx, j_idx = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols), indexing='ij')
    checkerboard = (i_idx + j_idx) % 2
    
    def update_phase(current_spins, phase_key, phase_mask):
        local_field = calculate_local_field(current_spins, external_field, coupling_j)
        dE = 2.0 * current_spins * local_field
        probability = jnp.exp(-dE / temp)
        random_vals = random.uniform(phase_key, shape=current_spins.shape)
        should_flip = (dE < 0) | (random_vals < probability)
        return jnp.where(phase_mask & should_flip, -current_spins, current_spins)

    k1, k2 = random.split(key)
    spins = update_phase(spins, k1, checkerboard == 0)
    spins = update_phase(spins, k2, checkerboard == 1)
    return spins

def anneal_layer(weights, steps=2500, t_start=5.0, t_end=0.1, coupling=0.7):
    """Main Annealing Loop."""
    mag = jnp.abs(weights)
    field = (mag - jnp.mean(mag)) / jnp.std(mag)
    
    key = random.PRNGKey(42)
    spins = random.choice(key, jnp.array([-1.0, 1.0]), shape=weights.shape)
    temps = jnp.linspace(t_start, t_end, steps)
    
    step_fn = metropolis_step_checkerboard
    for i, t in enumerate(temps):
        key, subkey = random.split(key)
        spins = step_fn(subkey, spins, field, coupling, t)
            
    return spins, field
