import jax.numpy as jnp


def rms_norm(x, weight, eps=1e-5):
    """Root mean square norm helps the training
    stabilize and that weights don't become too large."""
    # Calculate variance across last dimension
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

    # Normalize and scale
    return x * weight * jnp.reciprocal(jnp.sqrt(variance + eps))
