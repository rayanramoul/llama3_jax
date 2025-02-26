import jax.numpy as jnp


def rms_norm(x, weight, eps=1e-5):
    """Root mean square norm helps the training
    stabilize and that weights don't become too large."""
    # Calculate variance across last dimension
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

    # Normalize and scale
    return x * weight * jnp.reciprocal(jnp.sqrt(variance + eps))


def precompute_freqs_cis(dim: int, end: int, theta: float = 1000.0):
    """We first create table of rotation factors using a range of frequencies.

    This means each token gets it own unique rotation angle."""
    # Generate frequency bands
    freqs = 1.0 / (theta ** (jnp.arange(0, dim // 2, dtype=jnp.float32) / dim))

    # Generate position indices
    t = jnp.arange(end, dtype=jnp.float32)

    # Compute outer product
    freqs = jnp.outer(t, freqs)

    # Convert to complex exponential
    return jnp.complex64(jnp.exp(1j * freqs))


def apply_rotary_embedding(xq, xk, freqs_cis):
    # Reshape inputs for complex multiplication
    xq_r, xk_r = (
        jnp.reshape(xq, (*xq.shape[:-1], -1, 2)),
        jnp.reshape(xk, (*xk.shape[:-1], -1, 2)),
    )

    # Convert to complex numbers
    xq_complex = jnp.complex64(xq_r[..., 0] + 1j * xq_r[..., 1])
    xk_complex = jnp.complex64(xk_r[..., 0] + 1j * xk_r[..., 1])

    # Reshape frequencies for broadcasting
    freq_cis = jnp.reshape(freqs_cis, freq_cis.shape[0], 1, freq_cis.shape[1])

    # Apply rotation through complex multiplication
    xq_out = xq_complex * freq_cis
    xk_out = xk_complex * freq_cis

    # Convert back to real numbers and reshape
    xq = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(xq.shape)
    xk = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(xk.shape)

    return xq, xk
