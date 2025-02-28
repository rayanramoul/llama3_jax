import jax.numpy as jnp
import jax


def feed_forward(params, x):
    w3_ = jnp.dot(x, params["w3"])

    # SwiGLU(a,b)=SiLU(a)⊙b
    activated = jax.nn.silu(w3_)

    w1_ = jnp.dot(x, params["w1"])

    combined = activated * w1_

    # Final output projection with w2
    output = jnp.dot(combined, params["w2"])

    return output
