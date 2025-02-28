import tiktoken
import jax.numpy as jnp
import jax

encoder = tiktoken.get_encoding("gpt2")


def get_tokens(text: str) -> jax.Array:
    """Function to encode text into tokens."""
    return jnp.array(encoder.encode(text))
