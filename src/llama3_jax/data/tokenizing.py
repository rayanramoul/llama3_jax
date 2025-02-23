import tiktoken
import jax.numpy as jnp

encoder = tiktoken.get_encoding("gpt2")


def get_tokens(text: str) -> jnp.DeviceArray:
    """Function to encode text into tokens."""
    return jnp.array(encoder.encode(text))
