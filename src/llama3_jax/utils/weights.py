import math
import jax


def init_weights(
    random_key: jax.Array, shape: tuple[int], scale: float = None
) -> jax.Array:
    # calculate default scale if none
    if not scale:
        scale = 1.0 / math.sqrt(shape[0])
    print("Type of random key:", type(random_key))
    weights = jax.random.normal(random_key, shape) * scale
    print("Type of weights:", type(weights))
    return weights


def init_attention_layer_weights(
    key: jax.Array, dim: int, number_heads: int, number_kv_heads: int
) -> dict[str, jax.Array]:
    # 4 keys one for each of: key, query, value, output
    keys = jax.random.split(key, 4)
    print(f"Keys: {keys}")
    head_dim = dim // number_heads
    weights = {
        "wq": init_weights(keys[0], (dim, number_heads, head_dim)),
        "wk": init_weights(keys[1], (dim, number_kv_heads, head_dim)),
        "wv": init_weights(keys[2], (dim, number_kv_heads, head_dim)),
        "wo": init_weights(keys[3], (dim, number_heads, head_dim)),
    }
    return weights


def init_ffn_layer_weights(key: jax.Array, dim: int) -> dict[str, jax.Array]:
    keys = jax.random.split(key)
    return {
        "w1": init_weights(keys[0], (dim, 4 * dim)),  # first projection
        "w2": init_weights(keys[1], (4 * dim, dim)),  # second projection
        "w3": init_weights(keys[2], (dim, 4 * dim)),  # first projection
    }


def transformer_block_weights(
    key: jax.Array, dim: int, number_heads: int, number_kv_heads: int
) -> dict[str, dict[str, jax.Array] | jax.Array]:
    keys = jax.random.split(key, 2)
    return {
        "attention": init_attention_layer_weights(
            keys[0], dim, number_heads, number_kv_heads
        ),
        "ffn": init_ffn_layer_weights(keys[1], dim),
        "attention_norm": init_weights(
            key, (dim,), scale=1.0
        ),  # Pre-attention normalization
        "ffn_norm": init_weights(key, (dim,), scale=1.0),  # Pre-ffn normalization
    }


def init_model_parameters(
    key: jax.Array,
    vocab_size: int,
    dim: int,
    number_layers: int,
    number_heads: int,
    number_kv_heads: int,
) -> dict[str, jax.Array | list[dict[str, dict[str, jax.Array] | jax.Array]]]:
    keys = jax.random.split(key, 4)
    params = {
        "token_embedding": init_weights(keys[0], (vocab_size, dim)),
        "norm_f": init_weights(keys[1], (dim,), scale=1.0),  # Final normalization
        "output": init_weights(keys[2], (dim, vocab_size)),
    }
    # initialize transformer blocks
    transformers_blocks_keys = jax.random.split(keys[3], number_layers)
    params["transformer_blocks"] = [
        transformer_block_weights(key, dim, number_heads, number_kv_heads)
        for key in transformers_blocks_keys
    ]
    return params
