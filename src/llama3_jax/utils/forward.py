import jax.numpy as jnp
from llama3_jax.utils.mathematics import (
    precompute_freqs_cis,
    rms_norm,
)
from llama3_jax.blocks.transformer import transformer_block


def model_forward(params, inputs, config, cache=None, position=0):
    # Get batch dimensions

    B, T = inputs.shape

    # Convert input tokens to embeddings

    h = params["token_embedding"][inputs]

    # Compute freqs_cis for this forward pass
    freqs_cis = precompute_freqs_cis(
        config.model.embedding_dimension // config.model.number_attention_heads,
        config.model.maximum_sequence_length,
    )

    # Create causal mask
    mask = jnp.tril(
        jnp.ones(
            (config.model.maximum_sequence_length, config.model.maximum_sequence_length)
        )
    )

    mask = jnp.where(mask == 0, -1e9, 0.0)
    mask = mask.astype(h.dtype)
    mask = mask[None, None, :, :]

    # Process through transformer blocks
    new_caches = []
    for i, block in enumerate(params["transformer_blocks"]):
        layer_cache = cache[i] if cache is not None else None
        h, layer_cache = transformer_block(
            block,
            h,
            mask,
            freqs_cis,
            config.model.number_attention_heads,
            config.model.number_key_value_heads,
            layer_cache,
            position,
        )
        new_caches.append(layer_cache)

    # Final normalization and output projection
    h = rms_norm(h, params["norm_f"])
    logits = jnp.dot(h, params["output"])

    return logits, new_caches
