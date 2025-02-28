from llama3_jax.blocks.attention import attention
from llama3_jax.utils.mathematics import rms_norm
from llama3_jax.blocks.feed_forward import feed_forward


def transformer_block(
    params, x, mask, freqs_cis, n_heads, n_kv_heads, cache=None, position=0
):
    # Apply attention with normalization and residual connection
    attn_output, new_cache = attention(
        params["attention"],
        rms_norm(x, params["attention_norm"]),
        mask,
        freqs_cis,
        n_heads,
        n_kv_heads,
        cache,
        position,
    )

    # First residual connection
    h = x + attn_output

    # Apply feed-forward network with normalization and residual
    ffn_output = feed_forward(params["ffn"], rms_norm(h, params["ffn_norm"]))

    # Second residual connection

    out = h + ffn_output

    return out, new_cache
