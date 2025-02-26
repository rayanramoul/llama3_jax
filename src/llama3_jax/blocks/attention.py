"""
The goal is to implement Group-Query Attention.
Grouped Query Attention (GQA) is an optimized version of Multi-Head Attention 
that improves efficiency by sharing key and value representations among multiple query heads.
This reduces computational overhead and memory usage, 
enabling faster inference and better scaling for transformer models. 
At it's core, it's just self-attention but with some modification.
"""

def attention(params, x, mask, freqs_cis, n_heads, n_kv_heads, cache=None, position=0):
    # Input dimensions
    B, T, C = x.shape
    # B: Batch
    # T: Times, Sequence Length
    # C: Channels or Embedding dimension
    # Get input dimensions
    head_dim = C // n_heads
    
    # Project inputs to queries, keys, and values
    q = jnp.dot(x, params['wq']).reshape(B, T, n_heads, head_dim)
    k = jnp.dot(x, params['wk']).reshape(B, T, n_kv_heads, head_dim)
    v = jnp.dot(x, params['wv']).reshape(B, T, n_kv_heads, head_dim)
    
    # Apply rotary embeddings
    q, k = apply_rotary_emb(q, k, freqs_cis[position:position + T])
    
    # Handle cache for inference
    if cache is not None:
        k = jnp.concatenate([cache[0], k], axis=-1])
        v = jnp.concatenate([cache[1], v], axis=-1])
    new_cache = (k, v)
    
    # Repeat k/v heads for grouped-query attention
    k = repeat_kv(k, n_heads // n_kv_heads)
    v = repeat_kv(v, n_heads // n_kv_heads)
    
    # Compute attention scores and apply attention
    q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    
    # Apply attention mask if provided
    if mask is not None:
        scores = scores + mask[:, :, :T, :T]
    
    # Compute attention weights and final output
    scores = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(scores, v)
    output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)
    
    return jnp.dot(output, params['wo']), new_cache
