from jax import lax, vmap
import jax


def get_batch(key, data, batch_size, seq_len):
    """
    vmap is like a vectorized loop;
    it takes a function that processes a single index
    (using lax.dynamic_slice to get a sequence of tokens)
    and applies it to every element in our array of indices.
    This means our input sequences (x) and corresponding target sequences
    (y, which are shifted by one token for next-word prediction)
    are created in one go.
    """
    # Generate random starting indices
    ix = jax.random.randint(key, (batch_size,), 0, len(data) - seq_len)

    # Vectorized operation to get input and target sequences
    x = vmap(lambda i: lax.dynamic_slice(data, (i,), (seq_len,)))(ix)
    y = vmap(lambda i: lax.dynamic_slice(data, (i + 1,), (seq_len,)))(ix)

    return x, y
