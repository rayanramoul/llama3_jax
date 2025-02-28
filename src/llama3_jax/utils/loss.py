from llama3_jax.utils.forward import model_forward
import jax.numpy as jnp
import jax


def compute_cross_entroy_loss(params, batch, cfg):
    # Split batch into inputs and targets
    inputs, targets = batch
    # Forward pass to get logits
    (logits,) = model_forward(params, inputs, cfg)
    # Reshape for loss computation
    logits = logits.reshape(-1, cfg.vocab_size)
    targets = targets.reshape(-1)
    # Calculate negative log likelihood
    loss = -jnp.mean(
        jnp.take_along_axis(jax.nn.log_softmax(logits), targets[:, None], axis=1)
    )
    return loss
