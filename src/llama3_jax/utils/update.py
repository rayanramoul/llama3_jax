import jax
from llama3_jax.utils.loss import compute_cross_entroy_loss

"""
    Jax first traces your function to build an optimized computation graph. This tracing happens the first time the function is called and converts the Python code into machine code.
Because of this tracing, any side effects like the print statement; are only executed during the initial tracing. Once the function is compiled, other remaining calls use the compiled version, and you might not see the print output every time.
"""


@jax.tree_util.Partial(jax.jit, static_argnames=["config"])
def update_step(params, batch, config):
    # Compute both loss and gradients in a single pass using value_and_grad
    # This is more efficient than computing them separately
    loss, grads = jax.value_and_grad(compute_cross_entroy_loss)(params, batch, config)

    # Update parameters using gradient descent
    # jax.tree.map applies the update rule to each parameter in the model
    # The lambda function implements: p_new = p_old - learning_rate * gradient
    params = jax.tree.map(lambda p, g: p - config.learning_rate * g, params, grads)

    # Return updated parameters and the loss value for monitoring training
    return params, loss
