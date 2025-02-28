import random
import os

import hydra
from cloudpathlib import AnyPath
import loguru
import jax
import omegaconf
import jax.numpy as jnp

from llama3_jax.utils.weights import init_model_parameters
from llama3_jax.data.tokenizing import get_tokens
from llama3_jax.utils.update import update_step
from llama3_jax.data.batching import get_batch

CWD = AnyPath(os.getcwd())


@hydra.main(config_path="configs", config_name="train")
def train(cfg: omegaconf.DictConfig):
    # Jax uses pseudo random key which get derived for each operation
    key = jax.random.PRNGKey(cfg.seed)

    print("Currently Training.")
    print(cfg)
    params_state = init_model_parameters(
        key,
        vocab_size=76,
        dim=cfg.model.embedding_dimension,
        number_layers=cfg.model.number_transformer_layers,
        number_heads=cfg.model.number_attention_heads,
        number_kv_heads=cfg.model.number_key_value_heads,
    )
    loguru.logger.info(f"Currently in directory {CWD}")
    data_path = CWD / "data" / "shakespeare.txt"
    with open(data_path) as f:
        text = f.read()
    tokens = get_tokens(text)
    tokens_array = jnp.array(tokens)
    epoch_losses = []
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0

        for step in range(cfg.steps_per_epoch):
            # Generate new random keys for reproducibility

            key, batch_key = jax.random.split(key)

            # Sample random batch of sequences
            batch = get_batch(batch_key, tokens_array, cfg.batch_size, cfg.max_seq_len)

            # Forward pass, compute loss and update parameters
            params_state, loss = update_step(params_state, batch, cfg)

            # loss for epoch average
            epoch_loss += loss

            if step % 100 == 0:
                print(
                    f"epoch {epoch + 1}, step {step}/{cfg.steps_per_epoch}: loss = {loss:.4f}"
                )

        avg_epoch_loss = epoch_loss / cfg.steps_per_epoch

        epoch_losses.append(avg_epoch_loss)

        print(f"\nepoch {epoch + 1} | average loss: {avg_epoch_loss:.4f}")


if __name__ == "__main__":
    train()
