import jax
import omegaconf
from rich import print
import hydra
import argparse
from llama3_jax.utils.weights import init_model_parameters


@hydra.main(config_path="configs", config_name="train")
def train(cfg: omegaconf.DictConfig):
    # Jax uses pseudo random key which get derived for each operation
    key = jax.random.PRNGKey(cfg.seed)

    print("Currently Training.")
    print(cfg)
    init_model_parameters(
        key,
        vocab_size=76,
        dim=cfg.model.embedding_dimension,
        number_layers=cfg.model.number_transformer_layers,
        number_heads=cfg.model.number_attention_heads,
        number_kv_heads=cfg.model.number_key_value_heads,
    )


if __name__ == "__main__":
    train()
