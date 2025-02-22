import omegaconf
from rich import print
import hydra
import argparse


@hydra.main(config_path="configs", config_name="config")
def train(cfg: omegaconf.DictConfig):
    print("Currently Training.")
    print(cfg)


if __name__ == "__main__":
    train()
