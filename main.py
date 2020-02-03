import argparse
from config import Config

from agents.mcvae import MCVAE


def main():
    config = Config()

    # Create the Agent and pass all the configuration to it then run it..
    agent = MCVAE(config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()