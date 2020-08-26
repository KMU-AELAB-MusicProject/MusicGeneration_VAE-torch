import argparse
from config import Config

from agent.barGen2 import BarGen


def main():
    config = Config()

    # Create the Agent and pass all the configuration to it then run it..
    agent = BarGen(config)
    agent.run()


if __name__ == '__main__':
    main()
