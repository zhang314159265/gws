import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Inference tool.")
    parser.add_argument("--interactive", action="store_true", help="Whether to run the tool in interactive mode")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    return parser.parse_args()
