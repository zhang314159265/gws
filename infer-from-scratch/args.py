import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Inference tool.")
    parser.add_argument("--interactive", action="store_true", help="Whether to run the tool in interactive mode")
    return parser.parse_args()

