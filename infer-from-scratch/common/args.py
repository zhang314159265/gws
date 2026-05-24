import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Inference tool.")
    parser.add_argument("--interactive", action="store_true", help="Whether to run the tool in interactive mode")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--disable-cudagraphs", action="store_true", help="Whether to disable cudagraphs")
    return parser.parse_args()
