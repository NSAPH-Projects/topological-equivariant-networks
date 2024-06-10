import argparse

import parser_utils
from qm9.utils import process_qm9_dataset
from utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_utils.add_common_arguments(parser)
    parsed_args = parser.parse_args()
    parsed_args = parser_utils.add_common_derived_arguments(parsed_args)

    set_seed(parsed_args.seed)
    process_qm9_dataset(parsed_args)
