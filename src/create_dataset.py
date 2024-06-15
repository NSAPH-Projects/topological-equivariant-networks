import argparse
import os

from dotenv import load_dotenv

import parser_utils
from qm9.utils import process_qm9_dataset
from utils import set_seed

if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()
    if not os.getenv("STORAGE_PATH"):
        raise ValueError("STORAGE_PATH environment variable is not set.")

    parser = argparse.ArgumentParser()
    parser = parser_utils.add_common_arguments(parser)
    parsed_args = parser.parse_args()
    parsed_args = parser_utils.add_common_derived_arguments(parsed_args)

    set_seed(parsed_args.seed)
    process_qm9_dataset(parsed_args)
