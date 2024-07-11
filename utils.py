import hashlib
import json


def args_to_hash(args: dict):
    def is_json_serializable(value):
        try:
            json.dumps(value, sort_keys=True)
            return True
        except (TypeError, OverflowError):
            return False

    def serialize_value(value):
        if isinstance(value, list):
            return sorted(value)
        if not is_json_serializable(value):
            return str(value)
        return value

    # Convert and sort all arguments, excluding the key "lifters"
    args_dict = {
        k: serialize_value(v)
        for k, v in args.items()
        if k != "lifter" and serialize_value(v) is not None
    }

    # Sort the dictionary by keys to ensure consistent order
    sorted_args_dict = dict(sorted(args_dict.items()))

    # Convert to a JSON string
    args_str = json.dumps(sorted_args_dict, sort_keys=True)

    # Compute the hash
    args_hash = hashlib.md5(args_str.encode()).hexdigest()

    return args_hash
