import sys
import numpy as np
import json

# Function to get the total size of an object in bytes
def get_total_size(obj, seen=None):
    # Keep track of already seen objects to avoid infinite recursion
    if seen is None:
        seen = set()

    obj_id = id(obj)
    # Return 0 if the object is already counted
    if obj_id in seen:
        return 0

    # Mark the current object as seen
    seen.add(obj_id)

    # Start with the size of the object itself
    size = sys.getsizeof(obj)

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # If it's an object array, recursively calculate the size of each element
            size += sum(get_total_size(item, seen) for item in obj.flat)
        else:
            # Use nbytes for regular numpy arrays to get the size of the data buffer
            size += obj.nbytes

    # If the object is a dictionary, add the size of its keys and values
    if isinstance(obj, dict):
        size += sum(get_total_size(k, seen) + get_total_size(v, seen) for k, v in obj.items())
    # If the object is an iterable (like a list, tuple, set, etc.), add the size of its elements
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_total_size(i, seen) for i in obj)

    # Return the total computed size
    return size

# Custom JSON encoder for numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)