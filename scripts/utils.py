import sys
import numpy as np

def get_total_size(obj, seen=None):
    # Keep track of already seen objects to avoid infinite recursion
    if seen is None:
        seen = set()
    
    # Return 0 if the object is already counted
    if id(obj) in seen:
        return 0

    # Mark the current object as seen
    seen.add(id(obj))

    # Handle numpy arrays specifically
    if isinstance(obj, np.ndarray):
        # Use the nbytes attribute for numpy arrays to get the size of the data buffer
        size = obj.nbytes + sys.getsizeof(obj)
    else:
        # Start with the size of the object itself
        size = sys.getsizeof(obj)

    # If the object is a dictionary, add the size of its keys and values
    if isinstance(obj, dict):
        size += sum(get_total_size(k, seen) + get_total_size(v, seen) for k, v in obj.items())
    # If the object is an iterable (like a list, tuple, set, etc.), add the size of its elements
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_total_size(i, seen) for i in obj)

    # Return the total computed size
    return size
