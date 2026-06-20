import numpy as np
def poisoned_update(honest_update,scale=10.0):
    """Sign-flip + amplify: tries to drag the model toward whitelisting fraud."""
    return -scale*honest_update
