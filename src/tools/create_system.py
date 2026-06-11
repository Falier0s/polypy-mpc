import numpy as np

from .statesystem import StateSystem


def create_system(sys, sampling_time=None):
    """Create the system for the cMPC class

    Args:
        sys (dict): The dictionnary which contains matrices A,B and C
        sampling_time (None/Float, optional): The sampling time. Defaults to None.

    Returns:
        scipy.signal.StateSystem: a State Space system
    """
    if len(sys["A"].shape) == 2:
        sys["A"] = sys["A"][:, :, np.newaxis]

    if len(sys["B"].shape) == 2:
        sys["B"] = sys["B"][:, :, np.newaxis]

    if len(sys["C"].shape) == 2:
        sys["C"] = sys["C"][:, :, np.newaxis]
    if sampling_time is None or sampling_time == 0:
        final_system = StateSystem(sys["A"], sys["B"], sys["C"])
        return final_system
    identity_tensor = np.zeros(
        (sys["A"].shape[0], sys["A"].shape[1], sys["A"].shape[2])
    )
    identity_tensor[:, :, 0] = np.eye(sys["A"].shape[2])

    final_system = StateSystem(
        sampling_time * sys["A"] + identity_tensor,
        sampling_time * sys["B"],
        sys["C"],
    )
    return final_system
