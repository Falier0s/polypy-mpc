class StateSystem:
    """A simple wrapper for system matrices and related operations."""

    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
