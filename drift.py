import math
from .config import MIN_DDM_SAMPLES


class DDM:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.errors = 0
        self.p = 0.0
        self.s = 0.0
        self.best_p = float("inf")
        self.best_s = float("inf")
        self.warning = False

    def update(self, error):
        error = 1 if bool(error) else 0
        self.n += 1
        self.errors += error
        self.p = self.errors / self.n
        self.s = math.sqrt(max(self.p * (1.0 - self.p), 0.0) / self.n)

        if self.n < MIN_DDM_SAMPLES:
            return False

        if self.p + self.s < self.best_p + self.best_s:
            self.best_p, self.best_s = self.p, self.s

        # Warning threshold can be used for diagnostics; drift threshold
        # triggers adaptation.
        self.warning = self.p + self.s > self.best_p + 2 * self.best_s
        drift = (
            self.p + self.s > self.best_p + 3 * self.best_s
            and self.p > 0.0
        )
        return drift
