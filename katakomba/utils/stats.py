import dataclasses
import math
import torch


@dataclasses.dataclass
class StatMean:
    # Compute using Welford'd Online Algorithm
    # Algo: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # Math: https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
    n: int = 0
    mu: float = 0
    m2: float = 0
    cumulative: bool = False

    def result(self):
        if self.n == 0:
            return None
        return self.mu

    def mean(self):
        return self.mu

    def std(self):
        if self.n < 1:
            return None
        return math.sqrt(self.m2 / self.n)

    def __sub__(self, other):
        assert isinstance(other, StatMean)
        n_new = self.n - other.n
        if n_new == 0:
            return StatMean(0, 0, 0)
        mu_new = (self.mu * self.n - other.mu * other.n) / n_new
        delta = other.mu - mu_new
        m2_new = self.m2 - other.m2 - (delta**2) * n_new * other.n / self.n
        return StatMean(n_new, mu_new, m2_new)

    def __iadd__(self, other):
        if isinstance(other, StatMean):
            other_n = other.n
            other_mu = other.mu
            other_m2 = other.m2
        elif isinstance(other, torch.Tensor):
            other_n = other.numel()
            other_mu = other.mean().item()
            other_m2 = ((other - other_mu) ** 2).sum().item()
        else:
            other_n = 1
            other_mu = other
            other_m2 = 0
        # See parallelized Welford in wiki
        new_n = other_n + self.n
        delta = other_mu - self.mu
        self.mu += delta * (other_n / max(new_n, 1))
        delta2 = other_mu - self.mu
        self.m2 += other_m2 + (delta2**2) * (self.n * other_n / max(new_n, 1))
        self.n = new_n
        return self

    def reset(self):
        if not self.cumulative:
            self.mu = 0
            self.n = 0

    def decay_cumulative(self, n=1e6):
        """Adjust sample size downwards to upweight recent samples"""
        if not self.cumulative:
            return
        if self.n > n:
            self.m2 *= n / self.n
            self.n = n

    def __repr__(self):
        return repr(self.result())
