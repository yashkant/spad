import numpy as np


class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n,**kwargs)


class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """
    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                    1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f

class LambdaStepScheduler:
    def __init__(self, decay_steps, decay_factors, warmup_steps=-1, f_min=0.1, f_max=1.0, verbosity_interval=0):
        assert len(decay_steps) == len(decay_factors)
        self.decay_steps = decay_steps
        self.decay_factors = decay_factors
        self.verbosity_interval = verbosity_interval

        # add inital step to decay 
        self.decay_steps = self.decay_steps + [np.inf]
        self.decay_factors = [1.0] + self.decay_factors

        # add linear warmup
        self.lr_warm_up_steps = warmup_steps
        self.f_min = f_min
        self.f_max = f_max

        # warmup steps must be smaller than first decay step
        assert self.lr_warm_up_steps < self.decay_steps[0]

        # print scheduler config complete
        print(f"LambdaStepScheduler: decay_steps: {self.decay_steps}, decay_factors: {self.decay_factors}, "
              f"warm_up_steps: {self.lr_warm_up_steps}, f_min: {self.f_min}, f_max: {self.f_max}")

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}")
        
        if n < self.lr_warm_up_steps:
            f = (self.f_max - self.f_min) / self.lr_warm_up_steps * n + self.f_min
            return f
        else:
            for i, ds in enumerate(self.decay_steps):
                if n < ds:
                    return self.decay_factors[i]
        return self.decay_factors[-1]

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


if __name__ == "__main__":
    # lr_scheduler = LambdaLinearScheduler(
    #     warm_up_steps=[1000],
    #     cycle_lengths=[10000000000000],
    #     f_start=[1.0],
    #     f_max=[1.0],
    #     f_min=[1.0],
    # )

    lr_scheduler = LambdaStepScheduler(
        decay_steps=[2000, 3000, 4000],
        decay_factors=[0.1, 0.01, 0.001],
    )

    for i in range(0, 10000, 100):
        print(f"step {i}: {lr_scheduler(i)}")