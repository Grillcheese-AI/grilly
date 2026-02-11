"""
Learning Rate Schedulers

Implements various learning rate scheduling strategies to match PyTorch's API.
"""

import math


class LRScheduler:
    """
    Base class for learning rate schedulers.

    All schedulers should inherit from this class and implement the get_lr() method.
    """

    def __init__(self, optimizer, last_epoch=-1):
        """
        Initialize base scheduler.

        Args:
            optimizer: Wrapped optimizer
            last_epoch: The index of last epoch (default: -1)
        """
        self.optimizer = optimizer

        # Store initial learning rates
        if not isinstance(optimizer.param_groups, list):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")

        self.base_lrs = []
        for i, group in enumerate(optimizer.param_groups):
            if "lr" not in group:
                raise KeyError(
                    f"param 'lr' is not specified in param_groups[{i}] when resuming an optimizer"
                )
            self.base_lrs.append(group["lr"])

        self.last_epoch = last_epoch
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return self._last_lr

    def get_lr(self):
        """
        Compute learning rate using chainable form of the scheduler.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def step(self, epoch=None):
        """
        Perform a scheduler step.

        Args:
            epoch: Optional epoch number to use instead of incrementing
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        # Get learning rates for this epoch
        lrs = self.get_lr()

        # Update optimizer learning rates
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class StepLR(LRScheduler):
    """
    Decays the learning rate by gamma every step_size epochs.

    Matches torch.optim.lr_scheduler.StepLR
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        """
        Initialize StepLR scheduler.

        Args:
            optimizer: Wrapped optimizer
            step_size: Period of learning rate decay
            gamma: Multiplicative factor of learning rate decay (default: 0.1)
            last_epoch: The index of last epoch (default: -1)
        """
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current epoch."""
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]


class CosineAnnealingLR(LRScheduler):
    """
    Set the learning rate using a cosine annealing schedule.

    Matches torch.optim.lr_scheduler.CosineAnnealingLR
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        """
        Initialize CosineAnnealingLR scheduler.

        Args:
            optimizer: Wrapped optimizer
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate (default: 0)
            last_epoch: The index of last epoch (default: -1)
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate using cosine annealing."""
        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.

    Matches torch.optim.lr_scheduler.ReduceLROnPlateau
    """

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    ):
        """
        Initialize ReduceLROnPlateau scheduler.

        Args:
            optimizer: Wrapped optimizer
            mode: One of 'min' or 'max'. In 'min' mode, lr will be reduced when
                  the quantity monitored has stopped decreasing (default: 'min')
            factor: Factor by which the learning rate will be reduced (default: 0.1)
            patience: Number of epochs with no improvement after which learning rate
                     will be reduced (default: 10)
            threshold: Threshold for measuring the new optimum (default: 1e-4)
            threshold_mode: One of 'rel', 'abs' (default: 'rel')
            cooldown: Number of epochs to wait before resuming normal operation
                     after lr has been reduced (default: 0)
            min_lr: A lower bound on the learning rate (default: 0)
            eps: Minimal decay applied to lr (default: 1e-8)
        """
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        if not isinstance(optimizer.param_groups, list):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}"
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Reset num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        """
        Perform a scheduler step based on metric.

        Args:
            metrics: The metric to monitor
            epoch: Optional epoch number
        """
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        """Reduce learning rate."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr

    @property
    def in_cooldown(self):
        """Check if scheduler is in cooldown period."""
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        """Check if metric 'a' is better than 'best'."""
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon
        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon
        else:  # mode == 'max' and threshold_mode == 'abs'
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        """Initialize comparison function."""
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = float("inf")
        else:  # mode == 'max'
            self.mode_worse = -float("inf")

    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )


class OneCycleLR(LRScheduler):
    """
    Sets the learning rate according to the 1cycle learning rate policy.

    Matches torch.optim.lr_scheduler.OneCycleLR
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        epochs=None,
        steps_per_epoch=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        last_epoch=-1,
    ):
        """
        Initialize OneCycleLR scheduler.

        Args:
            optimizer: Wrapped optimizer
            max_lr: Upper learning rate boundary in the cycle
            total_steps: Total number of steps in the cycle (optional)
            epochs: Number of epochs to train for (optional)
            steps_per_epoch: Number of steps per epoch (optional)
            pct_start: Percentage of the cycle spent increasing the learning rate (default: 0.3)
            anneal_strategy: Specifies the annealing strategy: 'cos' or 'linear' (default: 'cos')
            cycle_momentum: If True, momentum is cycled inversely (default: True)
            base_momentum: Lower momentum boundary in the cycle (default: 0.85)
            max_momentum: Upper momentum boundary in the cycle (default: 0.95)
            div_factor: Determines the initial learning rate via initial_lr = max_lr/div_factor (default: 25)
            final_div_factor: Determines the minimum learning rate via min_lr = initial_lr/final_div_factor (default: 1e4)
            last_epoch: The index of last epoch (default: -1)
        """
        # Validate total_steps
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError(f"Expected positive integer total_steps, but got {total_steps}")
            self.total_steps = total_steps
        else:
            if epochs is None or steps_per_epoch is None:
                raise ValueError("You must define both epochs and steps_per_epoch")
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError(f"Expected positive integer epochs, but got {epochs}")
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError(
                    f"Expected positive integer steps_per_epoch, but got {steps_per_epoch}"
                )
            self.total_steps = epochs * steps_per_epoch

        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        # Validate pct_start
        if pct_start < 0 or pct_start > 1:
            raise ValueError(f"Expected float between 0 and 1 pct_start, but got {pct_start}")

        # Validate anneal_strategy
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                f"anneal_strategy must be one of 'cos' or 'linear', instead got {anneal_strategy}"
            )
        elif anneal_strategy == "cos":
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == "linear":
            self.anneal_func = self._annealing_linear

        # Initialize learning rate variables
        self.max_lrs = self._format_param("max_lr", optimizer, max_lr)
        self.initial_lrs = [max_lr / div_factor for max_lr in self.max_lrs]
        self.min_lrs = [initial_lr / final_div_factor for initial_lr in self.initial_lrs]

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if "momentum" not in optimizer.defaults and "betas" not in optimizer.defaults:
                raise ValueError(
                    "optimizer must support momentum or betas with cycle_momentum option enabled"
                )

            self.use_beta1 = "betas" in optimizer.defaults
            self.max_momentums = self._format_param("max_momentum", optimizer, max_momentum)
            self.base_momentums = self._format_param("base_momentum", optimizer, base_momentum)

        super().__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Format parameter to be a list per parameter group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} values for {name}, got {len(param)}"
                )
            return list(param)
        else:
            return [param] * len(optimizer.param_groups)

    def _annealing_cos(self, start, end, pct):
        """Cosine annealing from start to end as pct goes from 0.0 to 1.0."""
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        """Linear annealing from start to end as pct goes from 0.0 to 1.0."""
        return (end - start) * pct + start

    def get_lr(self):
        """Compute learning rate at current step."""
        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError(
                f"Tried to step {step_num + 1} times. The specified number of total steps is {self.total_steps}"
            )

        for initial_lr, max_lr, min_lr in zip(self.initial_lrs, self.max_lrs, self.min_lrs):
            if step_num <= self.step_size_up:
                # Annealing from initial_lr to max_lr
                pct = step_num / self.step_size_up
                lr = self.anneal_func(initial_lr, max_lr, pct)
            else:
                # Annealing from max_lr to min_lr
                pct = (step_num - self.step_size_up) / self.step_size_down
                lr = self.anneal_func(max_lr, min_lr, pct)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                if step_num <= self.step_size_up:
                    # Annealing from max_momentum to base_momentum (inverse of lr)
                    pct = step_num / self.step_size_up
                    momentum = self.anneal_func(max_momentum, base_momentum, pct)
                else:
                    # Annealing from base_momentum to max_momentum
                    pct = (step_num - self.step_size_up) / self.step_size_down
                    momentum = self.anneal_func(base_momentum, max_momentum, pct)
                momentums.append(momentum)

            # Update momentum in optimizer
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                if self.use_beta1:
                    # For Adam-style optimizers, update beta1
                    betas = param_group["betas"]
                    param_group["betas"] = (momentum, betas[1])
                else:
                    # For SGD-style optimizers
                    param_group["momentum"] = momentum

        return lrs
