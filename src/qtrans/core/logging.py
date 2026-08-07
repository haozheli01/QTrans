"""QTrans core.logging helpers."""

import sys
import time


class time_logger:
    """A progress bar with timing information for loop-based tasks."""

    def __init__(self, total_cycles: int, routine_name: str) -> None:
        self.scale: int = 50
        self.total_cycles: int = total_cycles
        self.start: float = time.perf_counter()
        self.last_time: float = self.start
        self.routine_name: str = routine_name

    def step(self, current_cycle: int) -> None:
        if current_cycle == 0:
            print(f"Running {self.routine_name}".center(100, "-"))

        progress = int(current_cycle / self.total_cycles * self.scale)
        filled = "*" * progress
        remaining = "." * (self.scale - progress)
        percent = current_cycle / self.total_cycles * 100

        now = time.perf_counter()
        elapsed = now - self.start
        step_time = now - self.last_time
        self.last_time = now

        # Use average time per cycle for a more stable estimate
        if current_cycle > 0:
            avg_time = elapsed / current_cycle
            remaining_time = avg_time * (self.total_cycles - current_cycle)
        else:
            remaining_time = 0.0

        print(
            f"\r{percent:^3.0f}%[{filled}->{remaining}]"
            f"[total: {elapsed:.2f}s, step: {step_time:.2f}s, remaining: {remaining_time:.2f}s]",
            end="",
            flush=True,
        )

        if current_cycle >= self.total_cycles:
            print()  # Move to next line after completion
            print(f"{self.routine_name} has run successfully!".center(100, "-"))
        sys.stdout.flush()
