"""QTrans core.logging helpers."""

from qtrans.core.constants import QTcts
import sys
import time


def log(message="", rank: int = 0, root: int = 0, **print_kwargs) -> None:
    """Print a message only from the selected root rank."""
    if rank == root:
        print(message, **print_kwargs)


def print_banner(
        message=None, banner_char: str = "#", width: int = QTcts.LOG_LENGTH,
        rank: int = 0, root: int = 0) -> None:
    """Print a fixed-width banner with an optional centered message.

    Args:
        message: Text embedded in the center. ``None`` or an empty string prints
            a banner containing only ``banner_char``.
        banner_char: Single character used to fill the banner.
        width: Total output width, including the optional message.
        rank: Rank of the caller.
        root: Rank allowed to print the banner.
    """
    if not isinstance(banner_char, str) or len(banner_char) != 1:
        raise ValueError("banner_char must be exactly one character.")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer.")

    message_text = "" if message is None else str(message).strip()
    if not message_text:
        log(banner_char * width, rank=rank, root=root)
        return

    centered_text = f" {message_text} "
    if len(centered_text) > width:
        raise ValueError(
            f"The banner message requires {len(centered_text)} characters, "
            f"which exceeds width={width}."
        )
    left_width = (width - len(centered_text)) // 2
    right_width = width - len(centered_text) - left_width
    log(
        banner_char * left_width + centered_text + banner_char * right_width,
        rank=rank,
        root=root,
    )


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
            print(f"Running {self.routine_name}".center(QTcts.LOG_LENGTH, "-"))

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
            print(f"{self.routine_name} has run successfully!".center(QTcts.LOG_LENGTH, "-"))
        sys.stdout.flush()
