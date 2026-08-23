"""Command-line entry points for QTrans."""

import argparse

from .config import load_configuration


def _print_failure(rank, error):
    if rank != 0:
        return

    title = " QTrans Failed "
    message_lines = str(error).splitlines() or [repr(error)]
    content_width = max([len(title)] + [len(line) for line in message_lines])
    border = "!" * (content_width + 4)
    print("\n" + border, flush=True)
    print("! " + title.center(content_width) + " !", flush=True)
    for line in message_lines:
        print("! " + line.ljust(content_width) + " !", flush=True)
    print(border + "\n", flush=True)


def _parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser


def calculator_main(argv=None):
    from mpi4py import MPI

    from .calculator import QTransCalculator

    args = _parser("Run a QTrans calculation.").parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        calculator = QTransCalculator(load_configuration(args.config), comm)
        calculator.run()
    except Exception as error:
        _print_failure(rank, error)
        raise SystemExit(1)


def preparer_main(argv=None):
    from mpi4py import MPI

    from .preparer import QTransPreparer

    args = _parser("Prepare input data for QTrans.").parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    calculator_comm = None if comm.Get_size() == 1 else comm

    try:
        preparer = QTransPreparer(load_configuration(args.config), calculator_comm)
        preparer.run()
    except Exception as error:
        _print_failure(rank, error)
        raise SystemExit(1)
