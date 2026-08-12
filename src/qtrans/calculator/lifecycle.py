"""Mechanically extracted QTrans methods."""

from collections import Counter
from mpi4py import MPI
from qtrans.calculator.registry import resolve_calculation
from qtrans._version import __version__
from qtrans.config import apply_defaults, snapshot_configuration
from qtrans.core.defaults import calculator_default_parameters
from qtrans.core.errors import InputParameterError
from qtrans.io.external import get_job_system


class _LifecycleMixin:
    def __str__(self):
        class_list = []
        function_list = []
        variable_list = []
        for name in dir(self):
            if name.startswith("__"):
                continue
            if type(getattr(self, name)) == type:
                class_list.append(f"  Class {name}")
            elif callable(getattr(self, name)):
                function_list.append(f"  Function {name}")
            else:
                variable_list.append(f"  Variable {name}")
        output_str = "Class QTransCalculator:\n"
        output_str = output_str + '\n'.join(class_list) + '\n'
        output_str = output_str + '\n'.join(function_list) + '\n'
        output_str = output_str + '\n'.join(variable_list)
        return output_str

    def __logo__(self):
        logo = rf"""
          ____ _______                  
         / __ \__   __|                 
        | |  | | | |_ __ __ _ _ __  ___ 
        | |  | | | | '__/ _` | '_ \/ __|
        | |__| | | | | | (_| | | | \__ \
         \___\_\ |_|_|  \__,_|_| |_|___/
        
        version: {__version__}
        """
        print(logo)

    def __init__(self, config: dict = None, comm=None):
        """
        Initialize QTrans calculator.

        Args:
            config: Configuration dictionary parsed from YAML.
                    If None, prints usage instructions and returns.
            comm: MPI communicator for parallel execution.
                  If None, runs in single-process mode.
        """
        if config is None:
            print("This is QTransCalculator\nRun 'QTrans_run --config config.yaml'.")
            return

        # parallelization
        self.comm:MPI.Intracomm = comm

        if self.comm is None:
            self.rank = 0
            self.rank_size = 1
            self.nnodes = 1
            self.min_ntasks_per_node = 1
        else:
            self.rank = self.comm.Get_rank()
            self.rank_size = self.comm.Get_size()
            host = MPI.Get_processor_name()
            hosts = self.comm.allgather(host)
            count = Counter(hosts)
            self.nnodes = len(count)
            self.min_ntasks_per_node = min(count.values())
            
        # Show logo and parallel
        if self.rank == 0:
            self.__logo__()
            self.scheduler = get_job_system()
            print(f"Running QTrans using {self.rank_size} cores across {self.nnodes} nodes using {self.scheduler.upper()} scheduler.\n")
            
        # load default parameters
        try:
            apply_defaults(self, calculator_default_parameters)
        except Exception as e:
            raise RuntimeError(f"Error(9999): Fail to load default configuration.\n{e}")
        # read input
        try:
            self._parse_input(config=snapshot_configuration(config))
        except Exception as e:
            raise RuntimeError(f"Fail to load configuration.\n{e}")

    def _log(self, msg:str):
        if self.rank == 0:
            print(msg)

    def _set(self, name, validator, block, error_code, *args):
        try:
            setattr(self, name, validator(getattr(self, name), name, *args))
        except Exception as e:
            raise InputParameterError(error_code, block, name, e)

    def run(self):
        """
        Run QTrans in terms of the cal_mode.
        """
        action = resolve_calculation(self, self.cal_mode)
        if action:
            try:
                action()
            except Exception as e:
                raise RuntimeError(f"Error in '{self.cal_mode}' calculation.\n{e}")
        else:
            raise RuntimeError("Error(9999): Serious problem. Please contact the developer team.")

        self._show_job_success()

    def _show_job_success(self):
        self._log("\n######################################## QTRANS JOB SUCCESS ########################################")
