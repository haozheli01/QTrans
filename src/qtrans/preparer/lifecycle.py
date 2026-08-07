"""Mechanically extracted QTrans methods."""

from collections import Counter
from mpi4py import MPI
from qtrans._version import __version__
from qtrans.config import apply_defaults, snapshot_configuration
from qtrans.core.defaults import preparer_default_parameters
from qtrans.io.external import get_job_system
from qtrans.preparer.registry import resolve_preparation


class _PreparerLifecycleMixin:
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
        output_str = "Class QTransPreparer:\n"
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

    def __init__(self, config:dict=None, comm=None):
        if config == None:
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
            self.min_ntasks_per_node = min(list(count.values()))

        # Show logo and parallel condition
        if self.comm is None:
            self.__logo__()
            self.scheduler = get_job_system()
            print(f"Running QTrans in serial mode using {self.scheduler.upper()} scheduler.")
            print()
        elif self.rank == 0:
            self.__logo__()
            self.scheduler = get_job_system()
            print(f"Running QTrans using {self.rank_size} cores across {self.nnodes} nodes using {self.scheduler.upper()} scheduler.")
            print()

        # load default paramters
        apply_defaults(self, preparer_default_parameters)

        # read input
        self._parse_input(config=snapshot_configuration(config))

    def run(self):
        """
        Run QTrans preparation in terms of the prep_mode.
        """
        action = resolve_preparation(self, self.prep_mode)
        if action is not None:
            action()

        if self.rank == 0:
            self._show_job_success()

    def _show_job_success(self):
        print("\n######################################## QTRANS JOB SUCCESS ########################################")
