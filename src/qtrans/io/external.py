"""QTrans io.external helpers."""

from typing import List
import os
import subprocess


def run_vaspkit(vaspkit_path:str, tasks:List[str], cwd='./'):
    """
    Run VASPKIT with given task ID.

    Args:
        vaspkit_path (str): The path to vaspkit executable.
        tasks (List[str]): Tasks name to run.
        cwd (str): Directory to run in.

    Returns:
        stdout (str): Standard output.
        stderr (str): Standard error output.
        proc.returncode (int): Return code.
    """
    proc = subprocess.Popen(
        [vaspkit_path],
        stdin=subprocess.PIPE,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if type(tasks) == str:
        tasks = [tasks]
    tasks = '\n'.join(tasks)
    input_str = f"{tasks}\n"
    stdout, stderr = proc.communicate(input_str)
    return stdout, stderr, proc.returncode

def get_job_system():
    if 'SLURM_JOB_ID' in os.environ:
        scheduler = "slurm"
    elif 'PBS_JOBID' in os.environ:
        scheduler = "pbs"
    else:
        scheduler = "Unknown"
    return scheduler
