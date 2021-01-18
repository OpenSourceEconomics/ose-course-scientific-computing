"""This module contains some auxiliary functions shared across the utility scripts."""
import argparse
import difflib
import os
import subprocess as sp

LECTURES_ROOT = os.environ["PROJECT_ROOT"] + "/lectures"

LECTURES_NAME = []
LECTURES_NAME += ["optimization", "integration", "approximation"]
LECTURES_NAME += ["linear_equations", "nonlinear_equations"]


def run_notebook(notebook):
    """Execute a notebook."""
    cmd = ""
    cmd += " jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 "
    cmd += f"--execute {notebook} "
    sp.check_call(cmd, shell=True)


def parse_arguments(description):
    """Parse the arguments for the scripts."""
    parser = argparse.ArgumentParser(description=description)
    task = "lecture"

    parser.add_argument(
        "-n", "--name", type=str, help=f"name of {task}", default="all", dest="name"
    )

    args = parser.parse_args()

    # We can either request a single lecture or just act on all of them. We use string matching
    # to ease workflow.
    if args.name != "all":
        request = difflib.get_close_matches(args.name, LECTURES_NAME, n=1, cutoff=0.1)
        if not request:
            raise AssertionError(f"unable to match {task}")
    else:
        request = LECTURES_NAME

    request.sort()

    return request
