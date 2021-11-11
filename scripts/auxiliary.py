"""This module contains some auxiliary functions shared across the utility scripts."""
import argparse
import difflib
import os
import subprocess as sp

LABS_ROOT = os.environ["PROJECT_ROOT"] + "/labs"

LABS_NAME = []
LABS_NAME += ["optimization", "integration", "approximation"]
LABS_NAME += ["linear_equations", "nonlinear_equations"]


def run_notebook(notebook):
    """Execute a notebook."""
    cmd = ""
    cmd += " jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 "
    cmd += f"--execute {notebook} "
    sp.check_call(cmd, shell=True)


def parse_arguments(description):
    """Parse the arguments for the scripts."""
    parser = argparse.ArgumentParser(description=description)
    task = "lab"

    parser.add_argument(
        "-n", "--name", type=str, help=f"name of {task}", default="all", dest="name"
    )

    args = parser.parse_args()

    # We can either request a single lab or just act on all of them. We use string matching
    # to ease workflow.
    if args.name != "all":
        request = difflib.get_close_matches(args.name, LABS_NAME, n=1, cutoff=0.1)
        if not request:
            raise AssertionError(f"unable to match {task}")
    else:
        request = LABS_NAME

    request.sort()

    return request
