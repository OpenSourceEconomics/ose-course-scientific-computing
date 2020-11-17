"""This module contains some auxiliary functions shared across the utility scripts."""
import argparse
import difflib
import glob
import os
import subprocess as sp

LECTURES_ROOT = os.environ["PROJECT_ROOT"] + "/lectures"


def run_notebook(notebook):
    """Execute a notebook."""
    cmd = ""
    cmd += " jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 "
    cmd += f"--execute {notebook} "
    sp.check_call(cmd, shell=True)


def parse_arguments(description):
    """Parse the arguments for the scripts."""
    parser = argparse.ArgumentParser(description=description)
    task, task_dir = "lecture", LECTURES_ROOT

    parser.add_argument(
        "-n", "--name", type=str, help=f"name of {task}", default="all", dest="name"
    )

    args = parser.parse_args()

    # We can either request a single lecture or just act on all of them. We use string matching
    # to ease workflow.
    if args.name != "all":
        request = difflib.get_close_matches(args.name, get_list_tasks(task_dir), n=1, cutoff=0.1)
        if not request:
            raise AssertionError(f"unable to match {task}")
    else:
        request = get_list_tasks(task_dir)

    request.sort()

    return request


def get_list_tasks(task_dir):
    """Get a list of tasks."""
    cwd = os.getcwd()
    os.chdir(task_dir)
    lectures = glob.glob("*_*") + ["optimization"]
    os.chdir(cwd)

    return lectures
