#!/usr/bin/env python
"""Run notebooks.

This script allows to run the lab notebooks. One can either run all notebooks at once or just a
single lab. It is enough to provide a substring for the name.

Examples
--------
>> run-lab             Run all labs.

>> run-lab -n intr     Run lab 01-introduction.
"""
import glob
import os

from auxiliary import LABS_ROOT
from auxiliary import parse_arguments
from auxiliary import run_notebook


if __name__ == "__main__":

    request = parse_arguments("Execute notebook")
    os.chdir(LABS_ROOT)

    for dirname in request:

        os.chdir(dirname)
        for fname in glob.glob("*.ipynb"):
            print(f"\n {os.getcwd().split('/')[-1]}\n")  # noqa
            run_notebook(fname)
        os.chdir("../")
