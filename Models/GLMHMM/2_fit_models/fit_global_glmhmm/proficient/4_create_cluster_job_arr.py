#  In order to facilitate parallelization of jobs, create a job array that
#  can be used on e.g. a cluster
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--run', dest='run', action='store_true', default=False, help='Run code serially')
args = parser.parse_args()
run = args.run

K_vals = [2, 3, 4, 5]
num_folds = 5
num_iterations = 20

if __name__ == '__main__':
    cluster_job_arr = []
    with open("4_job_array_pars.txt", "w") as f:
        for K in K_vals[::-1]:
            for i in range(num_folds):
                for n in range(num_iterations):
                    f.write(f"{K}\t{i}\t{n}\n")
                    if run:
                        os.system(f"python 4_run_inference_global_fit.py  {K}  {i}  {n}")
