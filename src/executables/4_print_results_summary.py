"""
Print a summary of the simulation jobs in a folder (i.e. from training or testing)
Metrics:
- sample success rate
- sample reach rate
- task success rate
Provides an approximate confidence interval (2*sigma)
"""

from tools.data_tools.utils import print_detailed_summary


if __name__ == '__main__':
    # path to the folder containing sim jobs (wrt the src folder)
    # also works on nested directories
    jobs_folder = ""
    print_detailed_summary(jobs_folder=jobs_folder)
