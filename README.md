# IE7374 Lab Submissions
This repository contains my lab assignments.

## GitHub Action
Workflows defined in the `.github/workflows` folder.
### lab1
The code is located in the `GitHub_Action/lab1/` folder.
GitHub Actions is configured to run `lab1_script.py` automatically on every push.
### lab2
This workflow implements a daily scheduled check that triggers model validation only when updates are detected in the `GitHub_Action/lab2` directory.\
It utilizes a Python-based quality gate to enforce a 95% accuracy threshold, \
automatically failing the build and generating a visual performance report in the GitHub Action Summary if the model underperforms.