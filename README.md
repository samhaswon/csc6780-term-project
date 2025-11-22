# csc6780-term-project
Distributed Computing Term Project

# Setup

## Model Downloads

Some of the models used in this project are too large to commit to GitHub, so they are all in a release.

First copy/clone the code into your folder on the HPC cluster. 
This should be the following if you are Dr. Rogers.
Otherwise, replace his username with your own.
```bash
cd /work/projects/csc6780-2025f-inference/mrogers/
```

Then change directory into the project.
```bash
cd csc6780-term-project/
```

After cloning/copying the repo to the cluster, download the models.
If you want to run this in the interactive shell later, you can.
```bash
./download_models.sh
```

## Environment Setup

First, allocate a job on the cluster to run the setup with. 
You should have access to the account used here.
```bash
hpcshell --account=csc6780-2025f-inference --cpus-per-task=4 --mem=8G
```

Next, we're going to set up the virtual environment.
To do so, we're going to load the appropriate spack package.
We're just going to put everything in here for simplicity as there are some issues with some of the packages on the cluster.
```bash
spack load py-virtualenv
```

Then, create the virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

Then, to install everything, run this.
This will take a few minutes as PyTorch and the associated CUDA packages are somewhat large.
If you get an error, just run it again.
```bash
pip install -r requirements.txt
```

Afterward, you can just `exit` the interactive session as we are done with it.
