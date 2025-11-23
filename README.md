# csc6780-term-project
Distributed Computing Term Project

# Setup

The following instructions are for the HPC cluster.
If you're running locally, basically just don't do the cluster commands.

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

# Running The Code

## Running On The Cluster

If you're on the cluster, we have a single script that sets up and runs all the job stuff.
There are a few parameters near the top to change:

- `RUN_ROOT`: The root path where the code is
- `#SBATCH --nodes=n`: The number of worker nodes for patches
  - This is located in the first block of the script which handles creating the patch server jobs.
- Towards the end of the section that generates `patch_servers.sbatch` in an indented region, 
  you can change aspects of the generated config like `cpu` or `gpu` for the patch servers and manager instance.
  - There is a comment calling this part of the section out.
  - The only thing you will likely change of this is `base_session`, which has a comment with each valid option.
  - If you run the patch servers with GPU acceleration, 
    make sure to change the SBATCH options at the top including the partition.
    Additionally, at the end of the sbatch file, change `patch_server.py` to `patch_server_torch.py` to use GPU acceleration.

Then, to run the pipeline, it is simply:

```bash
./run_pipeline.sh
```

After the run, you can use this to extract timings to a CSV file:
```bash
python3 extract_timings.py
```

Basically, the flow on the cluster was:
```bash
git pull
./run_pipeline.sh
python3 extract_timings.py
```

## Running Locally

To run the code locally, you will need to start each server and the client.
To configure the servers, just edit the config file ([`config.yml`](./config.yml)).

Then run the servers and wait till they print their "listening" message to stderr.

ONNX Runtime servers:
```bash
python3 patch_manager.py
python3 patch_server.py
```

PyTorch servers:
```bash
python3 patch_manager_torch.py
python3 patch_server_torch.py
```

You can mix and match ONNX and PyTorch implementations if you want ONNX for CPU and PyTorch for GPU.

There's only one client implementation as it only really reads and writes images, 
so there's no point in having multiple runtimes.
Run it with:
```bash
python3 patch_client.py
```

If you want to use your own input image, you can change the file loaded in `patch_client.py`.
