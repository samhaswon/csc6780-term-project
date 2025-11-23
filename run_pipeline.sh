#!/bin/bash
set -euo pipefail

# Where to store run-specific files; must be on a shared filesystem
# Change for your path/username
RUN_ROOT=/work/projects/csc6780-2025f-inference/sahoward42/csc6780-term-project
RUN_DIR="${RUN_ROOT}/runs/run-$(date +%s)"
mkdir -p "$RUN_DIR"

# Backup the config for git
cp config.yml config.yml.bak

echo "Run directory: $RUN_DIR"

###############################################################################
# 1. Create patch_servers job script (one worker per node on batch-impulse)
#
# Additionally, this is the file generation step where you will modify the
# parameters of the run like nodes, initial model, and devices
###############################################################################
cat > "${RUN_DIR}/patch_servers.sbatch" << 'EOF'
#!/bin/bash

#SBATCH --account=csc6780-2025f-inference
#SBATCH --job-name=patch_servers
#SBATCH --partition=batch-impulse
#SBATCH --nodes=5              # Set the number of worker nodes for this run
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=runs/patch_servers-%j.out

set -euo pipefail

: "${RUN_DIR:?RUN_DIR must be set via --export=RUN_DIR=/path}"
: "${RUN_ROOT:?RUN_ROOT must be set via --export=RUN_ROOT=/path}"

# Load Spack virtualenv package and activate the venv
spack load py-virtualenv

VENV_DIR="${RUN_ROOT}/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtualenv not found at $VENV_DIR" >&2
    exit 1
fi

. "${VENV_DIR}/bin/activate"

PATCH_PORT=5432
CONFIG="${RUN_ROOT}/config.yml"

echo "Patch job id: $SLURM_JOB_ID"
echo "Run dir: $RUN_DIR"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

echo "Allocated nodes:"
echo "$nodes"

# Write config.yml with base_session, devices, ports, and patch_servers.
# Adjust those parameters here.
{
  echo 'base_session:'
  echo '  - "u2netp"'          # u2net, u2netp, or birefnet

  echo 'patch_server_device:'
  echo '  - "cpu"'

  echo 'manager_server_device:'
  echo '  - "cuda"'

  echo 'patch_server_port:'
  echo "  - \"${PATCH_PORT}\""

  echo 'manager_server_port:'
  echo '  - "5433"'

  echo 'patch_servers:'
  for node in $nodes; do
    echo "  - \"${node}:${PATCH_PORT}\""
  done
} > "$CONFIG"

echo "Wrote patch_servers and devices to $CONFIG"

# Start one patch_server per node
srun --ntasks-per-node=1 python3 -u "${RUN_ROOT}/patch_server.py"
EOF

chmod +x "${RUN_DIR}/patch_servers.sbatch"

###############################################################################
# 2. Create manager job script (GPU, default gpu-warp partition)
###############################################################################
cat > "${RUN_DIR}/manager.sbatch" << 'EOF'
#!/bin/bash

#SBATCH --account=csc6780-2025f-inference
#SBATCH --job-name=manager
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=runs/manager-%j.out

set -euo pipefail

: "${RUN_DIR:?RUN_DIR must be set via --export=RUN_DIR=/path}"
: "${RUN_ROOT:?RUN_ROOT must be set via --export=RUN_ROOT=/path}"

spack load py-virtualenv

VENV_DIR="${RUN_ROOT}/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtualenv not found at $VENV_DIR" >&2
    exit 1
fi

. "${VENV_DIR}/bin/activate"

CONFIG="${RUN_ROOT}/config.yml"
MANAGER_PORT=5433
MANAGER_ADDR_FILE="${RUN_DIR}/manager_addr.txt"

echo "Manager job id: $SLURM_JOB_ID"
echo "Run dir: $RUN_DIR"

# Wait until config.yml exists (written by patch_servers job)
while [ ! -f "$CONFIG" ]; do
    echo "[$(date)] Waiting for $CONFIG to exist..."
    sleep 5
done

echo "Found $CONFIG, appending manager info"

if ! grep -q "^manager_servers:" "$CONFIG"; then
    {
        echo 'manager_servers:'
        echo "  - \"${HOSTNAME}:${MANAGER_PORT}\""
        echo 'manager_server_port:'
        echo "  - \"${MANAGER_PORT}\""
    } >> "$CONFIG"
else
    echo "manager_servers already present in $CONFIG, not appending"
fi

echo "${HOSTNAME}:${MANAGER_PORT}" > "$MANAGER_ADDR_FILE"
echo "Wrote manager address to $MANAGER_ADDR_FILE"

python3 -u "${RUN_ROOT}/patch_manager_torch.py"
EOF

chmod +x "${RUN_DIR}/manager.sbatch"

###############################################################################
# 3. Create client job script (waits for manager, then exits)
###############################################################################
cat > "${RUN_DIR}/client.sbatch" << 'EOF'
#!/bin/bash

#SBATCH --account=csc6780-2025f-inference
#SBATCH --job-name=client
#SBATCH --time=00:30:00
#SBATCH --output=runs/client-%j.out

set -euo pipefail

: "${RUN_DIR:?RUN_DIR must be set via --export=RUN_DIR=/path}"
: "${RUN_ROOT:?RUN_ROOT must be set via --export=RUN_ROOT=/path}"

spack load py-virtualenv

VENV_DIR="${RUN_ROOT}/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtualenv not found at $VENV_DIR" >&2
    exit 1
fi

. "${VENV_DIR}/bin/activate"

CONFIG="${RUN_DIR}/config.yml"
MANAGER_ADDR_FILE="${RUN_DIR}/manager_addr.txt"

echo "Client job id: $SLURM_JOB_ID"
echo "Run dir: $RUN_DIR"

while [ ! -f "$MANAGER_ADDR_FILE" ]; do
    echo "[$(date)] Waiting for $MANAGER_ADDR_FILE to exist..."
    sleep 5
done

MANAGER_ADDR=$(cat "$MANAGER_ADDR_FILE")
echo "Manager address: $MANAGER_ADDR"

# Small grace delay to let manager start listening
sleep 5

python3 -u "${RUN_ROOT}/patch_client.py"
EOF

chmod +x "${RUN_DIR}/client.sbatch"

###############################################################################
# 4. Submit jobs
###############################################################################

PATCH_JOBID=$(sbatch --export=ALL,RUN_ROOT="$RUN_ROOT",RUN_DIR="$RUN_DIR" \
    "${RUN_DIR}/patch_servers.sbatch" | awk '{print $4}')

# Wait for job allocation, but not server startup. We'll wait on that later.
sleep 5
MANAGER_JOBID=$(sbatch --export=ALL,RUN_ROOT="$RUN_ROOT",RUN_DIR="$RUN_DIR" \
    "${RUN_DIR}/manager.sbatch" | awk '{print $4}')

# Wait a little bit for all of the servers to start
sleep 20
CLIENT_JOBID=$(sbatch --export=ALL,RUN_ROOT="$RUN_ROOT",RUN_DIR="$RUN_DIR" \
    "${RUN_DIR}/client.sbatch" | awk '{print $4}')

echo "Submitted jobs:"
echo "  patch_servers: ${PATCH_JOBID}"
echo "  manager:       ${MANAGER_JOBID}"
echo "  client:        ${CLIENT_JOBID}"

###############################################################################
# 5. Wait for client to finish
###############################################################################
echo "Waiting for client job ${CLIENT_JOBID} to finish..."


while true; do
    # If squeue shows no such job, we are done (completed, failed, or cancelled).
    if ! squeue -h -j "$CLIENT_JOBID" | grep -q "$CLIENT_JOBID"; then
        echo "Client job ${CLIENT_JOBID} is no longer in the queue."
        break
    fi
    sleep 10
done

###############################################################################
# 6. Kill manager and workers
###############################################################################
echo "Cancelling manager job ${MANAGER_JOBID} and patch job ${PATCH_JOBID}..."
scancel "$MANAGER_JOBID" || echo "Manager job ${MANAGER_JOBID} already gone."
scancel "$PATCH_JOBID"   || echo "Patch job ${PATCH_JOBID} already gone."

echo "Done. Logs and config are in: $RUN_DIR"

# Restore the old config
mv config.yml.bak config.yml
