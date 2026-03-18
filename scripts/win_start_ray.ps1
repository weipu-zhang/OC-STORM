# These should be executed at the client node
conda activate HKRL # replace with your conda environment name

$env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER="1"
ray start --address='<HEAD_NODE_IP>:6379' --resources='{\"env_runner\": 1}' # replace with your head node IP address