set -e

ENV_NAME="venv_smart"

echo "Starting environment creation: $ENV_NAME"

uv venv --python 3.9 $ENV_NAME
source $ENV_NAME/bin/activate

echo "Installing PyTorch 1.12.1 (CUDA 11.3 build)..."
uv pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --index-url https://download.pytorch.org/whl/cu113

echo "Installing PyTorch Geometric stack (Pre-compiled wheels)..."
uv pip install \
    torch-scatter==2.1.0 \
    torch-sparse==0.6.16 \
    torch-cluster==1.6.0 \
    torch-spline-conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

uv pip install torch-geometric==2.6.1

echo "Installing remaining dependencies from requirements.txt..."
uv pip install -r /app/packages/smart/requirements.txt

echo "Installing Waymo Open Dataset separately..."
uv pip install waymo-open-dataset-tf-2-12-0

echo "Success! Environment '$ENV_NAME' is ready."
echo "To activate it, run: source $ENV_NAME/bin/activate"