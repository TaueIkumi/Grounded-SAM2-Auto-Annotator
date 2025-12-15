
## Installation
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```
For model downloads and other details regarding Grounded-SAM-2, please refer to the [README.md](README.md).

### Docker
Build
```bash
cd Grounded-SAM-2
docker build -t gsam2-base -f Dockerfile .
cd ..

docker build -t custom .
```
Run
```
docker run --gpus all -it --rm --net=host --privileged `
  --shm-size=8g `
  -v "${PWD}/Grounded-SAM-2:/home/appuser/Grounded-SAM-2" `
  -v "${PWD}:/home/appuser/workspace" `
  -w /home/appuser/workspace `
  custom
```
To exit the Docker environment, type `exit`.

Verify GPU Availability
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```
If the output looks like `>> True NVIDIA GeForce RTX ****, `the GPU is ready to use.


## Troubleshooting
UserWarning: **Failed to load custom C++ ops. Running on CPU mode Only!** \
If you encounter this error, you can fix it by running the following command after starting the Docker container (Reference: [github-issue](https://github.com/IDEA-Research/Grounded-SAM-2/issues/56#issuecomment-2471647093))

```bash
pip install --no-build-isolation -e Grounded-SAM-2/grounding_dino
```

**ModuleNotFoundError: No module named 'grounding_dino'** \
If you receive an error stating that grounding_dino cannot be found when running the script, you need to set the PYTHONPATH. This is required because the internal code of Grounded-SAM-2 relies on a specific directory structure (involving absolute/relative imports with folder names).

Execute the following command inside the Docker container to set the path:

```bash
export PYTHONPATH=$PYTHONPATH:/home/appuser/workspace/Grounded-SAM-2
```