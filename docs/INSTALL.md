
## Environment Setup
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```
Please refer to the [README.md](../README.md) in the above repository for downloading the Grounded-SAM-2 models and other prerequisites.
### Docker
#### Build
```bash
cd Grounded-SAM-2
docker build -t gsam2-base -f Dockerfile .
cd ..

docker build -t custom .
```
#### Run
```
docker run --gpus all -it --rm --net=host --privileged `
  --shm-size=8g `
  -v "${PWD}/Grounded-SAM-2:/home/appuser/Grounded-SAM-2" `
  -v "${PWD}:/home/appuser/workspace" `
  -w /home/appuser/workspace `
  custom
```
To exit the Docker container, type ```exit```
#### Verify GPU Availability
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```
If the output is similar to ```True NVIDIA GeForce RTX ****``` then the GPU is available.

## issues
###  UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
If you see this warning, you can fix it by running the following command inside the Docker container
(see [github issue](https://github.com/IDEA-Research/Grounded-SAM-2/issues/56#issuecomment-2471647093)):
```bash
pip install --no-build-isolation -e Grounded-SAM-2/grounding_dino
```

### ModuleNotFoundError: No module named 'grounding_dino'
If you encounter an error indicating that ```grounding_dino``` cannot be found when running a script, you need to set the ```PYTHONPATH```.

This is because the internal code of Grounded-SAM-2 relies on a specific directory structure (including folder names) for absolute and relative imports.

Run the following command **inside the Docker container**:

```bash
export PYTHONPATH=$PYTHONPATH:/home/appuser/workspace/Grounded-SAM-2
```