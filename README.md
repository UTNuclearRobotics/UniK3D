# UniK3D + YOLO

This repository builds on the work of UniK3D. It is not affiliated with ETH Zurich. Please consider their original [repo](https://github.com/lpiccinelli-eth/UniK3D) and [paper](https://arxiv.org/pdf/2503.16591) for more information.

This follows from my original, more detailed implentation of [UniDepth](https://github.com/NikiMoelders/UniDepth--custom/blob/main/README.md?plain=1).

## Installation

The following worked for both Jetson and SSH. You may follow the setup of the original repo with CUDA 12.1 but unsure if it works for Jetson. I recommend starting with the environment of the previous repo.

Install the environment needed to run UniK3D with:

```shell
export VENV_DIR=<YOUR-VENVS-DIR>
export NAME=Unidepth

python -m venv $VENV_DIR/$NAME
source $VENV_DIR/$NAME/bin/activate
```
### Install UniDepth and dependencies, cuda >11.8 work fine, too.
```shell
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
```
### Install Ultralytics for YOLO

```shell
pip install ultralytics
```

### Install Pillow-SIMD (Optional)
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

### Install KNN (for evaluation only)
cd ./unik3d/ops/knn;bash compile.sh;cd ../../../
```

If you use conda, you should change the following: 
```shell
python -m venv $VENV_DIR/$NAME -> conda create -n $NAME python=3.11
source $VENV_DIR/$NAME/bin/activate -> conda activate $NAME
```

Run UniK3D on the given assets to test your installation (you can check this script as guideline for further usage):
```shell
python ./scripts/demo.py
```
If everything runs correctly, `demo.py` should print: `RMSE on 3D clouds for ScanNet sample: 21.9cm`.
`demo.py` allows you also to save output information, e.g. rays, depth and 3D pointcloud as `.ply` file.

## Depth Estimation + Object Detection

- Scripts
  - [depth_jetson.py](scripts/depth_jetson.py) is optimzed for the Jetson

- YOLO Weights
  - A fine-tuned YOLO model is employed- [yolo11n-uav-vehicle-bbox.pt](yolo_models/yolo11n-uav-vehicle-bbox.pt)

- Inference

```shell
python scripts/depth_jetson.py
```
or similarly

```shell
python scripts/depth_jetson.py --video VIDEO_PATH --fps DESIRED_FPS --conf DESIRED_YOLO_CONFIDENCE 
```
The annotated video will save in the output folder. If running on an SSH, you will need to download the video onto your local machine to play it.

## Gradio Demo

- Plase visit our [HugginFace Space](https://huggingface.co/spaces/lpiccinelli/UniK3D-demo) for an installation-free test on your images!
- You can use a local Gradio demo if the HuggingFace is too slow (CPU-based) by running `python ./gradio_demo.py` after installation.


## Get Started

After installing the dependencies, you can load the pre-trained models easily from [Hugging Face](https://huggingface.co/lpiccinelli) as follows:

```python
from unik3d.models import UniK3D

model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl") # vitl for ViT-L backbone
```

Then you can generate the metric 3D estimation and rays prediction directly from a single RGB image only as follows:

```python
import numpy as np
from PIL import Image

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the RGB image and the normalization will be taken care of by the model
image_path = "./assets/demo/scannet.jpg"
rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W

predictions = model.infer(rgb)

# Point Cloud in Camera Coordinate
xyz = predictions["points"]

# Unprojected rays
rays = predictions["rays"]

# Metric Depth Estimation
depth = predictions["depth"]
```

You can use ground truth camera parameters or rays as input to the model as well:
```python
from unik3d.utils.camera import (Pinhole, OPENCV, Fisheye624, MEI, Spherical)

camera_path = "assets/demo/scannet.json" # any other json file
with open(camera_path, "r") as f:
    camera_dict = json.load(f)

params = torch.tensor(camera_dict["params"])
name = camera_dict["name"]
camera = eval(name)(params=params)
predictions = model.infer(rgb, camera)
```

To use the forward method for your custom training, you should:  
1) Take care of the dataloading:  
  a) ImageNet-normalization  
  b) Long-edge based resizing (and padding) with input shape provided in `image_shape` under configs  
  c) `BxCxHxW` format  
  d) If any intriniscs given, adapt them accordingly to your resizing  
2) Format the input data structure as:  
```python
data = {"image": rgb, "rays": rays}
predictions = model(data, {})
```

## Infer

To run locally, you can use the script `./scripts/infer.py` via the following command:

```bash
# Save the output maps and ply
python ./scripts/infer.py --input IMAGE_PATH --output OUTPUT_FOLDER --config-file configs/eval/vitl.json --camera-path CAMERA_JSON --save --save-ply
```

```
Usage: scripts/infer.py [OPTIONS]

Options:
  --input PATH                Path to input image.
  --output PATH               Path to output directory.
  --config-file PATH          Path to config file. Please check ./configs/eval.
  --camera-path PATH          (Optional) Path to camera parameters json file. See assets/demo
                              for a few examples. The file needs a 'name' field with
                              the camera model from unik3d/utils/camera.py and a
                              'params' field with the camera parameters as in the
                              corresponding class docstring.
  --resolution-level INTEGER  Resolution level in [0,10). Higher values means it will
                              resize to larger resolution which increases details but
                              decreases speed. Lower values lead to opposite.
  --save                      Save outputs as (colorized) png.
  --save-ply                  Save pointcloud as ply.
```

See also [`./scripts/infer.py`](./scripts/infer.py)



## Model Zoo

The available models are the following:

<table border="0">
    <tr>
        <th>Model</th>
        <th>Backbone</th>
        <th>Name</th>
    </tr>
    <hr style="border: 2px solid black;">
    <tr>
        <td rowspan="3"><b>UniK3D</b></td>
        <td>ViT-S</td>
        <td><a href="https://huggingface.co/lpiccinelli/unik3d-vits">unik3d-vits</a></td>
    </tr>
    <tr>
        <td>ViT-B</td>
        <td><a href="https://huggingface.co/lpiccinelli/unik3d-vitb">unik3d-vitb</a></td>
    </tr>
    <tr>
        <td>ViT-L</td>
        <td><a href="https://huggingface.co/lpiccinelli/unik3d-vitl">unik3d-vitl</a></td>
    </tr>
</table>

Please visit [Hugging Face](https://huggingface.co/lpiccinelli) or click on the links above to access the repo models with weights.
You can load UniK3D as the following, with `name` variable matching the table above:

```python
from unik3d.models import UniK3D

model_v1 = UniK3D.from_pretrained(f"lpiccinelli/{name}")
```

In addition, we provide loading from TorchHub as:

```python
backbone = "vitl"

model = torch.hub.load("lpiccinelli-eth/UniK3D", "UniK3D", backbone=backbone, pretrained=True, trust_repo=True, force_reload=True)
```

You can look into function `UniK3D` in [hubconf.py](hubconf.py) to see how to instantiate the model from local file: provide a local `path` in line 23.


## Training

Please visit the [docs/train](docs/train.md) for more information.


## Results

Please visit the [docs/eval](docs/eval.md) for more information about running evaluation..

### Metric 3D Estimation
The metrics is F1 over metric 3D pointcloud (higher is better) on zero-shot evaluation. 

| Model | SmallFoV | SmallFoV+Distort | LargeFoV | Panoramic |
| :-: | :-: | :-: | :-: | :-: |
| UniDepth | 59.0 | 43.0 | 16.9 | 2.0 |
| MASt3R | 37.8 | 35.2 | 29.7 | 3.7 |
| DepthPro | 56.0 | 29.4 | 26.1 | 1.9 |
| UniK3D-Small | 61.3 | 48.4 | 55.5 | 72.5 |
| UniK3D-Base | 64.9 | 50.2 | 67.7 | 73.7 |
| UniK3D-Large | 68.1 | 54.5 | 71.6 | 80.2 |


## Contributions

If you find any bug in the code, please report to Luigi Piccinelli (lpiccinelli@ethz.ch)


## Citation

If you find our work useful in your research please consider citing our publications:
```bibtex
@inproceedings{piccinelli2025unik3d,
    title     = {{U}ni{K3D}: Universal Camera Monocular 3D Estimation},
    author    = {Piccinelli, Luigi and Sakaridis, Christos and Segu, Mattia and Yang, Yung-Hsu and Li, Siyuan and Abbeloos, Wim and Van Gool, Luc},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```


## License

This software is released under Creatives Common BY-NC 4.0 license. You can view a license summary [here](LICENSE).


## Acknowledgement

This work is funded by Toyota Motor Europe via the research project [TRACE-Zurich](https://trace.ethz.ch) (Toyota Research on Automated Cars Europe).
