# MarmoPose

Welcome to **MarmoPose**, a comprehensive real-time multi-marmoset 3D pose tracking system.  

<div align="center">
  <img src="resources/demo.jpg" alt="MarmoPose Demo" width="600"/>
</div>


## Installation Guide

MarmoPose has been tested on **Windows**. Follow the steps below to set up your environment.

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a new Conda environment and activate it.

```shell
conda create --name marmopose python=3.8
conda activate marmopose
```

**Step 2.** Follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The recommended installation command is:

```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Step 3.** Install [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv/tree/2.x), [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMPose](https://github.com/open-mmlab/mmpose/tree/main) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmpose==1.3.1
pip install mmdeploy==1.3.1
pip install mmdeploy-runtime-gpu==1.3.1
```

**Step 4.** Clone this repository and install dependencies.

```shell
git clone https://github.com/Leoswordy/MarmoPose.git
cd MarmoPose
pip install -e .
```

**Step 5.** (Optional) If you want to run TensorRT-deployed models (for realtime processing), follow these steps:
1. Install CUDA 11.8 by following the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
2. Install TensorRT 8.6 following the **Zip File Installation** section in [TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).


**Step 6.** (Optional) To deploy new models, refer to the [MMDeploy](https://mmdeploy.readthedocs.io/en/latest/get_started.html).


## Usage
First download the pretrained **models** and **demos** from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/7f190f04b2e541b1a408/), and place them in the same directory as `README.md`. Alternatively, store them in another directory and specify it in the config file. For getting started, please refer to the Jupyter Notebooks in the `examples` directory.

### Models
Currently, we provide 6 pretrained models tailored for different scenarios. The training data is **Marmoset3K**, containing 1527 images with one marmoset and 1646 images with two marmosets (where one is dyed blue).

#### `detection_model`: 
  - **Trained on**: Marmoset3K
  - **Purpose**: Predict bounding boxes and identities for 1 or 2 marmosets.
  - **Identities**: 
    - `'white_head_marmoset': 0`
    - `'blue_head_marmoset': 1`
  - **Use case**: Videos containing 1 or 2 marmosets. If two are present, one must be dyed blue.

#### `pose_model`:
  - **Trained on**: Marmoset3K
  - **Purpose**: Predict the pose of each instance in cropped images based on the bboxes predicted by the detection model.
  - **Use case**: Can be combined with any type of detection model; no specific color requirement.

#### `dae_model`
  - **Trained on**: Marmoset3D, a 3D pose dataset constructed by triangulation.
  - **Purpose**: Fill in missing values in 3D poses.
  - **Use case**: Any scenario where necessary.

#### `detection_model_family_finetune`
  - **Finetuned on**: 100 images (only bboxes annotated) containing a family of 4 differently colored marmosets
  - **Purpose**: Predicts bounding boxes and identities for up to 4 marmosets.
  - **Identities**: 
    - `'white_head_marmoset': 0`
    - `'blue_head_marmoset': 1`
    - `'green_head_marmoset': 2`
    - `'red_head_marmoset': 3`
  - **Use case**: Videos containing up to 4 marmosets; the color of marmosets should be a subset of the specified colors.

#### `detection_model_deployed`
  - Deployed `detection_model` using TensorRT, offers faster inference speed with slightly lower precision.
  - **Use case**: Specific to RTX4090, may require redeployment on other hardware. See [MMDeploy](https://mmdeploy.readthedocs.io/en/latest/get_started.html) for instructions on re-deploying on new hardware.

#### `pose_model_deployed`
  - Deployed `pose_model` using TensorRT, offers faster inference speed with slightly lower precision.
  - **Use case**: Specific to RTX4090, may require redeployment on other hardware. See [MMDeploy](https://mmdeploy.readthedocs.io/en/latest/get_started.html) for instructions on re-deploying on new hardware.


### Demos and Examples

We provide examples and corresponding demos to help you get started. Please use the default configuration file provided in the `configs` directory. For each project, place calibrated camera parameters or videos for calibration in the `calibration` directory. Place raw videos for analysis in the `videos_raw` directory.
> **Note**: Calibration only needs to be done once as long as the camera setup is not changed, refer to `examples/calibrate.ipynb`.

#### `2 marmosets` 
> We highly recommend running this demo first, as the notebook includes detailed comments on the configuration, processing steps, and functions.

For scenarios containing 2 marmosets (one dyed blue), refer to `examples/demo_pair.ipynb` and `demos/pair`.

#### `1 marmoset`
For scenarios containing 1 marmoset, refer to `examples/demo_single.ipynb` and `demos/single`.

#### `4 marmosets`
For scenarios containing 4 marmosets, refer to `examples/demo_family.ipynb` and `demos/family`. 

> **Note**: The training data from **Marmoset3K** does **NOT** cover scenarios with 4 marmosets. You may use the `detection_model_family` and `pose_model` for preliminary results, but these might not meet practical demands. Finetuning with new labeled data is required.

#### `Track a subset of marmosets in the video`
For scenarios involving 4(or 3, or 2) marmosets where only a subset needs to be tracked, refer to `examples/demo_family_subset.ipynb` and `demos/family_subset`. 
  - **Example 1** If there are 4 marmosets, 2 adults and 2 young, and the young marmosets are difficult to dye, you can choose to only track the two adults.
  - **Example 2** If there are 4 marmosets with different identifiable landmarks (e.g., new colors), but no model exists to track the new type, you may choose to label new data for finetuning the `detection_model` or track only the subset of marmosets with existing colors.

> **Note**: This demo is in a complex scenario (family marmosets with young, and more obstacles in the home cage). The training data from **Marmoset3K** does **NOT** cover these scenarios. Although preliminary results can be obtained with the `detection_model_family` and `pose_model`, the performance may not be satisfactory (You may see that current version of **MarmoPose** is not effective in this demo, that's because the landmarks of the `green_head_marmoset` are invisible at most of the time and views, making it difficult to detect and assign label. Additionally, new poses not covered in the training data are currently not recognized well.). More pose data is necessary to finetune the models.

### Datasets

The **Marmoset3K** dataset is  available at [Zenodo](https://doi.org/10.5281/zenodo.14672425).

## Realtime Processing
For scenarios requiring real-time processing, refer to `examples/realtime.py` and `demos/realtime`.


## Fine-tune models

### Label data

It is essential to first label new data accurately. We recommend using [SLEAP](https://sleap.ai/) for this purpose. Detailed instructions for labeling are provided in `tools/Data Annotation Guide.pdf`.

2D prediction models in MarmoPose are trained with data in [COCO format](https://cocodataset.org/#format-data). After labeling new data with SLEAP, you must convert it to COCO format. Refer to `tools/sleap2coco.py` to for this conversion process.


### Fine-tune detection model

1. Modify these parameters in  `tools/train_config/detection_config.py`:
    - `data_root`: Path to new training data
    - `model.bbox_head.num_classes`: The number of marmosets in the new dataset
    - `dataset.metainfo.classes` in `train_dataloader` and `val_dataloader`: The number and names should match the categories in the converted COCO dataset
    - `max_epochs`: Based on the size of new dataset
    - `others (optional)`: In general, keep other settings unchanged. For additional customization options, refer to [MMDetection Train](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#train-with-customized-datasets)

2. Run the following command to continue training from a checkpoint

```shell
python tools/train.py tools/train_config/detection_config.py --resume models/detection_model_family/best.pth --work-dir data/detection_model_family_finetune
```

> **Note**: Select the checkpoint that best matches your scenario, e.g., `detection_model` for 1 or 2 marmosets, `detection_model_family` for 4 marmosets. For other number of marmosets, it will load the backbone and neck weights from the chosen checkpoint, and initialize the head weights randomly.


### Fine-tune pose model

1. Modify these parameters in  `tools/train_config/pose_config.py`:
    - `data_root`: Path to new training data
    - `max_epochs`: Based on the size of new dataset
    - `others (optional)`: In general, keep other settings unchanged. For additional customization options, refer to [MMPose Train](https://mmpose.readthedocs.io/en/latest/user_guides/train_and_test.html)

2. Run the following command to resume training from a checkpoint

```shell
python tools/train.py tools/train_config/pose_config.py --resume models/pose_model/best.pth --work-dir data/pose_model_finetune
```

> **Note**: Remove `--resume` if you want to start training from scratch.


## Useful Tools

MarmoPose provides several utility scripts in the `tools` directory for processing videos and RTSP camera streams.

#### `video_converter.py`
Converts and aligns video files for further processing.

#### `realtime_synchronized_record.py`
Records synchronized video streams from multiple cameras and saves them locally.


## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@article{MarmoPose,
  title={A real-time, multi-subject three-dimensional pose tracking system for the behavioral analysis of non-human primates},
  author={Cheng, Chaoqun and Huang, Zijian and Zhang, Ruiming and Huang, Guozheng and Wang, Han and Tang, Likai and Wang, Xiaoqin},
  journal={Cell Reports Methods},
  year={2025}
}
```

