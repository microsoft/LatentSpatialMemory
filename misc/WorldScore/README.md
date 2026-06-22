# <p align="center"><img src="assets/worldscore.png" height=100></p>WorldScore: A Unified Evaluation Benchmark for World Generation

<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 10px;">
    <a href='https://arxiv.org/abs/2504.00983'><img src='https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red'></a>
    <a href='https://haoyi-duan.github.io/WorldScore/'><img src='https://img.shields.io/badge/WorldScore-Website-green?logo=googlechrome&logoColor=green'></a>
    <a href='https://huggingface.co/datasets/Howieeeee/WorldScore'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow?logo=huggingface&logoColor=yellow'></a>
    <a href='https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard'><img src='https://img.shields.io/badge/Leaderboard-Huggingface-yellow?logo=huggingface&logoColor=yellow'></a>
</div>


> #### [WorldScore: A Unified Evaluation Benchmark for World Generation](https://arxiv.org/abs/2504.00983)
>
> ##### [Haoyi Duan*](https://haoyi-duan.github.io/), [Hong-Xing "Koven" Yu*](https://kovenyu.com/), [Sirui Chen](https://www.linkedin.com/in/sirui-c-6492a0232/), [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li), [Jiajun Wu](https://jiajunwu.com/) ("*" denotes equal contribution)



## Table of Contents

- [Updates](#updates)
- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [World Generation](#world-generation )
- [Evaluation](#evaluation)
- [Leaderboard](#leaderboard)
- [World Generation Models Info](#world-generation-models-info)
- [Citation](#citation)

## 🔥 Updates <a name="updates"></a>
- [11/2025] The evaluation code for [WonderJourney](https://kovenyu.com/wonderjourney/) and [WonderWorld](https://kovenyu.com/wonderworld/) is uploaded [here](https://github.com/haoyi-duan/WorldScore/blob/main/world_generators/README.md#adaption-for-world-generation-models).
- [06/2025] [Voyager](https://voyager-world.github.io/) results uploaded.
- [06/2025] Paper accepted to ICCV 2025!
- [04/2025] Paper released <a href='https://arxiv.org/abs/2504.00983'><img src='https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red'></a>.
- [04/2025] Code released.
- [03/2025] Leaderboard released <a href='https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard'><img src='https://img.shields.io/badge/Leaderboard-Huggingface-yellow?logo=huggingface&logoColor=yellow'></a>.
- [03/2025] Dataset released <a href='https://huggingface.co/datasets/Howieeeee/WorldScore'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow?logo=huggingface&logoColor=yellow'></a>.

## 📣 Overview <a name="overview"></a>



https://github.com/user-attachments/assets/2fb6d8fa-050b-4ded-99cc-46051cf2e4f9


Here we showcase how the WorldScore-Static metric measures two models given an initial scene of **a bedroom** with a specified camera path—**"pan left" → "move left" → "pull out"**. While existing benchmarks rate Models A and B similarly based on single-scene video quality, our WorldScore benchmark differentiates their world generation capabilities by identifying that Model B fails to generate a new scene or follow the instructed camera movement.

## 🚀 Setup Instructions <a name="setup-instructions"></a>

#### 1. Clone the repository

```shell
git clone https://github.com/haoyi-duan/WorldScore.git
cd WorldScore
```

#### 2. Configure Environment Paths

Before running, you need to set up environment paths by creating a `.env` file in the root of this repository. This file should contain the following variables: `WORLDSCORE_PATH` is the root path where this repo was cloned.
`MODEL_PATH` is the root path of the model repo (e.g., `MODEL_PATH/CogVideo`) as well as where the evaluation outputs will be saved (e.g., `MODEL_PATH/CogVideo/worldscore_output`). Finally, `DATA_PATH` is the path to where the WorldScore dataset will be stored (`DATA_PATH/WorldScore-Dataset`).

```sh
WORLDSCORE_PATH=/path/to/worldscore
MODEL_PATH=/path/to/model
DATA_PATH=/path/to/dataset
```

#### 3. Export the Environment Variables

After creating the `.env` file, make sure to export the variables so that they are accessible throughout the workflow. 

> [!NOTE]
>
> **This step must be repeated in every new terminal session.**

```sh
export $(grep -v '^#' .env | xargs)
```

#### 4. API Access (Optional)

If you plan to run models that require API access (e.g., OpenAI), create a `.secrets` file in the root directory and include the required API keys.

## 🌍 World Generation <a name="world-generation"></a>

This section guides you through setting up your environment, downloading the dataset, and generating videos for evaluation using your own world generation models.

#### 1. Environment Setup

First, create and activate the environment for your world generation model, then install required dependencies:

```shell
# Create the environment (example command)
conda create -n world_gen python=3.10
...
# Activate the environment
conda activate world_gen
# Install worldscore dependencies
pip install -e .
```

#### 2. Dataset Download

Download the **WorldScore-Dataset** to the specified directory `DATA_PATH`.

```shell
python download.py
```

This will automatically download and organize the dataset into:

```shell
$DATA_PATH/WorldScore-Dataset
```

Ensure that your `.env` file has correctly defined `DATA_PATH` and you've exported the environment variables as explained in the [Setup](#3-export-the-environment-variables) section.

#### 3. Generating Videos for Evaluation

- ###### Register your model

  Create a configuration file named `model_name.yaml` in the  [config/model_configs](https://github.com/haoyi-duan/WorldScore/tree/main/config/model_configs) directory:

  ```yaml
  model: <model_name>
  
  runs_root: ${oc.env:MODEL_PATH}/<model_name_repo>
  
  resolution: [<W>, <H>]
  generate_type: i2v # or t2v
  
  frames: <frames> # Total number of frames per generation
  fps: <fps> # Frames per second
  ```

- ###### Add model to `modeltype.py`

  Register your model in the file [worldscore/benchmark/utils/modeltype.py](https://github.com/haoyi-duan/WorldScore/blob/main/worldscore/benchmark/utils/modeltype.py) within corresponding model type. We support the following model types:

  - `"threedgen"`: 3D scene generation models
  - `"fourdgen"`: 4D scene generation models
  - `"videogen"`: video generation models

  Example:

  ```yaml
  type2model = {
      "threedgen": [
          "wonderjourney",
          ...
      ],
      "fourdgen": [
          "4dfy",
        	...
      ],
      "videogen": [
          "cogvideox_5b_i2v",
          "model_name",
          ...
      ]
  }
  ```

- ###### Implement your model

  There are two ways of adapting your model to support WorldScore generation. One way is to create a model class  `model_name.py` in [world_generators](https://github.com/haoyi-duan/WorldScore/tree/main/world_generators) to support world generation:

  ```python
  class model_name:
    def __init__(
      self,
      model_name: str,
      generation_type: Literal["t2v", "i2v"],
      **kwargs
    ):
      # Initialize your model
      self.generate = ...
      
    def generate_video(
    	self,
      prompt: str,
      image_path: Optional[str] = None,
    ):
      # Generate frames
      frames = generate(prompt, image_path)
      
      # Must return either: 
      # - List[Image.Image], or 
      # - torch.Tensor of shape [N, 3, H, W] with values in [0, 1]
      return frames
  ```

  Store your model's keyword arguments `model_name.yaml` in the  [world_generators/configs](https://github.com/haoyi-duan/WorldScore/tree/main/world_generators/configs) directory:

  ```yaml
  _target_: world_generators.<model_name>.<model_name>
  model_name:  <model_name>
  generation_type: i2v # or t2v
  # Add any other model-specific keyword arguments here
  **kwargs
  ```

  This way only supports video generation models for now, refer to [Model Families](https://github.com/haoyi-duan/WorldScore/blob/main/world_generators/README.md#model-families) for more examples. For 3D scene generation models and 4D scene generation models that are more complicated to adapt, also refer to [Adaptation](https://github.com/haoyi-duan/WorldScore/blob/main/world_generators/README.md#adaption-for-world-generation-models) for more details.

- ###### Run Generation <a name="run_generation"></a>

  Single-GPU:

  ```shell
  python world_generators/generate_videos.py --model-name <model_name>
  ```

  Multi-GPU with Slurm:

  ```shell
  python world_generators/generate_videos.py \
  	--model_name <model_name> \
  	--use_slurm True \
  	--num_jobs <num_gpu> \
  	--slurm_partition <your_partition> \
  	...
  ```
  
  Here is an overview of [output format](https://github.com/haoyi-duan/WorldScore/blob/main/world_generators/README.md#output_format).


> [!TIP]
>
> For more information on distributed job launching, refer to [submitit](https://github.com/facebookincubator/submitit).

## ✅ Evaluation <a name="evaluation"></a>

#### 1. Environment Setup

- ###### Create a new conda environment

  ```shell
  # Tested on cuda 12.1 version
  export CUDA_HOME=/path/to/cuda-12.1/
  
  conda create -n worldscore python=3.10 && conda activate worldscore
  ```

- ###### Install following key dependencies

  - ###### Droid-SLAM (About 10 Mins)

    ```shell
    conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
    pip install --index-url https://download.pytorch.org/whl/cu121 xformers
    conda install suitesparse -c conda-forge
    pip install open3d tensorboard scipy opencv-python tqdm matplotlib pyyaml
    pip install evo --upgrade --no-binary evo
    pip install gdown
    
    git submodule update --init --recursive thirdparty/DROID-SLAM
    cd thirdparty/DROID-SLAM/
    python setup.py install
    cd ../..
    ```

  - ###### Other dependencies

    ```shell
    pip install yacs loguru einops timm imageio spacy catalogue pyiqa torchmetrics pytorch_lightning cvxpy
    python -m spacy download en_core_web_sm
    ```

  - ###### Grounding-SAM

    ```shell
    git submodule update --init thirdparty/Grounded-Segment-Anything
    cd thirdparty/Grounded-Segment-Anything/
    export AM_I_DOCKER=False
    export BUILD_WITH_CUDA=True
    python -m pip install -e segment_anything
    pip install --no-build-isolation -e GroundingDINO
    cd ../..
    ```

  - ###### SAM2

    ```shell
    git submodule update --init thirdparty/sam2
    cd thirdparty/sam2/
    pip install -e .
    cd ../..
    ```

  - ###### VFIMamba

    ```shell
    pip install causal_conv1d==1.5.0.post8 mamba_ssm==2.2.4

    # If above installation failed, you can try building from source
    git clone https://github.com/state-spaces/mamba.git && cd mamba
    pip install .
    cd ../
    ```

  - ###### Install WorldScore dependencies

    ```shell
    pip install .
    ```

(Optional) Run the following command to check the completeness of world generation:

```shell
worldscore-analysis -cd --model_name <model_name>
```

If incomplete, run [generation](#run_generation) first. If completed, now you can run the WorldScore evaluation to assess your model performance.

#### 2. Download Evaluation Checkpoints

Run the following commands to download all the required model checkpoints:

```sh
wget -q -P ./worldscore/benchmark/metrics/checkpoints https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

wget -q -P ./worldscore/benchmark/metrics/checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

wget -q -P ./worldscore/benchmark/metrics/checkpoints https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip ./worldscore/benchmark/metrics/checkpoints/models.zip -d ./worldscore/benchmark/metrics/checkpoints/

wget -q -P ./worldscore/benchmark/metrics/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
wget -q -P ./worldscore/benchmark/metrics/checkpoints https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt

wget -q -P ./worldscore/benchmark/metrics/checkpoints https://huggingface.co/MCG-NJU/VFIMamba_ckpts/resolve/main/ckpt/VFIMamba.pkl

# Download droid.pth
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh
mv droid.pth ./worldscore/benchmark/metrics/checkpoints/
```

#### 3. Run WorldScore Evaluation

WorldScore evaluates the generated videos using various spatial and temporal metrics. The evaluation pipeline supports both single-GPU and multi-GPU (via Slurm) setups.

Single-GPU:

```sh
python worldscore/run_evaluate.py --model_name <model_name>
```

Multi-GPU with Slurm:

```shell
python worldscore/run_evaluate.py \
	--model_name <model_name> \
	--use_slurm True \
	--num_jobs <num_gpu> \
	--slurm_partition <your_partition> \
	...
```

> [!TIP]
>
> For more information on distributed job launching, refer to [submitit](https://github.com/facebookincubator/submitit).

After evaluation is completed, the results will be saved at `worldscore_output/worldscore.json`.

## 🏆 Leaderboard <a name="leaderboard"></a>

See most updated ranking and numerical results at our Leaderboard <a href='https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard'><img src='https://img.shields.io/badge/Leaderboard-Huggingface-yellow?logo=huggingface&logoColor=yellow'></a> 🥇🥈🥉 . There are 2 options to join WorldScore Leaderboard:

| Sampled by | Evaluated by | Comments                                                     |
| ---------- | ------------ | ------------------------------------------------------------ |
| Your team  | Your team    | **Highly recommended**, you can follow instructions from [Setup Instructions](#setup-instructions), [World Generation](#world-generation ), and [Evaluation](#evaluation), and submit the evaluation result `worldscore_output/worldscore.json` to haoyid@stanford.edu. The evaluation results will be updated to the leaderboard by WorldScore Team. |
| Your team  | WorldScore   | You can also submit your video samples to us for evaluation, but the progress depends on our available time and resources. |

> [!NOTE]
>
> If you choose to join the leaderboard using the first option (submitting evaluation results), make sure to run: 
>
> ```shell
> worldscore-analysis -cs --model_name <model_name>
> ```
>
> to verify the score completeness before submitting. 



## 📈 World Generation Models Info <a name="world-generation-models-info"></a>

| Model Type | Model Name                                                   | Ability | Version    | Resolution | Video Length(s) | FPS  | Frame Number |
| ---------- | ------------------------------------------------------------ | ------- | ---------- | ---------- | --------------- | ---- | ------------ |
| Video      | [Gen-3](https://runwayml.com/)                               | I2V     | 2024.07.01 | 1280x768   | 10              | 24   | 253          |
| Video      | [Hailuo](https://hailuoai.video/)                            | I2V     | 2024.08.31 | 1072x720   | 5.6             | 25   | 141          |
| Video      | [DynamiCrafter](https://doubiiu.github.io/projects/DynamiCrafter/) | I2V     | 2023.10.18 | 1024x576   | 5               | 10   | 50           |
| Video      | [VideoCrafter1-T2V](https://ailab-cvc.github.io/videocrafter1/) | T2V     | 2023.10.30 | 1024x576   | 2               | 8    | 16           |
| Video      | [VideoCrafter1-I2V](https://ailab-cvc.github.io/videocrafter1/) | I2V     | 2023.10.30 | 512x320    | 2               | 8    | 16           |
| Video      | [VideoCrafter2](https://ailab-cvc.github.io/videocrafter2/)  | T2V     | 2024.01.17 | 512x320    | 2               | 8    | 16           |
| Video      | [T2V-Turbo](https://t2v-turbo.github.io/)                    | T2V     | 2024.05.29 | 512x320    | 3               | 16   | 48           |
| Video      | [EasyAnimate](https://easyanimate.github.io/)                | I2V     | 2024.05.29 | 1344x768   | 6               | 8    | 49           |
| Video      | [CogVideoX-T2V](https://github.com/THUDM/CogVideo)           | T2V     | 2024.08.12 | 720x480    | 6               | 8    | 49           |
| Video      | [CogVideoX-I2V](https://github.com/THUDM/CogVideo)           | I2V     | 2024.08.12 | 720x480    | 6               | 8    | 49           |
| Video      | [Allegro](https://github.com/rhymes-ai/Allegro)              | I2V     | 2024.10.20 | 1280x720   | 6               | 15   | 88           |
| Video      | [Vchitect-2.0](https://vchitect.intern-ai.org.cn/)           | T2V     | 2025.01.14 | 768x432    | 5               | 8    | 40           |
| 3D         | [SceneScape](https://scenescape.github.io/)                  | T2V     | 2023.02.02 | 512x512    | 5               | 10   | 50           |
| 3D         | [Text2Room](https://lukashoel.github.io/text-to-room/)       | I2V     | 2023.03.21 | 512x512    | 5               | 10   | 50           |
| 3D         | [LucidDreamer](https://luciddreamer-cvlab.github.io/)        | I2V     | 2023.11.22 | 512x512    | 5               | 10   | 50           |
| 3D         | [WonderJourney](https://kovenyu.com/wonderjourney/)          | I2V     | 2023.12.06 | 512x512    | 5               | 10   | 50           |
| 3D         | [InvisibleStitch](https://research.paulengstler.com/invisible-stitch/) | I2V     | 2024.04.30 | 512x512    | 5               | 10   | 50           |
| 3D         | [WonderWorld](https://kovenyu.com/wonderworld/)              | I2V     | 2024.06.13 | 512x512    | 5               | 10   | 50           |
| 4D         | [4D-fy](https://sherwinbahmani.github.io/4dfy/)              | T2V     | 2023.11.29 | 256x256    | 4               | 30   | 120          |



## ✒️ Citation <a name="citation"></a>

```
@article{duan2025worldscore,
  title={WorldScore: A Unified Evaluation Benchmark for World Generation},
  author={Duan, Haoyi and Yu, Hong-Xing and Chen, Sirui and Fei-Fei, Li and Wu, Jiajun},
  journal={arXiv preprint arXiv:2504.00983},
  year={2025}
}
```
