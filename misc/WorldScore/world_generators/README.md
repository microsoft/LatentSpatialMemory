# WorldScore

## 🧩 Adaptation for World Generation Models <a name="adaptation-for-world-generation-models"></a>

Here we provide another way to adapt the models to support WorldScore generation. This hard-coded way can support 3D scene generation models, 4D scene generation models, and video generation models.

#### WorldScore dependencies

```python
from worldscore.benchmark.utils.utils import check_model, type2model, get_model2type
from worldscore.benchmark.helpers import GetHelpers
```

#### Hard-Code

```python
def generate(start_keyframe, inpainting_prompt_list, cameras, cameras_interp, helper):
    # Generate frames
    frames = model(start_keyframe, inpainting_prompt_list, cameras, cameras_interp)
    
    # Must return either: 
    # - List[Image.Image], or 
    # - torch.Tensor of shape [N, 3, H, W] with values in [0, 1]
    helper.save(frames)

if __name__ == "__main__":
    model_name = ...

    assert check_model(model_name), 'Model not exists!'
    model_type = get_model2type(type2model)[model_name]
    if model_type == "threedgen":
        visual_movement_list = ["static"]
    else:
        visual_movement_list = ["static", "dynamic"]

    for visual_movement in visual_movement_list:       
        dataloader, helper = GetHelpers(model_name, visual_movement)
        
        for data in dataloader:
            start_keyframe, inpainting_prompt_list, cameras, cameras_interp = helper.adapt(data)
            generate(start_keyframe, inpainting_prompt_list, cameras, cameras_interp, helper)
```

#### Examples

- WonderJourney

```
git clone https://github.com/KovenYu/WonderJourney.git
cd WonderJourney
```

Then copy [run_wj_worldscore.py](https://github.com/haoyi-duan/WorldScore/blob/main/world_generators/run_wj_worldscore.py) to the root directory.

- WonderWorld

```
git clone https://github.com/KovenYu/WonderWorld.git
cd WonderWorld
```

Then copy [run_ww_worldscore.py](https://github.com/haoyi-duan/WorldScore/blob/main/world_generators/run_ww_worldscore.py) to the root directory.

## 🎬 Model Families <a name="model-families"></a>

We currently provide some examples of video generation models that we have included in the WorldScore benchmark. We welcome further participation in the WorldScore challenge to push the boundaries of world generation.

#### CogVideo

We evaluate several image-to-video and text-to-video models from the [CogVideoX](https://github.com/THUDM/CogVideo) family. To generate videos, first create a virtual environment.

```sh
python -m venv .venv/cogvideo
source .venv/cogvideo/bin/activate
pip install -r requirements/cogvideo.txt
pip install .
```

Then we generate videos for different model checkpoints as below.
```sh
# CogVideoX-5B text-to-video model.
python world_generators/generate_videos.py --model-name cogvideox_5b_t2v

# CogVideoX-2B text-to-video model.
python world_generators/generate_videos.py --model-name cogvideox_2b_t2v

# CogVideoX-5B image-to-video model.
python world_generators/generate_videos.py --model-name cogvideox_5b_i2v
```

#### VideoCrafter

We evaluate several image-to-video and text-to-video models from the [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter) model family. First, install the dependencies for the VideoCrafter model.

```sh
git submodule update --init thirdparty/VideoCrafter
python -m venv .venv/crafter
source .venv/crafter/bin/activate
pip install -r requirements/crafter.txt
pip install .
```

Download model checkpoints:
```sh
wget -O world_generators/checkpoints/videocrafter_t2v_1024_v1.ckpt https://huggingface.co/VideoCrafter/Text2Video-1024/resolve/main/model.ckpt
wget -O world_generators/checkpoints/videocrafter_i2v_512_v1.ckpt https://huggingface.co/VideoCrafter/Image2Video-512/resolve/main/model.ckpt
wget -O world_generators/checkpoints/videocrafter_t2v_512_v2.ckpt https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt
```

Generate the videos:

```sh
# VideoCrafter1 text-to-video model
python world_generators/generate_videos.py --model-name videocrafter1_t2v

# VideoCrafter1 image-to-video model
python world_generators/generate_videos.py --model-name videocrafter1_i2v

# VideoCrafter2 text-to-video model
python world_generators/generate_videos.py --model-name videocrafter2_t2v
```

#### DynamiCrafter

We evaluate two image-to-video checkpoints from the [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter) model family. First, install the dependencies (same as VideoCrafter).

```sh
git submodule update --init thirdparty/DynamiCrafter
python -m venv .venv/crafter
source .venv/crafter/bin/activate
pip install -r requirements/crafter.txt
pip install .

# download checkpoints for dynamicrafter_512_i2v
wget -O world_generators/checkpoints/dynamicrafter_512_v1.ckpt https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt

# download checkpoints for dynamicrafter_1024_i2v
wget -O world_generators/checkpoints/dynamicrafter_1024_i2v.ckpt https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt
```

Then we can generate the videos for different checkpoints using the following scripts:

```sh
# DynamiCrafter_512 image-to-video model
python world_generators/generate_videos.py --model-name dynamicrafter_512_i2v

# DynamiCrafter_1024 image-to-video model
python world_generators/generate_videos.py --model-name dynamicrafter_1024_i2v
```

#### T2V-Turbo

We evaluate [T2V-Turbo](https://github.com/Ji4chenLi/t2v-turbo.git) model. T2V-Turbo also relies on the flash-attn repository, so we need to install both dependencies.

```sh
git submodule update --init thirdparty/t2v_turbo
python -m venv .venv/t2vturbo
source .venv/t2vturbo/bin/activate
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/t2vturbo.txt
pip install flash-attn --no-build-isolation
pip install thirdparty/flash-attention/csrc/fused_dense_lib 
pip install thirdparty/flash-attention/csrc/layer_norm
pip install .
```

Download the lora checkpoints. Note that it uses the same model as VideoCrafter.
```sh
# download checkpoints of T2V-Turbo (VC2)
wget -O world_generators/checkpoints/videocrafter_t2v_512_v2.ckpt https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt
wget -O world_generators/checkpoints/t2v_turbo_unet_lora.pt https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2/resolve/main/unet_lora.pt
```

Generate videos:
```sh
python world_generators/generate_videos.py --model-name t2v_turbo_t2v
```

#### Vchitect-2.0

We evaluate [Vchitect-2.0](https://github.com/Vchitect/Vchitect-2.0) model. First, install dependencies. 

```shell
git submodule update --init thirdparty/Vchitect2
python -m venv .venv/vchitect
source .venv/vchitect/bin/activate
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/vchitect.txt
pip install .
```

Please note that while our script will automatically download the checkoint, you need to login to huggingface (run `huggingface-cli login`) and accept the [Vchitect-2.0](https://huggingface.co/Vchitect/Vchitect-2.0-2B) user agreement. Finally, download the checkpoint to the checkpoints folder:

```sh
huggingface-cli download Vchitect/Vchitect-2.0-2B --local-dir world_generators/checkpoints/vchitect
```

And then generate videos using the script below.
```sh
python world_generators/generate_videos.py --model-name vchitect_2_t2v
```

#### EasyAnimate

We evaluate [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) model. To generate the videos, we first install the dependencies for EasyAnimate as well as dependencies needed to generate the videos. 

```shell
git submodule update --init thirdparty/EasyAnimate
mv thirdparty/EasyAnimate/__init__.py thirdparty/EasyAnimate/__init__.py.bak
python -m venv .venv/easyanimate
source .venv/easyanimate/bin/activate
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/easyanimate.txt
pip install .
```

Download the [weights](https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh-InP) to the specified path:

```shell
mkdir -p world_generators/checkpoints/easyanimate
huggingface-cli download alibaba-pai/EasyAnimateV5-12b-zh-InP --local-dir world_generators/checkpoints/easyanimate
```

And then generate videos using the script below: 

```shell
python world_generators/generate_videos.py --model-name easyanimate_i2v
```
#### Allegro-TI2V

We evaluate [Allegro-TI2V](https://github.com/rhymes-ai/Allegro) model. To generate the videos, we first install the dependencies for Allegro-TI2V as well as dependencies needed to generate the videos. 

```shell
git submodule update --init thirdparty/Allegro
python -m venv .venv/allegro
source .venv/allegro/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/allegro.txt
pip install -e .
```

Download the [Allegro-TI2V model weights](https://huggingface.co/rhymes-ai/Allegro-TI2V) using the following command:
```shell
huggingface-cli download rhymes-ai/Allegro-TI2V --local-dir world_generators/checkpoints/allegro_ti2v
```

Finally, you generate videos using the following script: 
```shell
python world_generators/generate_videos.py --model-name allegro_ti2v
```

#### GEN-3

We evaluate [Gen-3 Alpha Turbo](https://runwayml.com/). To generate the videos, we first install the dependencies for Gen-3 as well as dependencies needed to generate the videos. 

```shell
python -m venv .venv/gen_3
source .venv/gen_3/bin/activate
pip install -r requirements/gen_3.txt
pip install -e .
```

Save api in `.secret`

```shell
RUNWAYML_API_SECRET="YOUR_GEN3_API_KEY"
```

and run

```shell
export $(grep -v '^#' .secrets | xargs)
```

Then generate videos using the following script: 

```shell
python world_generators/generate_videos.py --model-name gen_3_i2v
```

#### MINIMAX

We evaluate [MINIMAX](). To generate the videos, we first install the dependencies for MINIMAX as well as dependencies needed to generate the videos:

```shell
python -m venv .venv/minimax
source .venv/minimax/bin/activate
pip install -r requirements/minimax.txt
pip install -e .
```

Save api in `.secret`:

```shell
MINIMAX_API_KEY="YOUR_MINIMAX_API_KEY"
```

and run

```shell
export $(grep -v '^#' .secrets | xargs)
```

Then generate videos using the following script: 

```shell
python world_generators/generate_videos.py --model-name minimax_i2v
```

#### Wan2.1

We evaluate [Wan2.1](https://github.com/Wan-Video/Wan2.1). To generate videos, first create a virtual environment.

```sh
git submodule update --init thirdparty/Wan2.1
python -m venv .venv/wan
source .venv/wan/bin/activate
pip install -r requirements/wan.txt
pip install .
```

Download the model using higgingface-cli:

```shell
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./models/Wan2.1-I2V-14B-480P
```

Then we generate videos using the command below:

```sh
python world_generators/generate_videos.py --model-name wan2.1_i2v
```

## 📁 Output Format <a name="output_format"></a>

Once the generation is complete, the output are stored in `MODEL_PATH/<model_name>/worldscore_output` and have the following structure:

```sh
📂 worldscore_output
├── 📂 static
│   ├── 📂 photorealistic
│   │   ├── 📂 indoor
│   │   │   ├── 📂 dining_spaces
│   │   │   │   ├── 📂 000
│   │   │   │   │   ├── 📂 frames
│   │   │   │   │   │   ├── 🖼️ 000.png
│   │   │   │   │   │   ├── 🖼️ 001.png
│   │   │   │   │   │   ├── 🖼️ 002.png
│   │   │   │   │   │   └── ⋮
│   │   │   │   │   ├── 📂 videos
│   │   │   │   │   │   ├── 🎬 output.mp4
│   │   │   │   │   │   └── ⋮
│   │   │   │   │   ├── 📄 camera_data.json
│   │   │   │   │   └── 📄 image_data.json
│   │   │   │   └── ⋮ 
│   │   │   └── ⋮ 
│   │   └── 📂 outdoor
│   │       └── ⋮ 
│   └── 📂 stylized
│       ├── 📂 indoor
│       │   └── ⋮ (similar structure as photorealistic)
│       └── 📂 outdoor
│           └── ⋮
└── 📂 dynamic
    ├── 📂 photorealistic
    │    ├── 📂 articulated
    │    │   ├── 📂 000
    │    │   │   ├── 📂 frames
    │    │   │   │   ├── 🖼️ 000.png
    │    │   │   │   ├── 🖼️ 001.png
    │    │   │   │   └── ⋮
    │    │   │   ├── 📂 videos
    │    │   │   │   ├── 🎬 output.mp4
    │    │   │   │   └── ⋮
    │    │   │   └── 📄 image_data.json
    │    │   └── ⋮
    │    ├── 📂 deformable
    │    │   └── ⋮ (similar structure as articulated)
    │    ├── 📂 fluid
    │    │   └── ⋮
    │    ├── 📂 rigid
    │    │   └── ⋮
    │    └── 📂 multi-motion
    │        └── ⋮
    └── 📂 stylized
        └── ⋮ (similar structure as photorealistic)
```
