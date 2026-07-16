# Data Processing

This directory contains the data preparation pipeline for Mirage training data. Most runtime options are defined in `data_config.py`.

## Workflow

1. Configure paths and sampling options in `data_config.py`.
   - `video_dirs` points to raw source videos.
   - `output_root` is the sample directory, usually `data/train`.
   - Clip length, FPS, resolution, naming style, and VAE paths are also configured there.

2. Collect fixed-length clips.

   ```bash
   python -m data_process.run_video_collect
   ```

   This creates numbered sample folders such as `data/train/00000000/`, writes `clip.mp4`, records `source_video_path.txt`, and precomputes target-frame metadata in `train_sample.json`.

3. Use ViPE to extract geometry information for every smaple.

   Each folder must contain assets aligned with `clip.mp4`:

   - `mask.zip`: foreground or dynamic-object masks.
   - `depth.zip`: depth maps.
   - `pose.npz`: camera-to-world poses with `inds` and `data`.
   - `intrinsics.npz`: pinhole intrinsics with `inds` and `data`.

4. Build training samples.

   ```bash
   python -m data_process.run_pipeline
   ```

5. Generate captions.

   ```bash
   python -m data_process.run_video_captioning \
     --input-root data/train \
     --video-keys train_target_rgb,clip \
     --skip-existing
   ```

   Captions are written next to the videos as `.txt` files.

6. Encode video latents.

   ```bash
   python -m data_process.run_video_vae_encode \
     --input-root data/train \
     --video-keys train_preceding_rgb,train_target_rgb,train_reference_rgb \
     --skip-existing
   ```

7. Pack trainable samples.

   ```bash
   python -m data_process.pack_to_lmdb --data-root data/train
   ```

   The default output path is derived from the data root, for example `data/train_lmdb`.
