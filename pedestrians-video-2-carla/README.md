# pedestrians-video-2-carla

This is a part of the bigger project to bring the more realistic pedestrian movements to CARLA.
It isn't intended for fully standalone use. Please see the main project README.md for details.

## Running

Inside the container, you can run the following command to start the training:

```sh
python3 -m pedestrians_video_2_carla \
  --data_module_name=Carla2D3D \
  --model_name=LinearAE \
  --batch_size=256 \
  --num_workers=32 \
  --clip_length=180 \
  --input_nodes=CARLA_SKELETON \
  --output_nodes=CARLA_SKELETON \
  --max_epochs=500 \
  --loss_modes=loc_2d_3d \
  --max_videos=8 \
  --renderers none \
  --check_val_every_n_epoch=10 \
  --gpus=0,1 \
  --accelerator=ddp \
  --limit_train_batches=32 \
  --log_every_n_steps=16 \
  --flush_logs_every_n_steps=64
```

Full list of options is available by running:

```sh
python3 -m pedestrians_video_2_carla --help
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
