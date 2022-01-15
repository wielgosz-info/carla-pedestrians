# CARLA Pedestrians
Bringing more realistic pedestrians movements into CARLA.

## Cloning
This project contains submodules when there was no other option of getting the code in (not pip-installable code). So to ensure that all models run correctly, clone with:

```sh
git clone --recurse-submodules git@github.com:wielgosz-info/carla-pedestrians.git
```

## (Cumbersome) Running Steps

### Step 0
Copy `.env.template` to `.env` in the 'openpose' folderand adjust the variables (do the same for the 'pedestrians_video_2_carla', especially the path to datasets (e.g. for dataset root `DATASETS_PATH=/datasets` the expected structure would be `/datasets/JAAD`, `/datasets/PIE` etc.). By default, the project assumes [JAAD dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/).

### Step 1
Extract pedestrians skeletons from video clips with [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) using container specified in `openpose/docker-compose.yml`:

```sh
cd openpose
docker-compose -f "docker-compose.yml" --env-file .env up -d --build
docker exec -it carla-pedestrians_openpose_1 /bin/bash
```

Inside container (after reviewing/modifying `extract_poses_from_dataset.sh`):
```sh
cd /app
./extract_poses_from_dataset.sh
```

The generated files will be saved in the `carla-pedestrians_outputs` Docker volume. By default the `extract_poses.sh` scripts tries to use `JAAD` dataset.

### Step 2
Run the CARLA server & the container with our code (`carla-pedestrians_client_1`).

If you don't have the Nvidia GPU, there is CPU-only version available `docker-compose.cpu.yml`.
Please note that currently running CARLA server requires GPU, so without it the `source_carla`
`carla` renderers shouldn't be used, since it would result in errors.

```sh
cd pedestrians-video-2-carla
COMMIT=$(git rev-parse --short HEAD) docker-compose -f "docker-compose.yml" --env-file .env up -d --build
```

### Step 3
Magic up the `/outputs/JAAD/annotations.csv` file. Automation script for that can be found in `pedestrians-video-2-carla/src/pedestrians_video_2_carla/data/utils/jaad_annotations_xml_2_csv.py`.

(For now) it needs to have `video`, `frame`, `x1`,`y1`, `x2`, `y2`, `id`, `action`, `gender`, `age`, `group_size` and `speed` columns, where `x1`,`y1`, `x2`, `y2` define pedestrian bounding box, `id` is the pedestrian id, `action` is what the pedestrian is doing (since right now only the `walking` ones will be used) and `speed` is the car speed category (for now only `stopped` cars will be used). For now we are also only using fragments where `group_size=1`.

### Step 4
Run selected experiment inside `carla-pedestrians_client_1`, e.g.:

```sh
python3 -m pedestrians_video_2_carla \
    --data_module_name=Carla2D3D \
    --model_movements_name=LinearAE \
    --clip_length=180 \
    --batch_size=64 \
    --num_workers=32 \
    --input_nodes=CARLA_SKELETON \
    --output_nodes=CARLA_SKELETON \
    --loss_modes rot_3d loc_2d_3d \
    --renderers source_carla input_points projection_points carla \
    --max_videos=8 \
    --log_every_n_steps=50 \
    --flush_logs_every_n_steps=100 \
    --check_val_every_n_epoch=10 \
    --max_epochs=300 \
    --gpus=0,1 \
    --accelerator=ddp \
    --limit_train_batches=32 \
    --log_every_n_steps=16 \
    --flush_logs_every_n_steps=64
```

For full list of available options run:

```sh
python3 -m pedestrians_video_2_carla -h
```

Please note that data module and model specific options may change if you switch the DataModule or Model.

## Reference skeletons
Reference skeleton data in `pedestrians-video-2-carla/src/pedestrians_video_2_carla/skeletons/reference` are extracted form [CARLA project Walkers *.uasset files](https://bitbucket.org/carla-simulator/carla-content).

## License
Our code is released under [MIT License](https://github.com/wielgosz-info/carla-pedestrians/blob/main/LICENSE)

This project uses (and is developed to work with) [CARLA Simulator](https://carla.org/), which is released under [MIT License](https://github.com/carla-simulator/carla/blob/master/LICENSE).

This project uses videos and annotations from [JAAD dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/), created by Amir Rasouli, Iuliia Kotseruba, and John K. Tsotsos, to extract pedestrians movements and attributes. The videos and annotations are released under [MIT License](https://github.com/ykotseruba/JAAD/blob/JAAD_2.0/LICENSE).

This project uses [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), created by Ginés Hidalgo, Zhe Cao, Tomas Simon, Shih-En Wei, Yaadhav Raaj, Hanbyul Joo, and Yaser Sheikh, to extract pedestrians skeletons from videos. OpenPose has its [own licensing](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE) (basically, academic or non-profit organization noncommercial research use only).

This project uses software, models and datasets from [Max-Planck Institute for Intelligent Systems](https://is.mpg.de/en), namely [VPoser: Variational Human Pose Prior for Body Inverse Kinematics](https://github.com/nghorbani/human_body_prior), [Body Visualizer](https://github.com/nghorbani/body_visualizer), [Configer](https://github.com/MPI-IS/configer) and [Perceiving Systems Mesh Package](https://github.com/MPI-IS/mesh), which have their own licenses (non-commercial scientific research purposes, see each repo for details). The models can be downloaded from ["Expressive Body Capture: 3D Hands, Face, and Body from a Single Image" website](https://smpl-x.is.tue.mpg.de). Required are the "SMPL-X with removed head bun" or other SMPL-based model that can be fed into [BodyModel](https://github.com/nghorbani/human_body_prior/blob/master/src/human_body_prior/body_model/body_model.py) - right now our code utilizes only [first 22 common SMPL basic joints](https://meshcapade.wiki/SMPL#related-models-the-smpl-family#skeleton-layout). For VPoser, the "VPoser v2.0" model is used. Both downloaded models need to be put in `pedestrians-video-2-carla/models` directory. If using other SMPL models, the defaults in `pedestrians-video-2-carla/src/pedestrians_video_2_carla/renderers/smpl_renderer.py` may need to be modified. SMPL-compatible datasets can be obtained from [AMASS: Archive of Motion Capture As Surface Shapes](https://amass.is.tue.mpg.de/). Each available dataset has its own license / citing requirements. During the development of this project, we mainly used [CMU](http://mocap.cs.cmu.edu/) and [Human Eva](http://humaneva.is.tue.mpg.de/) SMPL-X Gender Specific datasets.

## Funding

|                                                                                                                                                                                                   |                                                                                                                                                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://github.com/wielgosz-info/pedestrians-video-2-carla/blob/main/docs/_static/images/logos/Logo Tecniospring INDUSTRY_white.JPG" alt="Tecniospring INDUSTRY" style="height: 24px;"> | <img src="https://github.com/wielgosz-info/pedestrians-video-2-carla/blob/main/docs/_static/images/logos/ACCIO_horizontal.PNG" alt="ACCIÓ Government of Catalonia" style="height: 35px;"> | <img src="https://github.com/wielgosz-info/pedestrians-video-2-carla/blob/main/docs/_static/images/logos/EU_emblem_and_funding_declaration_EN.PNG" alt="This project has received funding from the European Union's Horizon 2020 research and innovation programme under Marie Skłodowska-Curie grant agreement No. 801342 (Tecniospring INDUSTRY) and the Government of Catalonia's Agency for Business Competitiveness (ACCIÓ)." style="height: 70px;"> |
