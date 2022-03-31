# CARLA Pedestrians
Bringing more realistic pedestrians movements into CARLA.

## Cloning
This project contains submodules when there was no other option of getting the code in (not pip-installable code). So to ensure that all models run correctly, clone with:

```sh
git clone --recurse-submodules git@github.com:wielgosz-info/carla-pedestrians.git
```

## Running Steps

### Step 0
Copy `.env.template` to `.env` in the `openpose`, `pedestrians-video-2-carla`, `pedestrians-scenarios` folders and adjust the variables, especially the path to datasets (e.g. for dataset root `VIDEO2CARLA_DATASETS_PATH=/datasets` the expected structure would be `/datasets/JAAD`, `/datasets/PIE` etc.).

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
Run the CARLA server & the containers with our code. For convenience, there is a `compose-up.sh` script provided,
that brings together the multiple `docker-compose.yml` files from submodules and sets the common env variables.

When using NVIDIA GPU and some UNIX-like system, you can simply run:
```sh
./compose-up.sh
```

When using CPU, you need to specify `PLATFORM=cpu` or modify the script.
Additionally, on MacOS when using Docker Desktop the default GROUP_ID and SHM_SIZE will not work,
so they need to be set manually. The resulting example command to run on MacOS is:
```sh
PLATFORM=cpu GROUP_ID=1000 SHM_SIZE=2147483648 ./compose-up.sh
```

For details about running each individual container, see the relevant `README.md` files:
- [pedestrians-video-2-carla](https://github.com/wielgosz-info/pedestrians-video-2-carla/blob/main/README.md)
- [pedestrians-scenarios](https://github.com/wielgosz-info/pedestrians-scenarios/blob/main/README.md)

To quickly bring down all the containers in the `carla-pedestrians` project, use:

```sh
docker-compose down --remove-orphans
```

## Reference skeletons
Reference skeleton data in `pedestrians-video-2-carla/src/pedestrians_video_2_carla/data/carla/files` are extracted form [CARLA project Walkers *.uasset files](https://bitbucket.org/carla-simulator/carla-content).

## License
Our code is released under [MIT License](https://github.com/wielgosz-info/carla-pedestrians/blob/main/LICENSE).

The most up-to-date third-party info can be found in the submodules repositories, but here is a non-exhaustive list:

This project uses (and is developed to work with) [CARLA Simulator](https://carla.org/), which is released under [MIT License](https://github.com/carla-simulator/carla/blob/master/LICENSE).

This project uses videos and annotations from [JAAD dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/), created by Amir Rasouli, Iuliia Kotseruba, and John K. Tsotsos, to extract pedestrians movements and attributes. The videos and annotations are released under [MIT License](https://github.com/ykotseruba/JAAD/blob/JAAD_2.0/LICENSE).

This project uses [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), created by Ginés Hidalgo, Zhe Cao, Tomas Simon, Shih-En Wei, Yaadhav Raaj, Hanbyul Joo, and Yaser Sheikh, to extract pedestrians skeletons from videos. OpenPose has its [own licensing](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE) (basically, academic or non-profit organization noncommercial research use only).

This project uses software, models and datasets from [Max-Planck Institute for Intelligent Systems](https://is.mpg.de/en), namely [VPoser: Variational Human Pose Prior for Body Inverse Kinematics](https://github.com/nghorbani/human_body_prior), [Body Visualizer](https://github.com/nghorbani/body_visualizer), [Configer](https://github.com/MPI-IS/configer) and [Perceiving Systems Mesh Package](https://github.com/MPI-IS/mesh), which have their own licenses (non-commercial scientific research purposes, see each repo for details). The models can be downloaded from ["Expressive Body Capture: 3D Hands, Face, and Body from a Single Image" website](https://smpl-x.is.tue.mpg.de). Required are the "SMPL-X with removed head bun" or other SMPL-based model that can be fed into [BodyModel](https://github.com/nghorbani/human_body_prior/blob/master/src/human_body_prior/body_model/body_model.py) - right now our code utilizes only [first 22 common SMPL basic joints](https://meshcapade.wiki/SMPL#related-models-the-smpl-family#skeleton-layout). For VPoser, the "VPoser v2.0" model is used. Both downloaded models need to be put in `pedestrians-video-2-carla/models` directory. If using other SMPL models, the defaults in `pedestrians-video-2-carla/src/pedestrians_video_2_carla/data/smpl/constants.py` may need to be modified. SMPL-compatible datasets can be obtained from [AMASS: Archive of Motion Capture As Surface Shapes](https://amass.is.tue.mpg.de/). Each available dataset has its own license / citing requirements. During the development of this project, we mainly used [CMU](http://mocap.cs.cmu.edu/) and [Human Eva](http://humaneva.is.tue.mpg.de/) SMPL-X Gender Specific datasets.


## Funding

|                                                                                                                              |                                                                                                                      |                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <img src="docs/_static/images/logos/Logo Tecniospring INDUSTRY_white.JPG" alt="Tecniospring INDUSTRY" style="height: 24px;"> | <img src="docs/_static/images/logos/ACCIO_horizontal.PNG" alt="ACCIÓ Government of Catalonia" style="height: 35px;"> | <img src="docs/_static/images/logos/EU_emblem_and_funding_declaration_EN.PNG" alt="This project has received funding from the European Union's Horizon 2020 research and innovation programme under Marie Skłodowska-Curie grant agreement No. 801342 (Tecniospring INDUSTRY) and the Government of Catalonia's Agency for Business Competitiveness (ACCIÓ)." style="height: 70px;"> |

