# CARLA Pedestrians
Bringing more realistic pedestrians movements into CARLA.

## (Cumbersome) Running Steps

### Step 0
Copy `.env.template` to `.env` and adjust the variables, especially the path to datasets (e.g. for dataset root `OPENPOSE_DATASETS_PATH=/datasets` the expected structure would be `/datasets/JAAD`, `/datasets/PIE` etc.). By default, the project assumes [JAAD dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/).

### Step 1
Extract pedestrians skeletons from video clips with [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) using container specified in `docker-compose.openpose.yml`:

```sh
docker-compose -f "docker-compose.openpose.yml" --env-file .env up -d --build
docker exec -it carla-pedestrians_openpose_1 /bin/bash
```

Inside container (after reviewing/modifying `extract_poses.sh`):
```sh
cd /app
./extract_poses.sh
```

The generated files will be saved in the `carla-pedestrians_outputs` Docker volume. By default the `extract_poses.sh` scripts tries to use `JAAD` dataset.

### Step 2
Magic up the `/outputs/JAAD/annotations.csv` file. Automation script for that may be coming one day.

(For now) it needs to have `video`, `frame`, `x1`,`y1`, `x2`, `y2`, `id`, `action`, `gender`, `age`, and `speed` columns, where `x1`,`y1`, `x2`, `y2` define pedestrian bounding box, `id` is the pedestrian id, `action` is what the pedestrian is doing (since right now only the `walking` ones will be used) and `speed` is the car speed category (for now only `stopped` cars will be used).

### Step 3
Run the CARLA server & the container with our code (`carla-pedestrians_client_1`).

```sh
COMMIT=$(git rev-parse --short HEAD) docker-compose -f "docker-compose.yml" --env-file .env up -d --build
```

### More steps in progress...

## Reference skeletons
Reference skeleton data in `pedestrians-video-2-carla/src/pedestrians_video_2_carla/reference_skeletons` are extracted form [CARLA project Walkers *.uasset files](https://bitbucket.org/carla-simulator/carla-content).

## License
[MIT License](https://github.com/wielgosz-info/carla-pedestrians/blob/main/LICENSE)

This project uses videos and annotations from [JAAD dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/), created by Amir Rasouli, Iuliia Kotseruba, and John K. Tsotsos, to extract pedestrians movements and attributes. The videos and annotations are released under [MIT License](https://github.com/ykotseruba/JAAD/blob/JAAD_2.0/LICENSE).

This project uses [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), created by Ginés Hidalgo, Zhe Cao, Tomas Simon, Shih-En Wei, Yaadhav Raaj, Hanbyul Joo, and Yaser Sheikh, to extract pedestrians skeletons from videos. OpenPose has its [own licensing](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE) (basically, academic or non-profit organization noncommercial research use only).

## Funding

<div>
    <img src="pedestrians-video-2-carla/docs/_static/images/logos/Logo Tecniospring INDUSTRY_white.JPG" alt="Tecniospring INDUSTRY" style="height: 28px;">
    <img src="pedestrians-video-2-carla/docs/_static/images/logos/ACCIO_horizontal.PNG" alt="ACCIÓ Government of Catalonia" style="height: 40px;">
    <img src="pedestrians-video-2-carla/docs/_static/images/logos/EU_emblem_and_funding_declaration_EN.PNG" alt="This project has received funding from the European Union's Horizon 2020 research and innovation programme under Marie Skłodowska-Curie grant agreement No. 801342 (Tecniospring INDUSTRY) and the Government of Catalonia's Agency for Business Competitiveness (ACCIÓ)." style="height: 80px;">
</div>
