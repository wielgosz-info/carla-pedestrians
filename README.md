# CARLA Pedestrians
Bringing more realistic pedestrians movements into CARLA.

## (Cumbersome) Running Steps

### Step 0
Copy `.env.template` to `.env` and adjust the variables, especially the path to datasets (e.g. for dataset root `OPENPOSE_DATASETS_PATH=/datasets` the expected structure would be `/datasets/JAAD`, `/datasets/PIE` etc.).

### Step 1
Extract pedestrians skeletons from video clips using container specified in `docker-compose.openpose.yml`:

```sh
COMMIT=$(git rev-parse --short HEAD) docker-compose -f "docker-compose.openpose.yml" --env-file .env up -d --build
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

### More steps in progress...

## Reference skeletons
Reference skeleton data in `pedestrians-video-2-carla/src/pedestrians_video_2_carla/reference_skeletons` are extracted form [CARLA project Walkers *.uasset files](https://bitbucket.org/carla-simulator/carla-content).

## License
MIT License

Please be aware that this project uses [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract pedestrians skeletons from videos, which has its [own licensing](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE) (basically, academic or non-profit organization noncommercial research use only).