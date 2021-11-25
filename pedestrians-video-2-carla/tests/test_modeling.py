"""
Sanity checks to see if the overall flow is working.
"""
from pedestrians_video_2_carla.modeling import main
import os
import shutil
import glob


def test_flow(test_logs_dir, test_outputs_dir, loss_mode, movements_output_type):
    """
    Test the overall flow using Linear model.
    """
    main([
        "--data_module_name=Carla2D3D",
        "--movements_model_name=Linear",
        "--batch_size=2",
        "--num_workers=0",
        "--clip_length=32",
        "--input_nodes=CARLA_SKELETON",
        "--output_nodes=CARLA_SKELETON",
        "--max_epochs=1",
        "--limit_val_batches=1",
        "--limit_train_batches=1",
        "--loss_modes",
        loss_mode,
        "--renderers",
        "none",
        "--movements_output_type={}".format(movements_output_type),
        "--outputs_dir={}".format(test_outputs_dir),
        "--logs_dir={}".format(test_logs_dir)
    ])

    experiment_dir = os.path.join(
        test_logs_dir, "Carla2D3DDataModule", "ZeroTrajectory", "Linear", "version_0")

    # assert the experiments log dir exists
    assert os.path.exists(experiment_dir), 'Experiment logs dir was not created'


def test_flow_needs_confidence(test_logs_dir, test_outputs_dir, movements_output_type):
    """
    Test the basic flow using Linear model with needs_confidence flag enabled.
    """
    main([
        "--data_module_name=Carla2D3D",
        "--movements_model_name=Linear",
        "--batch_size=2",
        "--num_workers=0",
        "--clip_length=32",
        "--input_nodes=CARLA_SKELETON",
        "--output_nodes=CARLA_SKELETON",
        "--max_epochs=1",
        "--limit_val_batches=1",
        "--limit_train_batches=1",
        "--loss_modes",
        "common_loc_2d",
        "--renderers",
        "none",
        "--movements_output_type={}".format(movements_output_type),
        "--needs_confidence",
        "--outputs_dir={}".format(test_outputs_dir),
        "--logs_dir={}".format(test_logs_dir)
    ])

    experiment_dir = os.path.join(
        test_logs_dir, "Carla2D3DDataModule", "ZeroTrajectory", "Linear", "version_0")

    # assert the experiments log dir exists
    assert os.path.exists(experiment_dir), 'Experiment logs dir was not created'


def test_renderer(test_logs_dir, test_outputs_dir, renderer):
    """
    Test the renderers using Linear model.
    """
    main([
        "--data_module_name=Carla2D3D",
        "--movements_model_name=Linear",
        "--batch_size=2",
        "--num_workers=0",
        "--clip_length=32",
        "--input_nodes=CARLA_SKELETON",
        "--output_nodes=CARLA_SKELETON",
        "--max_epochs=1",
        "--limit_val_batches=1",
        "--limit_train_batches=1",
        "--loss_modes",
        "common_loc_2d",
        "--renderers",
        renderer,
        "--outputs_dir={}".format(test_outputs_dir),
        "--logs_dir={}".format(test_logs_dir)
    ])

    experiment_dir = os.path.join(
        test_logs_dir, "Carla2D3DDataModule", "ZeroTrajectory", "Linear", "version_0")

    # assert the experiments log dir exists
    assert os.path.exists(experiment_dir), 'Experiment logs dir was not created'

    # assert no video files were created
    if renderer == 'none':
        assert not os.path.exists(os.path.join(
            experiment_dir, "videos")), 'Videos dir was created'


def test_source_videos_jaad(test_logs_dir, test_outputs_dir):
    """
    Test the source videos rendering using JAADOpenPoseDataModule.
    """
    # JAADOpenPoseDataModule will look for the subsets in the tmp directory
    # and fail if it can't find the required files.
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), 'data', 'JAADOpenPoseDataModule'),
        test_outputs_dir,
        dirs_exist_ok=True
    )

    # We're not going to include the videos in the repo, so optionally provide the path
    source_videos_dir = os.getenv('JAAD_SOURCE_VIDEOS_DIR', '/datasets/JAAD/videos')

    main([
        "--data_module_name=JAADOpenPose",
        "--movements_model_name=Linear",
        "--batch_size=8",
        "--num_workers=0",
        "--clip_length=32",
        "--input_nodes=BODY_25_SKELETON",
        "--output_nodes=CARLA_SKELETON",
        "--max_epochs=0",
        "--limit_train_batches=0",
        "--limit_val_batches=1",
        "--loss_modes=common_loc_2d",
        "--max_videos=4",
        "--renderers",
        "source_videos",
        "--source_videos_dir={}".format(source_videos_dir),
        "--outputs_dir={}".format(test_outputs_dir),
        "--openpose_dir={}".format(test_outputs_dir),
        "--logs_dir={}".format(test_logs_dir)
    ])

    video_dir = os.path.join(
        test_logs_dir, "JAADOpenPoseDataModule", "ZeroTrajectory", "Linear", "version_0", "videos", "val")

    assert os.path.exists(video_dir), 'Videos dir was not created'

    videos = glob.glob(os.path.join(video_dir, '**', '*.mp4'))

    assert len(videos) == 4, 'Video files were not created'


def test_models(test_logs_dir, test_outputs_dir, movements_model_name, trajectory_model_name):
    """
    Test the overall flow using Linear model.
    """
    main([
        "--data_module_name=Carla2D3D",
        "--movements_model_name={}".format(movements_model_name),
        "--trajectory_model_name={}".format(trajectory_model_name),
        "--batch_size=2",
        "--num_workers=0",
        "--clip_length=32",
        "--input_nodes=CARLA_SKELETON",
        "--output_nodes=CARLA_SKELETON",
        "--max_epochs=1",
        "--limit_val_batches=1",
        "--limit_train_batches=1",
        "--loss_modes",
        "common_loc_2d",
        "--renderers",
        "none",
        "--outputs_dir={}".format(test_outputs_dir),
        "--logs_dir={}".format(test_logs_dir)
    ])
