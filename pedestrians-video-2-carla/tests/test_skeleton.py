"""
Sanity checks to see if the overall flow is working.
"""
from pedestrians_video_2_carla.skeleton import main
import os


def test_flow(test_logs_dir, loss_mode, projection_type):
    """
    Test the overall flow using Linear model.
    """
    main([
        "--data_module_name=Carla2D3D",
        "--model_name=Linear",
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
        "--projection_type={}".format(projection_type)
    ], test_logs_dir)

    experiment_dir = os.path.join(test_logs_dir, "Linear", "version_0")

    # assert the experiments log dir exists
    assert os.path.exists(experiment_dir), 'Experiment logs dir was not created'


def test_flow_needs_confidence(test_logs_dir, projection_type):
    """
    Test the basic flow using Linear model with needs_confidence flag enabled.
    """
    main([
        "--data_module_name=Carla2D3D",
        "--model_name=Linear",
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
        "--projection_type={}".format(projection_type),
        "--needs_confidence"
    ], test_logs_dir)

    experiment_dir = os.path.join(test_logs_dir, "Linear", "version_0")

    # assert the experiments log dir exists
    assert os.path.exists(experiment_dir), 'Experiment logs dir was not created'


def test_renderer(test_logs_dir, renderer):
    """
    Test the renderers using Linear model.
    """
    main([
        "--data_module_name=Carla2D3D",
        "--model_name=Linear",
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
        renderer
    ], test_logs_dir)

    experiment_dir = os.path.join(test_logs_dir, "Linear", "version_0")

    # assert the experiments log dir exists
    assert os.path.exists(experiment_dir), 'Experiment logs dir was not created'

    # assert no video files were created
    if renderer == 'none':
        assert not os.path.exists(os.path.join(
            experiment_dir, "videos")), 'Videos dir was created'


def test_source_videos_jaad(test_logs_dir):
    """
    Test the source videos rendering using JAADOpenPoseDataModule.
    """
    main([
        "--data_module_name=JAADOpenPose",
        "--model_name=Linear",
        "--batch_size=8",
        "--num_workers=0",
        "--clip_length=32",
        "--input_nodes=BODY_25_SKELETON",
        "--output_nodes=CARLA_SKELETON",
        "--max_epochs=1",
        "--limit_train_batches=1",
        "--limit_val_batches=1",
        "--loss_modes=common_loc_2d",
        "--max_videos=4",
        "--renderers",
        "source_videos",
        "--source_videos_dir=/datasets/JAAD/videos"
    ], test_logs_dir)

    video_dir = os.path.join(
        test_logs_dir, "Linear", "version_0", "videos")
    # assert video files were created
    assert os.path.exists(video_dir), 'Videos dir was not created'
