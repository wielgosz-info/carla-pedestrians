"""
Sanity checks to see if the overall flow is working.
"""
from pedestrians_video_2_carla.skeleton import main
import os


def test_carla_linear(test_logs_dir):
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
        "--loss_modes",
        "common_loc_2d",
        "--renderers",
        "none",
        "--limit_val_batches=1",
        "--limit_train_batches=1"
    ], test_logs_dir)

    experiment_dir = os.path.join(test_logs_dir, "Linear", "version_0")

    # assert the experiments log dir exists
    assert os.path.exists(experiment_dir), 'Experiment logs dir was not created'

    # assert no video files were created
    assert not os.path.exists(os.path.join(
        experiment_dir, "videos")), 'Videos dir was created'


def test_jaad_baseline(test_logs_dir):
    """
    Test the overall flow using JAADOpenPoseDataModule, ProjectionTypes.absolute_loc model and source videos rendering.
    """
    main([
        "--data_module_name=JAADOpenPose",
        "--model_name=Baseline3DPose",
        "--batch_size=8",
        "--num_workers=0",
        "--clip_length=32",
        "--input_nodes=BODY_25_SKELETON",
        "--output_nodes=CARLA_SKELETON",
        "--max_epochs=1",
        "--loss_modes=common_loc_2d",
        "--max_videos=4",
        "--renderers",
        "source_videos",
        "input_points",
        "projection_points",
        "--source_videos_dir=/datasets/JAAD/videos",
        "--limit_train_batches=1",
        "--limit_val_batches=1"
    ], test_logs_dir)

    video_dir = os.path.join(
        test_logs_dir, "Baseline3DPose", "version_0", "videos")
    # assert video files were created
    assert os.path.exists(video_dir), 'Videos dir was not created'


def test_carla_baseline(test_logs_dir):
    """
    Test the overall flow using Carla2D3DDataModule, ProjectionTypes.absolute_loc model and points renderer.
    """
    main([
        "--data_module_name=Carla2D3D",
        "--model_name=Baseline3DPose",
        "--batch_size=8",
        "--num_workers=0",
        "--clip_length=32",
        "--input_nodes=CARLA_SKELETON",
        "--output_nodes=CARLA_SKELETON",
        "--max_epochs=3",
        "--check_val_every_n_epoch=3",
        "--loss_modes=loc_2d_3d",
        "--max_videos=4",
        "--renderers",
        "input_points",
        "projection_points",
        "--limit_val_batches=1",
        "--limit_train_batches=1"
    ], test_logs_dir)

    video_dir = os.path.join(
        test_logs_dir, "Baseline3DPose", "version_0", "videos")
    # assert video files were created
    assert os.path.exists(video_dir), 'Videos dir was not created'


def test_carla_lstm(test_logs_dir):
    """
    Test the overall flow using cum_pose_changes loss and no rendering.
    """
    main([
        "--data_module_name=Carla2D3D",
        "--model_name=LSTM",
        "--batch_size=2",
        "--num_workers=0",
        "--clip_length=32",
        "--input_nodes=CARLA_SKELETON",
        "--output_nodes=CARLA_SKELETON",
        "--max_epochs=1",
        "--loss_modes",
        "cum_pose_changes",
        "loc_2d_3d",
        "--renderers",
        "none",
        "--limit_val_batches=1",
        "--limit_train_batches=1"
    ], test_logs_dir)


def test_carla_linear_autoencoder(test_logs_dir):
    """
    Test the overall flow using ProjectionTypes.pose_changes model, rot_3d loss and default rendering.
    """
    main([
        "--data_module_name=Carla2D3D",
        "--model_name=LinearAE",
        "--batch_size=2",
        "--num_workers=0",
        "--clip_length=32",
        "--input_nodes=CARLA_SKELETON",
        "--output_nodes=CARLA_SKELETON",
        "--max_epochs=1",
        "--loss_modes",
        "rot_3d",
        "loc_2d_3d",
        "--limit_val_batches=1",
        "--limit_train_batches=1"
    ], test_logs_dir)

    video_dir = os.path.join(
        test_logs_dir, "LinearAE", "version_0", "videos")
    # assert video files were created
    assert os.path.exists(video_dir), 'Videos dir was not created'
