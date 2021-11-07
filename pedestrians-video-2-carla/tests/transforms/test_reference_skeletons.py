from pedestrians_video_2_carla.transforms.reference_skeletons import ReferenceSkeletonsDenormalize
import torch


def test_reference_skeletons_denormalization_identity(device):
    denormalizer = ReferenceSkeletonsDenormalize(autonormalize=True)

    abs_reference = denormalizer.get_abs(device)

    abs_tensor = torch.stack([torch.tensor(ref)
                             for ref in abs_reference.values()], dim=0)
    meta = {
        'age': [],
        'gender': []
    }
    for (age, gender) in abs_reference.keys():
        meta['age'].append(age)
        meta['gender'].append(gender)

    denormalized = denormalizer.from_abs(abs_tensor, meta)

    assert torch.allclose(abs_tensor, denormalized), "Abs poses are not equal"

    abs_tensor_scaled = abs_tensor * torch.rand((1))
    denormalized_scaled = denormalizer.from_abs(abs_tensor_scaled, meta)

    assert torch.allclose(
        abs_tensor,
        denormalized_scaled,
        rtol=1e-4,
        atol=1e-4
    ), "Abs poses are not equal when input is scaled"
