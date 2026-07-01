import torch

from ICRDrag.dataset.raydiff_utils import cameras_to_rays, first_camera_transform, normalize_cameras
from pytorch3d.renderer import PerspectiveCameras

def _cameras_from_opencv_projection(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
    do_normalize_cameras,
    normalize_scale,
) -> PerspectiveCameras:
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    cams = PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=image_size,
        device=R.device,
    )
    
    if do_normalize_cameras:
        cams, _ = normalize_cameras(cams, scale=normalize_scale)
    
    cams = first_camera_transform(cams, rotation_only=False)
    return cams

def calculate_rays(Ks, sizes, Rs, Ts, target_size, use_plucker=True, do_normalize_cameras=False, normalize_scale=1.0):
    cameras = _cameras_from_opencv_projection(
        R=Rs,
        tvec=Ts,
        camera_matrix=Ks,
        image_size=sizes,
        do_normalize_cameras=do_normalize_cameras,
        normalize_scale=normalize_scale
    )
        
    rays_embedding = cameras_to_rays(
        cameras=cameras,
        num_patches_x=target_size,
        num_patches_y=target_size,
        crop_parameters=None,
        use_plucker=use_plucker
    )
        
    return rays_embedding.rays
