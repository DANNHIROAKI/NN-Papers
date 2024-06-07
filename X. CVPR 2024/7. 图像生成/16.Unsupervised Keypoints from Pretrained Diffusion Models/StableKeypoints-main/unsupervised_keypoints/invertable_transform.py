import math
import torch
import torch.nn.functional as F


class RandomAffineWithInverse:
    def __init__(
        self,
        degrees=0,
        scale=(1.0, 1.0),
        translate=(0.0, 0.0),
    ):
        self.degrees = degrees
        self.scale = scale
        self.translate = translate

        # Initialize self.last_params to 0s
        self.last_params = {
            "theta": torch.eye(2, 3).unsqueeze(0),
        }

    def create_affine_matrix(self, angle, scale, translations_percent):
        angle_rad = math.radians(angle)

        # Create affine matrix
        theta = torch.tensor(
            [
                [math.cos(angle_rad), math.sin(angle_rad), translations_percent[0]],
                [-math.sin(angle_rad), math.cos(angle_rad), translations_percent[1]],
            ],
            dtype=torch.float,
        )

        theta[:, :2] = theta[:, :2] * scale
        theta = theta.unsqueeze(0)  # Add batch dimension
        return theta

    def __call__(self, img_tensor, theta=None):

        if theta is None:
            theta = []
            for i in range(img_tensor.shape[0]):
                # Calculate random parameters
                angle = torch.rand(1).item() * (2 * self.degrees) - self.degrees
                scale_factor = torch.rand(1).item() * (self.scale[1] - self.scale[0]) + self.scale[0]
                translations_percent = (
                    torch.rand(1).item() * (2 * self.translate[0]) - self.translate[0],
                    torch.rand(1).item() * (2 * self.translate[1]) - self.translate[1],
                    # 1.0,
                    # 1.0,
                )

                # Create the affine matrix
                theta.append(self.create_affine_matrix(
                    angle, scale_factor, translations_percent
                ))
            theta = torch.cat(theta, dim=0).to(img_tensor.device)

        # Store them for inverse transformation
        self.last_params = {
            "theta": theta,
        }

        # Apply transformation
        grid = F.affine_grid(theta, img_tensor.size(), align_corners=False).to(
            img_tensor.device
        )
        transformed_img = F.grid_sample(img_tensor, grid, align_corners=False)

        return transformed_img

    def inverse(self, img_tensor):

        # Retrieve stored parameters
        theta = self.last_params["theta"]

        # Augment the affine matrix to make it 3x3
        theta_augmented = torch.cat(
            [theta, torch.Tensor([[0, 0, 1]]).expand(theta.shape[0], -1, -1)], dim=1
        )

        # Compute the inverse of the affine matrix
        theta_inv_augmented = torch.inverse(theta_augmented)
        theta_inv = theta_inv_augmented[:, :2, :]  # Take the 2x3 part back

        # Apply inverse transformation
        grid_inv = F.affine_grid(theta_inv, img_tensor.size(), align_corners=False).to(
            img_tensor.device
        )
        untransformed_img = F.grid_sample(img_tensor, grid_inv, align_corners=False)

        return untransformed_img



def return_theta(scale, pixel_loc, rotation_angle_degrees=0):
    """
    Pixel_loc between 0 and 1
    Rotation_angle_degrees between 0 and 360
    """

    rescaled_loc = pixel_loc * 2 - 1

    rotation_angle_radians = math.radians(rotation_angle_degrees)
    cos_theta = math.cos(rotation_angle_radians)
    sin_theta = math.sin(rotation_angle_radians)

    theta = torch.tensor(
        [
            [scale * cos_theta, -scale * sin_theta, rescaled_loc[1]],
            [scale * sin_theta, scale * cos_theta, rescaled_loc[0]],
        ]
    )
    theta = theta.unsqueeze(0)
    return theta
