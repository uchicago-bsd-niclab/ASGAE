# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds


def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: Union[str, None]
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction is not None and point_reduction not in ["mean", "sum", "max"]:
        raise ValueError(
            'point_reduction must be one of ["mean", "sum", "max"] or None'
        )
    if point_reduction is None and batch_reduction is not None:
        raise ValueError("Batch reduction must be None if point_reduction is None")


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    texture: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and texture) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        raise NotImplementedError("This function is not implemented for Pointclouds")
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if texture is not None and texture.ndim != 3:
            raise ValueError("Expected texture to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, texture


def _chamfer_distance_single_direction(
    x,
    y,
    x_lengths,
    y_lengths,
    x_texture,
    y_texture,
    weights,
    point_reduction: Union[str, None],
    norm: int
):
    return_texture = x_texture is not None and y_texture is not None

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_texture_x = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)

    if return_texture:
        # Gather the texture using the indices and keep only value for k=0
        x_texture_near = knn_gather(y_texture, x_nn.idx, y_lengths)[..., 0, :]

        # Compute mse for the texture coordinate of the closest point
        # cham_texture_x = F.mse_loss(x_texture_near, x_texture, reduction='none')
        cham_texture_x = (x_texture_near - x_texture).square().sum(-1)
        if is_x_heterogeneous:
            cham_texture_x[x_mask] = 0.0

        if weights is not None:
            cham_texture_x *= weights.view(N, 1)

    if point_reduction == "max":
        assert not return_texture
        cham_x = cham_x.max(1).values  # (N,)
    elif point_reduction is not None:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)
        if return_texture:
            cham_texture_x = cham_texture_x.sum(1)  # (N,)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_texture:
                cham_texture_x /= x_lengths_clamped

    cham_dist = cham_x
    cham_texture = cham_texture_x if return_texture else None
    return cham_dist, cham_texture


def _apply_batch_reduction(
    cham_x, cham_texture_x, weights, batch_reduction: Union[str, None]
):
    if batch_reduction is None:
        return (cham_x, cham_texture_x)
    # batch_reduction == "sum"
    N = cham_x.shape[0]
    cham_x = cham_x.sum()
    if cham_texture_x is not None:
        cham_texture_x = cham_texture_x.sum()
    if batch_reduction == "mean":
        if weights is None:
            div = max(N, 1)
        elif weights.sum() == 0.0:
            div = 1
        else:
            div = weights.sum()
        cham_x /= div
        if cham_texture_x is not None:
            cham_texture_x /= div
    return (cham_x, cham_texture_x)


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_texture=None,
    y_texture=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_texture: Optional FloatTensor of shape (N, P1, D).
        y_texture: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum", "max"] or None. Using "max" leads to the
            Hausdorff distance.
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.
        single_directional: If False (default), loss comes from both the distance between
            each point in x and its nearest neighbor in y and each point in y and its nearest
            neighbor in x. If True, loss is the distance between each point in x and its
            nearest neighbor in y.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
        - **loss_texture**: Tensor giving the reduced mean sqaure error of texture
          between pointclouds in x and pointclouds in y. Returns None if
          x_texture and y_texture are None. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    if point_reduction == "max" and (x_texture is not None or y_texture is not None):
        raise ValueError('Texture must be None if point_reduction is "max"')

    x, x_lengths, x_texture = _handle_pointcloud_input(x, x_lengths, x_texture)
    y, y_lengths, y_texture = _handle_pointcloud_input(y, y_lengths, y_texture)

    cham_x, cham_texture_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        x_texture,
        y_texture,
        weights,
        point_reduction,
        norm
    )
    if single_directional:
        loss = cham_x
        loss_texture = cham_texture_x
    else:
        cham_y, cham_texture_y = _chamfer_distance_single_direction(
            y,
            x,
            y_lengths,
            x_lengths,
            y_texture,
            x_texture,
            weights,
            point_reduction,
            norm
        )
        if point_reduction == "max":
            loss = torch.maximum(cham_x, cham_y)
            loss_texture = None
        elif point_reduction is not None:
            loss = cham_x + cham_y
            if cham_texture_x is not None:
                loss_texture = cham_texture_x + cham_texture_y
            else:
                loss_texture = None
        else:
            loss = (cham_x, cham_y)
            if cham_texture_x is not None:
                loss_texture = (cham_texture_x, cham_texture_y)
            else:
                loss_texture = None
    return _apply_batch_reduction(loss, loss_texture, weights, batch_reduction)
