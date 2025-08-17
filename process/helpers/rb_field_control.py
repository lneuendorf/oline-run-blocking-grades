import sys
import numpy as np
import numpy.typing as npt
import pandas as pd
from .influence import influence

def get_rb_field_control(
    tracking,
    every_n_frames: int = 1,
    vision_cone_angle: int = 45,
) -> npt.NDArray[np.float32]:
    """Calculate RB field control for a given play.
    
    Args:
        tracking: DataFrame with tracking data for the play.
    Returns:
        Average field control value in the RB's vision cone across all frames.
    """

    fc_values = []
    frame_ids = tracking['frame_id'].unique()

    # If every_n_frames > 1, filter frames to reduce computation
    if every_n_frames > 1:
        frame_ids = sorted(tracking['frame_id'].unique())
        frame_ids = {fid for i, fid in enumerate(frame_ids) if i % every_n_frames == 0}
        tracking = tracking[tracking['frame_id'].isin(frame_ids)]

    for frame_id in frame_ids:

        # Ball carrier coordinates, speed, and direction
        rb_data = tracking.query(
            'is_ball_carrier == 1 and frame_id == @frame_id'
        )[['x', 'y', 'dir', 's']]
        if rb_data.empty:
            game_play_id = tracking['game_play_id'].unique()[0]
            print(f"No ball carrier data found for game_play_id {game_play_id} "
                  f"at frame {frame_id}.")
            return 0.0
        rb_x = rb_data['x'].values[0]
        rb_y = rb_data['y'].values[0]
        rb_dir = rb_data['dir'].values[0]

        # RB field control coordinates are a combination of vision cone and 
        # semicircle surrounding the RB
        rb_coords = np.vstack([
            _get_vision_cone(
                rb_pos = np.array([rb_x, rb_y]),
                rb_dir_rad = np.radians(rb_dir),
                cone_angle_deg=vision_cone_angle,
                cone_length=3,
                spacing=0.25
            ),
            _get_rb_semicircle(
                rb_pos = np.array([rb_x, rb_y]),
                rb_dir_rad = np.radians(rb_dir),
                radius=1,
                cone_angle_deg=90,
                spacing=0.05
            )
        ])
        
        fc_values.append(_compute_field_control(
            tracking.query('frame_id == @frame_id and is_ball_carrier == 0'),
            rb_coords
        ))

    return np.array(fc_values, dtype=np.float32)

def _compute_field_control(
    frame_df: pd.DataFrame,
    rb_coords: np.ndarray,
) -> float:
    smoothing_param = sys.float_info.epsilon
    field_control = np.zeros(rb_coords.shape[:1])

    cols = ['x', 'y', 's', 'dir', 'offense', 'euclidean_dist_to_ball_carrier']
    for i, coord in enumerate(rb_coords):
        for row in frame_df[cols].itertuples(index=False):                    
            field_control[i] += influence(
                p=coord,
                p_i=np.array([row.x, row.y]),
                s=row.s + smoothing_param,
                theta=row.dir * np.pi / 180,
                rb_dist=row.euclidean_dist_to_ball_carrier,
                is_offense=row.offense
            )
    return np.mean(1 / (1 + np.exp(-field_control)))  # sigmoid normalization

def _get_vision_cone(
    rb_pos,
    rb_dir_rad,
    cone_angle_deg=45,
    cone_length=3,
    spacing=0.5
):
    """Generate evenly spaced points inside RB vision cone."""
    cone_half_angle = np.radians(cone_angle_deg / 2)

    # Create a grid that covers the cone bounding box
    max_radius = cone_length
    grid_x, grid_y = np.meshgrid(
        np.arange(-max_radius, max_radius + spacing, spacing),
        np.arange(-max_radius, max_radius + spacing, spacing)
    )
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Keep only points in front of RB within cone length and angle
    dist = np.linalg.norm(points, axis=1)
    angles = np.arctan2(points[:, 1], points[:, 0])  # relative to facing +x
    mask = (
        (dist <= cone_length) &
        (np.abs(angles) <= cone_half_angle)
    )
    cone_points = points[mask]

    # Rotate and translate to RB position
    rot_matrix = np.array([
        [np.cos(rb_dir_rad), -np.sin(rb_dir_rad)],
        [np.sin(rb_dir_rad), np.cos(rb_dir_rad)]
    ])
    return (rot_matrix @ cone_points.T).T + rb_pos

def _get_rb_semicircle(
    rb_pos,
    rb_dir_rad,
    radius=1,
    cone_angle_deg=45,
    spacing=0.5
):
    """Generate evenly spaced points in semicircle behind RB."""
    excluded_half_angle = np.radians((cone_angle_deg) / 2)

    # Create a grid that covers full circle bounding box
    grid_x, grid_y = np.meshgrid(
        np.arange(-radius, radius + spacing, spacing),
        np.arange(-radius, radius + spacing, spacing)
    )
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    dist = np.linalg.norm(points, axis=1)
    angles = np.arctan2(points[:, 1], points[:, 0])  # relative to +x direction

    # Keep points within circle but outside cone wedge
    mask = (
        (dist <= radius) &
        ((angles > excluded_half_angle) | (angles < -excluded_half_angle))
    )
    semi_points = points[mask]

    # Rotate and translate to RB position
    rot_matrix = np.array([
        [np.cos(rb_dir_rad), -np.sin(rb_dir_rad)],
        [np.sin(rb_dir_rad), np.cos(rb_dir_rad)]
    ])
    return (rot_matrix @ semi_points.T).T + rb_pos