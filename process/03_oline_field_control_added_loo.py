from os.path import join
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from matplotlib.path import Path
from tqdm import tqdm

from helpers.rb_field_control import get_rb_field_control

DATA_DIR = '../data/01_processed'
RESULTS_DIR = '../results'

import sys
if len(sys.argv) > 1:
    VISION_CONE_ANGLE = int(sys.argv[1])
else:
    VISION_CONE_ANGLE = 45

def add_rb_coords(tracking):
    """Add ball carrier coordinates to tracking data"""
    rb_coords = (
        tracking[tracking['is_ball_carrier'] == 1]
        .set_index(['game_play_id', 'frame_id'])[['x', 'y']]
        .rename(columns={'x': 'ball_carrier_x', 'y': 'ball_carrier_y'})
    )
    tracking = tracking.merge(
        rb_coords,
        left_on=['game_play_id', 'frame_id'],
        right_index=True,
        how='left'
    )
    tracking['euclidean_dist_to_ball_carrier'] = (
        ((tracking['x'] - tracking['ball_carrier_x']) ** 2 +
         (tracking['y'] - tracking['ball_carrier_y']) ** 2) ** 0.5
    ).round(2)
    tracking.drop(columns=['ball_carrier_x', 'ball_carrier_y'], inplace=True)

    return tracking

def _process_single_game_play(args):
    """Worker function to compute O-line attributions for a single game_play_id."""
    gpid, df_all = args

    # Identify O-line
    oline_positions = ['LG', 'LT', 'C', 'RG', 'RT']
    oline = df_all[df_all['position_by_loc'].isin(oline_positions)].copy()

    oline_ids = list(oline['nfl_id'].unique())
    positions_map = dict(
        zip(oline_ids, oline.groupby('nfl_id')['position_by_loc'].first())
    )

    if oline.position_by_loc.nunique() < 5:
        return {
            'game_play_id': gpid,
            'baseline_fc': None,
            'attributions': {pid: None for pid in oline_ids},
            'oline_positions': positions_map
        }

    # Everyone else (RB, WR, defense, etc.)
    non_oline_df = df_all[~df_all['nfl_id'].isin(oline_ids)]

    def fc_for_subset(subset):
        """Return RB field control for a subset of O-line player IDs."""
        coalition_df = pd.concat(
            [non_oline_df, oline[oline['nfl_id'].isin(subset)]],
            ignore_index=True
        )
        return get_rb_field_control(
            coalition_df, 
            every_n_frames=3,
            vision_cone_angle=VISION_CONE_ANGLE
        )

    # Baseline with all O-line
    baseline_fc = fc_for_subset(oline_ids)

    # Calculate LOO attributions
    loo_attributions = {}
    for pid in oline_ids:
        subset = [p for p in oline_ids if p != pid]
        loo_fc = fc_for_subset(subset)
        loo_attributions[pid] = baseline_fc - loo_fc  # How much FC drops when player is removed

    return {
        'game_play_id': gpid,
        'baseline_fc': baseline_fc,
        'attributions': loo_attributions,
        'oline_positions': positions_map
    }

def calculate_oline_attributions(tracking):
    """Calculate O-line attribution in parallel across game_play_id."""

    game_play_ids = list(tracking['game_play_id'].unique())
    grouped = {
        gpid: g.copy() for gpid, g in 
        tracking[tracking['game_play_id'].isin(game_play_ids)].groupby('game_play_id')
    }

    results = []
    for gpid, group in tqdm(grouped.items(), desc="Processing Game Plays"):
        results.append(_process_single_game_play((gpid, group)))

    return results

def save_results(attribution_results):
    """Save the results of O-line attribution to a CSV file"""
    attrib_df = pd.DataFrame([
        {
            'game_play_id': res['game_play_id'],
            'player_id': pid,
            'position': res['oline_positions'][pid],
            'fc_attribution': val,
            'baseline_fc': res['baseline_fc']
        }
        for res in attribution_results
        for pid, val in res['attributions'].items()
    ])

    attrib_df.to_csv(
        join(RESULTS_DIR, f'oline_loo_attribution_{VISION_CONE_ANGLE}.csv'), 
        index=False
    )

def main():
    tracking = (
        pd.read_parquet(join(DATA_DIR, 'tracking.parquet'))
        .query('club != "football"')
    )

    cols = ['x', 'y', 's', 'dir', 'offense', 'is_ball_carrier', 'position_by_loc']
    tracking = tracking[['game_play_id', 'frame_id', 'nfl_id'] + cols].copy()

    # Add euclidean_dist_to_ball_carrier column
    tracking = add_rb_coords(tracking)

    # Calculate offensive line players contribution to rb field control using LOO
    attribution_results = calculate_oline_attributions(tracking)

    # Save the results to a CSV file
    save_results(attribution_results)

if __name__ == "__main__":
    print(f"Using vision cone angle: {VISION_CONE_ANGLE} degrees")
    main()