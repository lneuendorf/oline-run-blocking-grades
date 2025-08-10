from os.path import join
from itertools import permutations
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from matplotlib.path import Path
from tqdm import tqdm

from helpers.rb_field_control import get_rb_field_control

DATA_DIR = '../data/01_processed'
RESULTS_DIR = '../results'

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

    # Cache for coalition evaluations (per game_play)
    coalition_cache = {}

    def fc_for_subset(subset):
        """Return RB field control for a subset of O-line player IDs."""
        subset_key = tuple(sorted(subset))
        if subset_key in coalition_cache:
            return coalition_cache[subset_key]
        coalition_df = pd.concat(
            [non_oline_df, oline[oline['nfl_id'].isin(subset)]],
            ignore_index=True
        )
        val = get_rb_field_control(coalition_df, every_n_frames=2)
        coalition_cache[subset_key] = val
        return val

    # Baseline with all O-line
    baseline_fc = fc_for_subset(oline_ids)

    # Store Shapley sums
    shapley_sums = {pid: 0.0 for pid in oline_ids}

    # All permutations of O-line players (5! = 120, but 32 unique groups with caching)
    all_perms = list(permutations(oline_ids))
    num_perms = len(all_perms)

    for perm in all_perms:
        current_subset = set()
        prev_fc = fc_for_subset(current_subset)
        for pid in perm:
            new_subset = current_subset | {pid}
            new_fc = fc_for_subset(new_subset)
            marginal_contribution = new_fc - prev_fc
            shapley_sums[pid] += marginal_contribution
            current_subset = new_subset
            prev_fc = new_fc

    shapley_values = {pid: shapley_sums[pid] / num_perms for pid in oline_ids}

    return {
        'game_play_id': gpid,
        'baseline_fc': baseline_fc,
        'attributions': shapley_values,
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
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(_process_single_game_play, (gpid, grouped[gpid])): gpid
            for gpid in game_play_ids
        }
        for fut in tqdm(
            iterable=as_completed(futures), 
            total=len(futures),
            desc="Calculating O-line attributions"
        ):
            res = fut.result()
            if res:
                results.append(res)

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

    attrib_df.to_csv(join(RESULTS_DIR, 'oline_attribution.csv'), index=False)

def main():
    tracking = (
        pd.read_parquet(join(DATA_DIR, 'tracking.parquet'))
        .query('club != "football"')
    )

    cols = ['x', 'y', 's', 'dir', 'offense', 'is_ball_carrier', 'position_by_loc']
    tracking = tracking[['game_play_id', 'frame_id', 'nfl_id'] + cols].copy()

    # Add euclidean_dist_to_ball_carrier column
    tracking = add_rb_coords(tracking)

    # Calculate offensive line players contribution to rb field control using
    # shapely influence function
    attribution_results = calculate_oline_attributions(tracking)

    # Save the results to a CSV file
    save_results(attribution_results)

if __name__ == "__main__":
    main()