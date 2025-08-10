import re
from typing import Tuple

import pandas as pd
import numpy as np

def standardize_direction(
        df_tracking: pd.DataFrame,
        df_play: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize the direction of the play and the players to be vertical.

    The direction of the play is set to be bottom to top, with the offensive
    moving from the bottom to the top.

    Args:
        df_tracking: The tracking data.
        df_play: The play data.

    Returns:
        The tracking data and the play data with the direction standardized.
    """
    
    left_play = df_tracking['play_direction'] == 'left'
    
    original_x = df_tracking['x'].copy()
    original_y = df_tracking['y'].copy()
    original_dir = df_tracking['dir'].copy()
    original_o = df_tracking['o'].copy()
    
    df_tracking['y'] = np.where(
        left_play, 
        120 - original_x,
        original_x
    )
    df_tracking['x'] = np.where(
        left_play, 
        original_y,
        53.3 - original_y
    )
    df_tracking['dir'] = np.where(
        left_play,
        (((180 - original_dir) % 360) + 180) % 360,
        (180 - original_dir) % 360,
    )
    df_tracking['o'] = np.where(
        left_play,
        (((180 - original_o) % 360) + 180) % 360,
        (180 - original_o) % 360,
    )

    df_play = df_play.merge(
        df_tracking[['game_id','play_id','play_direction']].drop_duplicates(['game_id','play_id']),
        on=['game_id','play_id'],
        how='left'
    )
    df_play = df_play.dropna(subset=['play_direction'])
    df_play['absolute_yardline_number'] = np.where(
        df_play.play_direction == "left", 
        120 - df_play.absolute_yardline_number, 
        df_play.absolute_yardline_number
    )
    
    df_play = df_play.drop('play_direction', axis=1)

    return df_tracking, df_play

def uncamelcase_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', word).lower() for word in df.columns]
    return df