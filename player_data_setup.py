import pandas as pd
from sklearn.model_selection import train_test_split


def read_misc_stats(filepath):
    team_stats_df = pd.read_csv(filepath)

    print(team_stats_df.columns)
    team_stats_df = team_stats_df.drop(columns=['W','L','PW','PL','MOV','SOS','SRS','ORtg','FTr','3PAr','eFG%','TOV%','ORB%','SOS','SRS','ORtg','FTr', '3PAr', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA'])
    team_stats_df.rename(columns={'eFG%.1': 'eFG%', 'TOV%.1': 'TOV%', 'FT/FGA.1':'FT/FGA'}, inplace=True)
    team_stats_df.to_csv(filepath, index=False)
    return

def player_data_merge(player_file):
    df = pd.read_csv(player_file)
    df2 = pd.read_csv('./data/team_def_stats.csv')
    df['Home'] = df['Unnamed: 5'].apply(lambda x: 1 if pd.isna(x) or x == '' else 0)
    df = df[df['MP'].notna() & (df['MP'] != '') & (df["MP"] != 'Inactive') & (df["MP"] != 'Did Not Play') & (df["MP"] != 'Did Not Dress')]
    merged_df = pd.merge(df, df2, left_on='Opp', right_on='Team', how='left')
    col_to_keep = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
       'eFG%_x', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK',
       'TOV', 'PF', 'PTS', 'Home',
       'DRtg', 'Pace', 'eFG%_y', 'TOV%', 'DRB%', 'FT/FGA']
    merged_df["MP"] = merged_df["MP"].apply(convert_mp)
    merged_df['PTS'] = pd.to_numeric(merged_df['PTS'], errors='coerce')
    merged_df['MP'] = pd.to_numeric(merged_df['MP'], errors='coerce')
    merged_df['FGA'] = pd.to_numeric(merged_df['PTS'], errors='coerce')
    merged_df['PTS'] = merged_df["PTS"].astype(float)
    merged_df['MP'] = merged_df["MP"].astype(float)
    merged_df['FGA'] = merged_df["FGA"].astype(float)
    return merged_df[col_to_keep]

def convert_mp(mp):
    minutes, seconds = mp.split(':')
    return int(minutes) + int(seconds) / 60


def get_rolling_avgs(player_df):

    player_df['PTS_last_5_avg'] = player_df['PTS'].rolling(window=10, min_periods=1).mean()
    player_df['FGA_last_5_avg'] = player_df['FGA'].rolling(window=10, min_periods=1).mean()
    player_df['MP_last_5_avg'] = player_df['MP'].rolling(window=10, min_periods=1).mean()
    player_df['Opp_DRtg_x_PTS'] = player_df['DRtg'] * player_df['PTS_last_5_avg']
    player_df['Opp_Pace_x_FGA'] = player_df['Pace'] * player_df['FGA_last_5_avg']
    player_df['Opp_eFG_x_PTS'] = player_df['eFG%_y'] * player_df['PTS_last_5_avg']
    player_df['PTS_5_game_trend'] = player_df['PTS'].rolling(10).apply(lambda x: x.iloc[-1] - x.iloc[0])
    player_df['PTS_volatility_5'] = player_df['PTS'].rolling(window=10).std().shift(1)
    player_df['Hot_Streak'] = (player_df['PTS'] > player_df['PTS'].expanding().mean().shift(1)).astype(int).rolling(window=3).sum().shift(1)
    # player_df['MP_x_FGA'] = player_df['MP'] * player_df['FGA']
    player_df['PTS_per_minute'] = player_df['PTS'] / player_df['MP']
    player_df['PTS_pct_of_max'] = player_df['PTS'] / player_df['PTS'].expanding().max().shift(1)

    return player_df

def get_splits(player_df):
    to_drop = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA',
       'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'eFG%_x', "2P", "2PA", "2P%"
       ]
    to_test = ['Home', 'DRtg', 'Pace', 'eFG%_y', 'TOV%', 'DRB%', 'FT/FGA',
       'PTS_last_5_avg', 'FGA_last_5_avg', 'MP_last_5_avg', 'Opp_DRtg_x_PTS',
       'Opp_Pace_x_FGA', 'Opp_eFG_x_PTS', 'PTS_5_game_trend', 'PTS_volatility_5', 'Hot_Streak',  "PTS_per_minute", "PTS_pct_of_max"]
    
    exiled = ["MP_x_FGA",] # non-useful (for now) terms

    y = player_df['PTS']  # Target variable

    X = player_df[to_test]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X, X_test, X_train, y_train, y_test

def prep_game(opponent, player, teams):
    game_index = len(player) - 1

    # Get team stats for that opponent
    opp_stats = teams[teams['Team'] == opponent].iloc[0]

    # Get last 11 games (including the current one at index `game_index`)
    last_10 = player.iloc[game_index - 10: game_index + 1].copy()

    # Coerce to numeric
    last_10['PTS'] = pd.to_numeric(last_10['PTS'], errors='coerce')
    last_10['FGA'] = pd.to_numeric(last_10['FGA'], errors='coerce')
    last_10['MP'] = pd.to_numeric(last_10['MP'], errors='coerce')

    # Clean types
    last_10['PTS'] = last_10['PTS'].astype(int)
    last_10['FGA'] = last_10['FGA'].astype(int)
    last_10['MP'] = last_10['MP'].astype(float)

    # Compute means
    pts_avg = last_10['PTS'].mean()
    fga_avg = last_10['FGA'].mean()
    mp_avg = last_10['MP'].mean()

    # Trend and volatility
    pts_trend = last_10['PTS'].iloc[-1] - last_10['PTS'].iloc[0]
    pts_volatility = last_10['PTS'].std()

    # Hot streak: is last game part of 3-game hot streak?
    hot_streak = (
        (last_10['PTS'] > last_10['PTS'].expanding().mean().shift(1))
        .astype(int)
        .rolling(window=3)
        .sum()
        .shift(1)
    ).iloc[-1]

    # MP x FGA and PTS/minute
    mp_x_fga = last_10['MP'].iloc[-1] * last_10['FGA'].iloc[-1]
    pts_per_minute = (
        last_10['PTS'].iloc[-1] / last_10['MP'].iloc[-1]
        if last_10['MP'].iloc[-1] != 0 else 0
    )

    # PTS as % of previous max
    pts_pct_of_max = (
        last_10['PTS'].iloc[-1] /
        last_10['PTS'].expanding().max().shift(1).iloc[-1]
        if last_10['PTS'].expanding().max().shift(1).iloc[-1] != 0 else 0
    )

    # Construct input row
    input_row = pd.DataFrame({
        'Home': [0],
        'DRtg': [opp_stats['DRtg']],
        'Pace': [opp_stats['Pace']],
        'eFG%_y': [opp_stats['eFG%']],
        'TOV%': [opp_stats['TOV%']],
        'DRB%': [opp_stats['DRB%']],
        'FT/FGA': [opp_stats['FT/FGA']],
        'PTS_last_5_avg': [pts_avg],
        'FGA_last_5_avg': [fga_avg],
        'MP_last_5_avg': [mp_avg],
        'Opp_DRtg_x_PTS': [opp_stats['DRtg'] * pts_avg],
        'Opp_Pace_x_FGA': [opp_stats['Pace'] * fga_avg],
        'Opp_eFG_x_PTS': [opp_stats['eFG%'] * pts_avg],
        'PTS_5_game_trend': [float(pts_trend)],
        'PTS_volatility_5': [float(pts_volatility)],
        'Hot_Streak': [hot_streak],
        # 'MP_x_FGA': [mp_x_fga],
        'PTS_per_minute': [pts_per_minute],
        'PTS_pct_of_max': [pts_pct_of_max]
    })

    return input_row

def predict_game(input_row, scaler, columns, model, X):
     # Ensure correct feature order
    input_row_scaled = input_row.copy()
    input_row_scaled = input_row_scaled[X.columns]
    input_row_scaled[columns] = scaler.transform(input_row[columns])
    

    predicted_pts = model.predict(input_row)[0]
    return predicted_pts
   

    