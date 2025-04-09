# 90-of-gamblers...

Money Hehehe

Everything is run in the player_predictions jupyter notebook at the moment.

Run the top cell, then enter the name of the player that you want the prediction for AS IT APPEARS IN THE CSV FILE NAME (ex. "devin_booker" rather than "devin booker" or "d-book")

Then enter the player's team and the opposing team (for the correct 3 letter abbreviations please refer to the "team_def_stats.csv" file).

From there, you can run all cells, and 3 models, (RandomForestClassifier, XGBoost, and a Stacked model combining both) will train on the player's game data, combined with the defensive stats of the teams from each game.

This logic is mostly carried out in player_data_setup.py.

You will be given the projected points from each model for the specific game, as well as the training results (MAE, MSE) for each model. Optionally this is written to "predicitons.csv"


All data thus far is pulled from basketball-reference.com