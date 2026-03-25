'''
Programmer: Jose Lara
Purpose: Use nfl_data_py to extract data from an active NFL QB and submit the data 
    into a DB to use for a linear regression model to predict passing yards/attempts
'''
#! Import nfl_data_py module for our data
import nfl_data_py as nfl 
import pandas as pd

# ?we only want the data from 2020-2025, if we wanted more years we could change this to a loop
years = [2020,2021,2022,2023,2024,2025]

# import seasonal pro football reference dataframe
szn_df = nfl.import_ngs_data('passing', years)

# select the columns we want to keep for our model
clean_up = [
    'season', 
    'week',
    'player_display_name',
    'team_abbr',
    'avg_time_to_throw',
    'pass_yards',   
    'pass_touchdowns',
    'completions',
    'attempts',
    'interceptions',
    'aggressiveness',
    'completion_percentage',
    'expected_completion_percentage'
]

#update the season dataframe
szn_df = szn_df[clean_up]

#import the schedule data, we need this for our home/away and stadium information
schedule_data = nfl.import_schedules(years) 

#clean up the data, some col worth keeping is roof and wind, can impact passing a lot
s_clean_up = [
    'season',
    'week',
    'home_team',
    'away_team',
    'roof',
    'away_rest',
    'home_rest',
    'wind',
    'temp'
]

#put the new columns inside df
schedule_data = schedule_data[s_clean_up]

'''
I only want Josh Allen's stats, but, incase down the line 
    I want to get the data for other players it is going 
        to be readily available
'''
player_name = input('Player name: ') 

#get only the data from the player we choose
player_df = szn_df[szn_df['player_display_name'].str.contains(player_name, na=False, case=False)]

# merge where team is home
home_merge = player_df.merge(
    schedule_data,
    left_on=['season', 'week', 'team_abbr'],
    right_on=['season', 'week', 'home_team'],
    how='inner'
)

# merge where team is away
away_merge = player_df.merge(
    schedule_data,
    left_on=['season', 'week', 'team_abbr'],
    right_on=['season', 'week', 'away_team'],
    how='inner'
)

# combine both df
player_df = pd.concat([home_merge, away_merge]).drop_duplicates()

# sort the df by the season/week
player_df = player_df.sort_values(by=['season', 'week'])

#save df to csv 

player_df.to_csv('josh_allen_data.csv', index=False)