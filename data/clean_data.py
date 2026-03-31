import pandas as pd 

#import our data 
df = pd.read_csv('data.csv')

#for our home/away we need to convert to binary for our model 
df['home_advantage'] = (df['team_abbr'] == df['home_team']).astype(int)

# clean up the data, we dont need all of the columns for our model, we can drop some of them
to_drop = [
    'player_display_name',
    'team_abbr',
    'home_team',
    'away_team'
]

#we know this data is just for Josh Allen so we can drop the player name
df = df.drop(columns=to_drop)

#get an idea of how many null values there are
print(df.isnull().sum())

#since players are inside of the dome there isnt any data on wind or temp, so we can set to 0 
df['wind'] = df['wind'].fillna(0)
df['temp'] = df['temp'].fillna(75)

#add a binary for roof, if it is a dome we can have a 1
df['roof_binary'] = (df['roof'] == 'dome').astype(int)
df = df.drop(columns='roof')
df = df.rename(columns={'roof_binary': 'dome'})

#df.to_csv('clean_data.csv', index=False) 

#print the first few rows of the cleaned data
print(df.head())

