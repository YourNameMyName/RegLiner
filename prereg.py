import pandas as pd
import os

os.chdir('F:/Data/last.FM/reg')
tag_dummy = pd.read_csv('tag_dummy.csv')
rating = pd.read_csv('user_artists.csv')
user_continuous = pd.read_csv('user_continuous_covariates.csv')
artist_continuous = pd.read_csv('artist_continuous_covariates.csv')

temp = pd.merge(rating, user_continuous, on='userID')
continuous_covariates = pd.merge(temp, artist_continuous, on='artistID')

tag_dummy = tag_dummy.fillna(0)
