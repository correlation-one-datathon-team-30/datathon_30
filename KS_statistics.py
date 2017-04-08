import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model, neural_network, model_selection, pipeline
from scipy import io
import xgboost
import pandas as pd
import pickle as cp
from scipy import stats

def main():
	#full = '/home/user/citadel_data/full_0.01.csv'
	#fulldf = pd.read_csv(full, index_col=0)
	boundary = '/home/user/citadel_data/boundary_cases.csv'
	boundarydf = pd.read_csv(boundary, index_col=0)
	features = ['Service', 'isuber', 'pickup_datetime', 'pickup_latitude',
	   'pickup_longitude', 'date', 'max_temp', 'min_temp', 'avg_temp',
	   'precipitation', 'snowfall', 'snow_depth', 'location', 'latitude',
	   'longitude', 'month', 'year', 'hour', 'DayOfWeek']
	useful_features = ['pickup_latitude',
	   'pickup_longitude', 'max_temp', 'min_temp', 'avg_temp',
	   'month', 'year', 'hour', 'DayOfWeek']
	u_df = boundarydf[boundarydf['isuber']==1]
	nu_df = boundarydf[boundarydf['isuber']!=1]
	p_vals = []
	ps = open("p_vals.txt",'w')
	for feat in useful_features:
		ps.write(str(calc_stats(u_df, nu_df, feat)) + "\n")
		


def calc_stats(u_df, nu_df, feature):
	print(feature)
	u_data = u_df.ix[:,feature].values
	u_mean = u_data.mean()
	u_std = u_data.std()
	print("Uber = " + str(u_mean) + " +- " + str(u_std) )
	nu_data = nu_df.ix[:,feature].values
	nu_mean = nu_data.mean()
	nu_std = nu_data.std()
	print("non-Uber = " + str(nu_mean) + " +- " + str(nu_std) )
	ks_stat, p_val = stats.ks_2samp(u_data, nu_data)
	if p_val > 0.1:
		print("at the 10% significance level we can accept the null hypothesis that the two samples come from the same distribution for the feature: " + feature + " which has p-value = " + str(p_val) + " from the two-sided KS test \n")
	else:
		print("at the 10% significance level we reject the null hypothesis that the two samples come from the same distribution for the feature: " + feature + " which has p-value = " + str(p_val) + " from the two-sided KS test \n")
	return p_val






if __name__ == '__main__':
	main()