#!/usr/bin/python
import sys
sys.path.append("tools/")

import pickle
import pprint
import matplotlib.pyplot
from feature_format import featureFormat



def plotData(data_dict, features):
	data = featureFormat(data_dict, features)
	poi_colors = ["b", "r"]
	for point in data:
	    matplotlib.pyplot.scatter( point[1], point[2], c=poi_colors[int(point[0])])

	matplotlib.pyplot.xlabel(features[1])
	matplotlib.pyplot.ylabel(features[2])
	matplotlib.pyplot.show()

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("final_project/final_project_dataset.pkl", "r") )
features = ["poi", "salary", "bonus"]
plotData(data_dict, features)

data_dict.pop('TOTAL', 0 )
plotData(data_dict, features)






