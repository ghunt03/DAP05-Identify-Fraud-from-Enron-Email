

import pickle
import pprint
enron_data = pickle.load(open("final_project/final_project_dataset.pkl", "r"))

def getFeature(person_name, feature, includeDescription=True):
	if includeDescription:
		return "{0} - {1}: {2} ".format(person_name, 
			feature, 
			enron_data[person_name][feature]) 
	else:
		return enron_data[person_name][feature]

def countFeature(data, feature, includeDescription=True, query=""):
	featureCount = 0
	for person_data in data.values():
		if (not query == ""): 
			if person_data[feature] == query: 
				featureCount += 1
		elif not person_data[feature] == 'NaN':
			featureCount += 1
	if includeDescription:
		return feature + ": " +str(featureCount)
	else:
		return featureCount

def calcPercentageOfFeature(data, feature, query=""):
	people = len(data)
	feature_count = countFeature(data, feature, False, query)
	percentage = (float(feature_count) / people) * 100
	return percentage


def getPeopleOfInterest(data):
	poi_data = {}
	people_names = []
	for key, person_data in data.iteritems():
		if person_data["poi"] == 1:
			people_names.append(key)
			poi_data[key] = person_data
	return people_names, poi_data


def getPeopleNames(data):
	people_names = []
	for key, person_data in data.iteritems():
		people_names.append(key)	
	return people_names

def getFeatureValues(data, feature):
	itemList = []
	for person_name, person_data in data.iteritems():
		value = enron_data[person_name][feature]
		if not value == 'NaN':
			itemList.append(value)
	return itemList

def getFeatureList(data):
	featureList = []
	for key, value in data["SKILLING JEFFREY K"].iteritems():
		featureList.append(key)
	return featureList 



total_people = len(enron_data)
poi_names, poi_data = getPeopleOfInterest(enron_data)

print "Number of people " + str(len(enron_data))
print "All names:"
pprint.pprint(getPeopleNames(enron_data))
print "Number of features " + str(len(enron_data["SKILLING JEFFREY K"]))
print "Number of Persons of interest (POI): " + str(len(poi_names))
print "People of interest: {}".format(poi_names)

percentage = calcPercentageOfFeature(enron_data, "total_payments", 'NaN')
print "Percentage of people with no payment data: " + str(percentage)

percentage = calcPercentageOfFeature(poi_data, "total_payments", 'NaN')
print "Percentage of POI with no payment data: " + str(percentage)


itemList = getFeatureValues(enron_data, "total_payments")
print max(itemList)

pprint.pprint(getFeatureList(enron_data))
enron_data.pop('TOTAL', 0)
print(min(getFeatureValues(enron_data, "salary")))
print(max(getFeatureValues(enron_data, "salary")))
pprint.pprint(enron_data["THE TRAVEL AGENCY IN THE PARK"])