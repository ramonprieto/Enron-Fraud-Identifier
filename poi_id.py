#!/usr/bin/python

import sys
import pickle
from matplotlib import pyplot as plt
import numpy as np
sys.path.append("/Users/ramonprieto/Google Drive/Data Analysis/machine-learning/mini-projects/ud120-projects/tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split

### Select features.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] 
excluded_features = ["loan_advances", "email_address", "poi"]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
#Remove the total from the data 
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

### Create new feature(s)
for name in data_dict:
	from_this_person_to_poi = data_dict[name]["from_this_person_to_poi"]
	from_messages = data_dict[name]["from_messages"]
	from_poi_to_this_person = data_dict[name]["from_poi_to_this_person"]
	to_messages = data_dict[name]["to_messages"]
	
	if from_messages=="NaN" or to_messages=="NaN":
		data_dict[name]["from_poi_ratio"] = 0
		data_dict[name]["to_poi_ratio"] = 0
	else:
		data_dict[name]["from_poi_ratio"] = float(from_this_person_to_poi)/(from_messages+from_this_person_to_poi)
		data_dict[name]["to_poi_ratio"] = float(from_poi_to_this_person)/(to_messages+from_poi_to_this_person)

#Remove poi data to prevent data leakage
if True:
	poi_data = ["from_poi_ratio", "to_poi_ratio", "from_poi_to_this_person", 
				"from_this_person_to_poi", "shared_receipt_with_poi"]
	for feature in poi_data:
		excluded_features.append(feature)

#Get all features into feature_list
for name in data_dict:
	for feature in data_dict[name]:
		if feature in excluded_features:
			continue
		else:
			features_list.append(feature)
	break

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)


labels, features = targetFeatureSplit(data)

#Split train and test data
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42, stratify = labels)

#Select and tune algorithm
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB

cv = StratifiedShuffleSplit(n_splits = 1000, test_size = .3, random_state = 42)

scaler = MinMaxScaler()
selector = SelectKBest()
pca = PCA()
gnb = GaussianNB()
pipeline = Pipeline([("selector", selector), ("pca", pca), ("gnb", gnb)])

#parameters to be changed in grid search
params_grid =  {
				"pca__n_components": [6], 
				"selector__k": [9], 
				}

gs = GridSearchCV(pipeline, params_grid, cv = cv, scoring = "f1")
gs.fit(features, labels)

#Select optimal classifier
clf = gs.best_estimator_
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

#Get features importance scores plot and features used in the model
if False:
	feature_importances = clf.named_steps["selector"].scores_
	features_selected = clf.named_steps["selector"].get_support()
	i = 1
	for feature in features_selected:
		if feature == True:
			print features_list[i]
		i+=1

	feature_importances = sorted(feature_importances)
	y_pos = np.arange(len(features_list[1:]))
	plt.bar(y_pos, feature_importances, align = "center")
	plt.xticks(y_pos, features_list[1:], rotation = 90)
	plt.ylabel("Importance Scores")

	plt.show()
### Dump the classifier, dataset, and features_list so anyone can
### check your results.
dump_classifier_and_data(clf, my_dataset, features_list)

#Test results using test classifier
from tester import test_classifier
test_classifier(clf, my_dataset, features_list)
