import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV

filename = "splice.data"

basesToValues = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'D': [0.3, 0, 0.3, 0.3], 'N': [0.25, 0.25, 0.25, 0.25], 'S': [0, 0.5, 0.5, 0], 'R': [0.5, 0, 0.5, 0]}

rawData = np.loadtxt(filename, dtype=np.str, converters = {0: lambda s: s[0:-1]}, usecols=(0, 2))

data = np.zeros((len(rawData), 60*4))
labels = np.zeros((len(rawData))).astype('S')
for r in range(0, len(rawData)):
    labels[r] = rawData[r][0]
    bases = list(rawData[r][1])
    baseData = np.array(basesToValues[bases[0]])
    for i in range(1, 60):
        baseData = np.append(baseData, basesToValues[bases[i]])
    data[r] = baseData

classifier = svm.SVC(C=1.5, gamma=0.02, random_state=42)
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size = 0.5, shuffle = True, random_state=42)
classifier.fit(trainData, trainLabels)
predicted = classifier.predict(testData)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(testLabels, predicted)))


# hyperparameter tuning

# C_range = np.logspace(0, 1, base=10, num=5)
# gamma_range = np.logspace(-2, -1, base=10, num=5)
# param_grid = {'C': C_range, 'gamma': gamma_range}
# cv = ShuffleSplit(n_splits=3, test_size=0.2, train_size=0.2)
# cv.split(data)
# grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, verbose=1)
# grid.fit(data, labels)

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))