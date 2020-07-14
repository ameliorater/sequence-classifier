import numpy as np
import sklearn
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

filename = "splice.data"

basesToDigits = {'A': 0, 'C': 1, 'T': 2, 'G': 3, 'D': 4, 'N': 5, 'S': 6, 'R': 7}
labelsToDigits = {'EI': 0, 'IE': 1, 'N': 2}

rawData = np.loadtxt(filename, dtype=np.str, converters = {0: lambda s: s[0:-1]}, usecols=(0, 2), unpack=False, ndmin=0, encoding='bytes', max_rows=None)

data = np.zeros((len(rawData), 60))
labels = np.zeros((len(rawData))).astype('S')
for r in range(0, len(rawData)):
    labels[r] = rawData[r][0]  # label
    for c in range (0, 60):  # bases
        data[r][c] = basesToDigits[list(rawData[r][1])[c]]

classifier = svm.SVC(gamma=0.001)
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size = 0.5, shuffle = True)
classifier.fit(trainData, trainLabels)
predicted = classifier.predict(testData)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(testLabels, predicted)))