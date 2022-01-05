# SUNG JUN LEE 2021

# Model training and testing routines.
# See report.pdf for more info and design
# as well as discussion on results.


import musical_features as mf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import plot_confusion_matrix, classification_report

from prep import Data

# Generate Count Vectorizer
# Unigram, Bigram, and Unigram-Bigram Mixed Modelling
cv = CountVectorizer(ngram_range=(1, 1), binary=True)
# cv = CountVectorizer(ngram_range=(1, 2), binary=True)
# cv = CountVectorizer(ngram_range=(2, 2), binary=True)
# cv = TfidfVectorizer(ngram_range=(1, 1), binary=True)
# cv = TfidfVectorizer(ngram_range=(1, 2), binary=True)
# cv = TfidfVectorizer(ngram_range=(2, 2), binary=True)

# Create Label vectors for Train, Test
# We only split into Train and Test since Cross Validation will be performed later using the Training set
y_train = (["rap"] * len(Data.trainRap)) + (["pop"] * len(Data.trainPop)) + (["r-b"] * len(Data.trainRnB)) + (
        ["rock"] * len(Data.trainRock))
y_test = (["rap"] * len(Data.testRap)) + (["pop"] * len(Data.testPop)) + (["r-b"] * len(Data.testRnB)) + (
            ["rock"] * len(Data.testRock))

# Merge All Genres into one vector
x_train = pd.concat([Data.trainRap, Data.trainPop, Data.trainRnB, Data.trainRock])
x_test = pd.concat([Data.testRap, Data.testPop, Data.testRnB, Data.testRock])

# Remove "." to generate BoW / BoN-Grams
x_train_no_period = x_train['lyrics'].str.replace(".", " ", regex=False)
x_test_no_period = x_test['lyrics'].str.replace(".", " ", regex=False)

# Fit CV and Transform into BoW / BoN-Grams
x_train_def = cv.fit_transform(x_train_no_period)
x_test_def = cv.transform(x_test_no_period)

# Generate Rhythm and/or Rhyme feature vectors
x_train_all_features = csr_matrix(np.array([mf.get_musical_feature_vector(lyric) for lyric in x_train['lyrics']]))
x_test_all_features = csr_matrix(np.array([mf.get_musical_feature_vector(lyric) for lyric in x_test['lyrics']]))
x_train_rhyme_features = csr_matrix(np.array([mf.get_musical_feature_vector(lyric, True, False) for lyric in x_train['lyrics']]))
x_test_rhyme_features = csr_matrix(np.array([mf.get_musical_feature_vector(lyric, True, False) for lyric in x_test['lyrics']]))
x_train_rhythm_features = csr_matrix(np.array([mf.get_musical_feature_vector(lyric, False, True) for lyric in x_train['lyrics']]))
x_test_rhythm_features = csr_matrix(np.array([mf.get_musical_feature_vector(lyric, False, True) for lyric in x_test['lyrics']]))

# Concatenate BoW vectors with Rhythm and/or Rhyme feature vectors
x_train_all = hstack([x_train_def, x_train_all_features])
x_test_all = hstack([x_test_def, x_test_all_features])
x_train_rhyme = hstack([x_train_def, x_train_rhyme_features])
x_test_rhyme = hstack([x_test_def, x_test_rhyme_features])
x_train_rhythm = hstack([x_train_def, x_train_rhythm_features])
x_test_rhythm = hstack([x_test_def, x_test_rhythm_features])

# Main Method which executes the final model
def main():
    # Generate best LogisticRegressionCV, SVM, KNN, Naive Bayes model
    lrclf = LogisticRegressionCV(multi_class='ovr', solver='newton-cg', max_iter=5000)
    svmclf = LinearSVC(multi_class='ovr', max_iter=5000)
    knnclf = KNeighborsClassifier(n_neighbors=3)
    berclf = BernoulliNB()

    print("Preprocessing Complete")
    print("Commencing Training")

    # Logistic Regression
    lrclf.fit(x_train_def, y_train)
    print("LR-BoW Test:", lrclf.score(x_test_def, y_test))
    lrclf.fit(x_train_rhyme, y_train)
    print("LR-BoW+Rhyme Test:", lrclf.score(x_test_rhyme, y_test))
    lrclf.fit(x_train_rhythm, y_train)
    print("LR-BoW+Rhythm Test:", lrclf.score(x_test_rhythm, y_test))
    lrclf.fit(x_train_all, y_train)
    print("LR-BoW+All Test:", lrclf.score(x_test_all, y_test))
    lrclf.fit(x_train_rhyme_features, y_train)
    print("LR-Rhyme Test:", lrclf.score(x_test_rhyme_features, y_test))
    lrclf.fit(x_train_rhythm_features, y_train)
    print("LR-Rhythm Test:", lrclf.score(x_test_rhythm_features, y_test))
    lrclf.fit(x_train_all_features, y_train)
    print("LR-All Test:", lrclf.score(x_test_all_features, y_test))

    # SVM
    svmclf.fit(x_train_def, y_train)
    print("svm-BoW Test:", svmclf.score(x_test_def, y_test))
    svmclf.fit(x_train_rhyme, y_train)
    print("svm-BoW+Rhyme Test:", svmclf.score(x_test_rhyme, y_test))
    svmclf.fit(x_train_rhythm, y_train)
    print("svm-BoW+Rhythm Test:", svmclf.score(x_test_rhythm, y_test))
    svmclf.fit(x_train_all, y_train)
    print("svm-BoW+All Test:", svmclf.score(x_test_all, y_test))
    svmclf.fit(x_train_rhyme_features, y_train)
    print("svm-Rhyme Test:", svmclf.score(x_test_rhyme_features, y_test))
    svmclf.fit(x_train_rhythm_features, y_train)
    print("svm-Rhythm Test:", svmclf.score(x_test_rhythm_features, y_test))
    svmclf.fit(x_train_all_features, y_train)
    print("svm-All Test:", svmclf.score(x_test_all_features, y_test))

    # KNN
    knnclf.fit(x_train_def, y_train)
    print("knn-BoW Test:", knnclf.score(x_test_def, y_test))
    knnclf.fit(x_train_rhyme, y_train)
    print("knn-BoW+Rhyme Test:", knnclf.score(x_test_rhyme, y_test))
    knnclf.fit(x_train_rhythm, y_train)
    print("knn-BoW+Rhythm Test:", knnclf.score(x_test_rhythm, y_test))
    knnclf.fit(x_train_all, y_train)
    print("knn-BoW+All Test:", knnclf.score(x_test_all, y_test))
    knnclf.fit(x_train_rhyme_features, y_train)
    print("knn-Rhyme Test:", knnclf.score(x_test_rhyme_features, y_test))
    knnclf.fit(x_train_rhythm_features, y_train)
    print("knn-Rhythm Test:", knnclf.score(x_test_rhythm_features, y_test))
    knnclf.fit(x_train_all_features, y_train)
    print("knn-All Test:", knnclf.score(x_test_all_features, y_test))

    # BER
    berclf.fit(x_train_def, y_train)
    print("ber-BoW Test:", berclf.score(x_test_def, y_test))
    berclf.fit(x_train_rhyme, y_train)
    print("ber-BoW+Rhyme Test:", berclf.score(x_test_rhyme, y_test))
    berclf.fit(x_train_rhythm, y_train)
    print("ber-BoW+Rhythm Test:", berclf.score(x_test_rhythm, y_test))
    berclf.fit(x_train_all, y_train)
    print("ber-BoW+All Test:", berclf.score(x_test_all, y_test))
    berclf.fit(x_train_rhyme_features, y_train)
    print("ber-Rhyme Test:", berclf.score(x_test_rhyme_features, y_test))
    berclf.fit(x_train_rhythm_features, y_train)
    print("ber-Rhythm Test:", berclf.score(x_test_rhythm_features, y_test))
    berclf.fit(x_train_all_features, y_train)
    print("ber-All Test:", berclf.score(x_test_all_features, y_test))

    lrclf.fit(x_train_rhythm, y_train)
    y_test_pred = lrclf.predict(x_test_rhythm)

    plot_confusion_matrix(lrclf, x_test_rhythm, y_test)
    plt.savefig("./plot.png")
    plt.show()

    print(classification_report(y_test, y_test_pred))

# Hyperparameter Tuning Method for Logistic Regression Model
# Calling this function will take a long time to compute
def hyperparameterTuningLR():
    solvers = ['newton-cg', 'saga', 'lbfgs']
    iterations = [1000, 5000, 10000]

    for solver in solvers:
        for iteration in iterations:
            lrclf = LogisticRegression(multi_class='ovr', solver=solver, max_iter=iteration)

            print("Preprocessing Complete")
            print("Commencing Training")

            # Logistic Regression
            lrclf.fit(x_train_def, y_train)
            print("LR-BoW Test:", lrclf.score(x_test_def, y_test))
            lrclf.fit(x_train_rhyme, y_train)
            print("LR-BoW+Rhyme Test:", lrclf.score(x_test_rhyme, y_test))
            lrclf.fit(x_train_rhythm, y_train)
            print("LR-BoW+Rhythm Test:", lrclf.score(x_test_rhythm, y_test))
            lrclf.fit(x_train_all, y_train)
            print("LR-BoW+All Test:", lrclf.score(x_test_all, y_test))
            lrclf.fit(x_train_rhyme_features, y_train)
            print("LR-Rhyme Test:", lrclf.score(x_test_rhyme_features, y_test))
            lrclf.fit(x_train_rhythm_features, y_train)
            print("LR-Rhythm Test:", lrclf.score(x_test_rhythm_features, y_test))
            lrclf.fit(x_train_all_features, y_train)
            print("LR-All Test:", lrclf.score(x_test_all_features, y_test))

# Hyperparameter Tuning Method for Support Vector Machine Model
# Calling this function will take a long time to compute
def hyperparamterTuningSVM():
    iterations = [1000, 5000, 10000]

    for iteration in iterations:
        svmclf = LinearSVC(multi_class='ovr', max_iter=iteration)

        print("Preprocessing Complete")
        print("Commencing Training")

        # SVM
        svmclf.fit(x_train_def, y_train)
        print("svm-BoW Test:", svmclf.score(x_test_def, y_test))
        svmclf.fit(x_train_rhyme, y_train)
        print("svm-BoW+Rhyme Test:", svmclf.score(x_test_rhyme, y_test))
        svmclf.fit(x_train_rhythm, y_train)
        print("svm-BoW+Rhythm Test:", svmclf.score(x_test_rhythm, y_test))
        svmclf.fit(x_train_all, y_train)
        print("svm-BoW+All Test:", svmclf.score(x_test_all, y_test))
        svmclf.fit(x_train_rhyme_features, y_train)
        print("svm-Rhyme Test:", svmclf.score(x_test_rhyme_features, y_test))
        svmclf.fit(x_train_rhythm_features, y_train)
        print("svm-Rhythm Test:", svmclf.score(x_test_rhythm_features, y_test))
        svmclf.fit(x_train_all_features, y_train)
        print("svm-All Test:", svmclf.score(x_test_all_features, y_test))

# Hyperparameter Tuning Method for K-Nearest Neighbours Model
# Calling this function will take a long time to compute
def hyperparameterTuningKNN():
    neighbours = [3,5,10]

    for neighbour in neighbours:
        knnclf = KNeighborsClassifier(n_neighbors=neighbour)

        print("Preprocessing Complete")
        print("Commencing Training")

        # KNN
        knnclf.fit(x_train_def, y_train)
        print("knn-BoW Test:", knnclf.score(x_test_def, y_test))
        knnclf.fit(x_train_rhyme, y_train)
        print("knn-BoW+Rhyme Test:", knnclf.score(x_test_rhyme, y_test))
        knnclf.fit(x_train_rhythm, y_train)
        print("knn-BoW+Rhythm Test:", knnclf.score(x_test_rhythm, y_test))
        knnclf.fit(x_train_all, y_train)
        print("knn-BoW+All Test:", knnclf.score(x_test_all, y_test))
        knnclf.fit(x_train_rhyme_features, y_train)
        print("knn-Rhyme Test:", knnclf.score(x_test_rhyme_features, y_test))
        knnclf.fit(x_train_rhythm_features, y_train)
        print("knn-Rhythm Test:", knnclf.score(x_test_rhythm_features, y_test))
        knnclf.fit(x_train_all_features, y_train)
        print("knn-All Test:", knnclf.score(x_test_all_features, y_test))

# Execute the final model
main()
