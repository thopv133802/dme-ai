import os
import pickle
import time

import numpy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC

from utils import FileUtils

def BASE_CLASSIFIER(): return SVC()

def load_classifier(classifier_path):
    classifier = BASE_CLASSIFIER()
    if os.path.exists(classifier_path):
        model_file = open(classifier_path, "rb")
        classifier = pickle.load(model_file)
        model_file.close()
    return classifier
def load_X(X_path):
    if os.path.exists(X_path):
        X_file = open(X_path, "rb")
        X = pickle.load(X_file)
        X_file.close()
    else:
        X = False
    return X

class FaceClassifier:
    model_folder = "model"
    classifier_path = f"{model_folder}/face_classifier.pkl"
    X_path = f"{model_folder}/X.pkl"
    classifier = load_classifier(classifier_path)
    X = load_X(X_path)
    strange_threshold = 2.0 / 3
    @classmethod
    def fit(cls, X, y):
        cls.X = X
        cls.classifier = BASE_CLASSIFIER()
        cls.classifier.fit(X, y)
        FileUtils.make_dirs(cls.model_folder)
        model_file = open(cls.classifier_path, "wb")
        pickle.dump(cls.classifier, model_file)
        model_file.close()
        X_file = open(cls.X_path, "wb")
        pickle.dump(cls.X, X_file)
        X_file.close()
    @classmethod
    def is_strangers(cls, X):
        similarities = cosine_similarity(X, cls.X)
        max_similarities = numpy.max(similarities, axis = 1)
        return max_similarities < cls.strange_threshold, max_similarities
    @classmethod
    def classifies(cls, X):
        if isinstance(cls.X, bool) and not cls.X:
            raise Exception("Vui lòng huấn luyện mô hình trước khi sử dụng.")
        is_strangers, confidents = cls.is_strangers(X)
        predictions = cls.classifier.predict(X)
        return ["Unknown" if is_strangers[index] else pred for (index, pred) in enumerate(predictions)], confidents
