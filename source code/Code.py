from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

iris = load_iris()

# print(iris.data)

# print(iris.feature_names)

# print(iris.target)

# print(iris.target_names)

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=4)
# print(X_train.shape)
# print(X_test.shape)
#
# print(Y_train.shape)
# print(Y_test.shape)

k_range = range(1,26)
score = {}
score_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    Y_pred = knn.predict(X_test)
    score[k] = metrics.accuracy_score(Y_test, Y_pred)
    score_list.append(metrics.accuracy_score(Y_test, Y_pred))

plt.plot(k_range,score_list)
plt.xlabel("Value of K for KNN")
plt.ylabel("Testing Accuracy")
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, Y_pred)
print(accuracy)
