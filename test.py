import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

mnist = fetch_openml("mnist_784", as_frame=False, parser = "auto")

X, y = mnist.data, mnist.target
# print(X,y)




# def plot_digit(image_data):
#     image = image_data.reshape(28, 28)
#     plt.imshow(image, cmap="binary")
#     plt.axis("off")


# some_digit = X[0]
# plot_digit(some_digit)
# plt.show()

# plt.figure(figsize=(9, 9))
# for idx, image_data in enumerate(X[:100]):
#     plt.subplot(10, 10, idx + 1)
#     plot_digit(image_data)
# plt.subplots_adjust(wspace=0, hspace=0)
# save_fig("more_digits_plot", tight_layout=False)
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == "5")
y_test_5 = (y_test == "5")


#Classifiers
kn_clf = KNeighborsClassifier()
kn_clf.fit(X_train, y_train_5)

dtc_clf = DecisionTreeClassifier(random_state=42)
dtc_clf.fit(X_train, y_train_5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
# print(sgd_clf.predict([some_digit]))

y_test_pred = cross_val_predict(kn_clf, X_test, y_test_5, cv=3) #Testset predictions
y_test_pred_sgd = cross_val_predict(sgd_clf, X_test, y_test_5, cv=3)
y_train_pred3 = cross_val_predict(kn_clf,X_train, y_train_5, cv=3) #Trainingset predictions
y_train_pred2 = cross_val_predict(dtc_clf,X_train, y_train_5, cv=3)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm3 = confusion_matrix(y_train_5, y_train_pred2) 
cm2 = confusion_matrix(y_train_5, y_train_pred2) 
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm3)
print(cm2)
print(cm)

print(precision_score(y_test_5, y_test_pred) + "kn test") #Running testset precisionscore
print(precision_score(y_test_5, y_test_pred_sgd)+ "sgd test") #Running testset precisiionscore with sgd
print(precision_score(y_train_5, y_train_pred3) + "kn training") #Running trainingset precisionscores
print(recall_score(y_train_5, y_train_pred3))
print(f1_score(y_train_5, y_train_pred3))
print(precision_score(y_train_5, y_train_pred2) + "dtc training")
print(recall_score(y_train_5, y_train_pred2))
print(f1_score(y_train_5, y_train_pred2))
print(precision_score(y_train_5, y_train_pred) + "sgd training")
print(recall_score(y_train_5, y_train_pred))
print(f1_score(y_train_5, y_train_pred))
