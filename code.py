import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import preprocessing, cross_validation, svm, neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from tabulate import tabulate

feature_column_names=['age', 'sex', 'c_pain', 'bp','serum', 'sugar','cardio','heart_rt', 'agina', 'old', 'slope', 'vessel', 'thal']

predicted_class_name=['prediction']

df=pd.read_csv("C:\Python27\heart2.csv")

x= df[feature_column_names].values
y=df[predicted_class_name].values


split_test_size=.2



x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=split_test_size,random_state=42)

mlp=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(500,),random_state=42)

mlp.fit(x_train, y_train.ravel())
expected=y_test
predicted_mlp=mlp.predict(x_test)
accuracy_mlp= mlp.score(x_test, y_test)
print(accuracy_mlp)
print(metrics.classification_report(expected,predicted_mlp))
print(metrics.confusion_matrix(expected,predicted_mlp))
cm_mlp=metrics.confusion_matrix(expected,predicted_mlp)
cm_mlp_list=cm_mlp.tolist()
cm_mlp_list[0].insert(0,'Real True')
cm_mlp_list[1].insert(0,'Real False')
print tabulate(cm_mlp_list,headers=['Real/Pred','Pred True', 'Pred False'])



svm_l=svm.SVC(kernel="linear")
svm_l.fit(x_train, y_train.ravel())
predicted_svm=svm_l.predict(x_test)
accuracy_svm= svm_l.score(x_test, y_test)
print(accuracy_svm)
print(metrics.classification_report(expected,predicted_svm))
print(metrics.confusion_matrix(expected,predicted_svm))
cm_svm=metrics.confusion_matrix(expected,predicted_svm)
cm_svm_list=cm_svm.tolist()
cm_svm_list[0].insert(0,'Real True')
cm_svm_list[1].insert(0,'Real False')
print tabulate(cm_svm_list,headers=['Real/Pred','Pred True', 'Pred False'])




K_NN=KNeighborsClassifier()

K_NN.fit(x_train,y_train.ravel())

predicted_knn=K_NN.predict(x_test)

accuracy_knn=K_NN.score(x_test,y_test)
print(accuracy_knn)
print(metrics.classification_report(expected,predicted_knn))
print(metrics.confusion_matrix(expected,predicted_knn))
cm_knn=metrics.confusion_matrix(expected,predicted_knn)
cm_knn_list=cm_knn.tolist()
cm_knn_list[0].insert(0,'Real True')
cm_knn_list[1].insert(0,'Real False')
print tabulate(cm_knn_list,headers=['Real/Pred','Pred True', 'Pred False'])




dtc=DecisionTreeClassifier(random_state=42)

dtc.fit(x_train,y_train.ravel())
predicted_dtc=dtc.predict(x_test)

accuracy_dtc=dtc.score(x_test,y_test)
print(accuracy_dtc)
print(metrics.classification_report(expected,predicted_dtc))
print(metrics.confusion_matrix(expected,predicted_dtc))
cm_dtc=metrics.confusion_matrix(expected,predicted_dtc)
cm_dtc_list=cm_dtc.tolist()
cm_dtc_list[0].insert(0,'Real True')
cm_dtc_list[1].insert(0,'Real False')
print tabulate(cm_dtc_list,headers=['Real/Pred','Pred True', 'Pred False'])




mlp_tpr=np.array([.16,1])
svm_tpr=np.array([.23,1])
knn_tpr=np.array([.3,.70])
dtc_tpr=np.array([.33,.63])


plt.scatter(mlp_tpr[0], mlp_tpr[1], label = 'MLPClassifier', facecolors='brown', edgecolors='orange', s=300)
plt.scatter(svm_tpr[0], svm_tpr[1], label = 'SVM', facecolors='black', edgecolors='orange', s=300)
plt.scatter(knn_tpr[0], knn_tpr[1], label = 'KNeigborsClassifier', facecolors='green', edgecolors='black', s=300)
plt.scatter(dtc_tpr[0], dtc_tpr[1], label = 'DecisionTreeClassifier', facecolors='red', edgecolors='black', s=300)

plt.plot([0, 1.0], [0, 1.5], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.5])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='upper right')

plt.show()

