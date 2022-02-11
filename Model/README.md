# Use SVC.model

Before applying the models on test data, use pd.get_dummies to convert categorical columns into binary objects.


                    Model  Accuracy  Precision    Recall        F1
0      LogisticRegression  0.865706   0.863985  0.873227  0.867570
1    KNeighborsClassifier  0.865714   0.848936  0.894006  0.870228
2  DecisionTreeClassifier  0.801098   0.810482  0.800932  0.802215
3  RandomForestClassifier  0.878634   0.865016  0.901632  0.882145
4             BernoulliNB  0.863142   0.865190  0.862970  0.863689
5              GaussianNB  0.864382   0.861131  0.873260  0.866033
6                     SVC  0.873473   0.855653  0.904262  0.878274

===========RF============

Confusion Matrix:

 [[ 92  17]
 [ 15 106]]

Accuracy: 0.8608695652173913
Precision: 0.8440366972477065
Misclassification: 0.1391304347826087
Specificity: 0.8617886178861789
Sensitivity: 0.8598130841121495
Matthews Corr: 0.7208353215797065

===========SVC_base============

Confusion Matrix:

 [[ 92  17]
 
 [ 11 110]]

Accuracy: 0.8782608695652174

Precision: 0.8440366972477065

Misclassification: 0.12173913043478261

Specificity: 0.8661417322834646

Sensitivity: 0.8932038834951457

Matthews Corr: 0.7562302202339356
