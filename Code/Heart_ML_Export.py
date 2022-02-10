import pandas as pd 

# Get data within range: Same Sex, age +- 8
def getdata(df, sex, age, threshold, code):
    upper = age + threshold
    lower = age - threshold
    return df[(df['Sex'] == sex) & (df['Age'] >= lower)  & (df['Age'] <= upper) & (df[code] > 0)]

def ImputeMissing(df, threshold):
    code = ['RestingBP', 'Cholesterol']

    for c in code:
        
        rbp = df[df[c] == 0]

        for index, value in rbp.iterrows():
            temp = getdata(df, value['Sex'], value['Age'],threshold,c)
            df.at[index,c] = temp[c].mean()

    return df

df = pd.read_csv('Data/patient_data_train.csv')

df = ImputeMissing(df, 8)


# Construct ML
X = pd.get_dummies(df.drop('HeartDisease',axis=1))
y = df['HeartDisease']

# Balance data
from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy='minority')
X_sm, y_sm = sm.fit_resample(X, y)
print('======== After balancing Data =========')
print(f"Num of heart disease data: {y_sm[y_sm==1].count()}")
print(f"Num of non-heart disease data: {y_sm[y_sm==0].count()}")

from sklearn.preprocessing import StandardScaler
def Standardizing(df):

    scaler = StandardScaler()
    X_sm_std = df.copy() # create copy
    X_sm_std[['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']] = scaler.fit_transform(X_sm_std[['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']])
    return X_sm_std

# Model
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC
import joblib
def model_comparison(x,y):
    all_model = [LogisticRegression(max_iter=10000), KNeighborsClassifier(), DecisionTreeClassifier(random_state = 41),
                RandomForestClassifier(random_state = 41), BernoulliNB(), GaussianNB(), SVC()]

    recall = []
    precision = []
    f1=[]
    accuracy = []

    modelname = ['LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier',
             'RandomForestClassifier', 'BernoulliNB', 'GaussianNB', 'SVC']
    count = 0
    for model in all_model:

        c = 5 # small dataset
        cv = cross_val_score(model, x, y, scoring='accuracy', cv=c, n_jobs = -1).mean()
        accuracy.append(cv)

        cv = cross_val_score(model, x, y, scoring='recall', cv=c, n_jobs = -1).mean()
        recall.append(cv)

        cv = cross_val_score(model, x, y, scoring='precision', cv=c, n_jobs = -1).mean()
        precision.append(cv)

        cv = cross_val_score(model, x, y, scoring='f1', cv=c, n_jobs = -1).mean()
        f1.append(cv)

        m = model
        m.fit(x,y)
        joblib.dump(m,f'{modelname[count]}.model')
        count += 1

    score = pd.DataFrame({'Model': modelname, 'Accuracy' : accuracy, 'Precision': precision, 'Recall': recall, 'F1':f1})
    score.style.background_gradient(high=1,axis=0)
    
    return score



model = model_comparison(X_sm,y_sm)
print(model)

rf = RandomForestClassifier(random_state = 41)
rf.fit(X_sm,y_sm)
