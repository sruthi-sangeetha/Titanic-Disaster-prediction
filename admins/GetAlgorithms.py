# linear algebra
import numpy as np

# data processing
import pandas as pd

# data visualization
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn import tree
import matplotlib
#matplotlib.use('Agg')




class AlgorithmsAccuracy:

    def processAlgorithms(self, trainPath, testPath):
        test_df = pd.read_csv(testPath)
        train_df = pd.read_csv(trainPath)
        total = train_df.isnull().sum().sort_values(ascending=False)
        percent_1 = train_df.isnull().sum() / train_df.isnull().count() * 100
        percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
        missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
        # 1. Age and Sex
        survived = 'survived'
        not_survived = 'not survived'
        #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        women = train_df[train_df['Sex'] == 'female']
        men = train_df[train_df['Sex'] == 'male']
        #ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[0],kde=False)
        #plt.show()
        #ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[0],kde=False)
        #plt.show()
        #ax.legend()
        #ax.set_title('Female')
        #ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[1], kde=False)
        #ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[1], kde=False)
        #ax.legend()
        #_ = ax.set_title('Male')

        # 3. Embarked, Pclass and Sex:

        #FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)
        #FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None, order=None, hue_order=None)
        #FacetGrid.add_legend()
        #plt.show()
        # 4. Pclass:

        #sns.barplot(x='Pclass', y='Survived', data=train_df)
        #plt.show()

        #grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
        #grid.map(plt.hist, 'Age', alpha=.5, bins=20)
        #grid.add_legend();
        #plt.show()

        #sns.heatmap(train_df[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].corr(), annot=True)
        #plt.show()
        # 5. SibSp and Parch
        # db_statuc = StoreTrainDataInDb(train_df)
        # db_statuc = StoreTestDataInDb(test_df)

        data = [train_df, test_df]
        for dataset in data:
            dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
            dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
            dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
            dataset['not_alone'] = dataset['not_alone'].astype(int)
        train_df['not_alone'].value_counts()
        #axes = sns.factorplot('relatives', 'Survived', data=train_df, aspect=2.5, )
        #plt.show()
        #html = train_df.to_html()

        # Data Preprocessing########################
        train_df = train_df.drop(['PassengerId'], axis=1)
        import re
        deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
        data = [train_df, test_df]
        for dataset in data:
            dataset['Cabin'] = dataset['Cabin'].fillna("U0")
            dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
            dataset['Deck'] = dataset['Deck'].map(deck)
            dataset['Deck'] = dataset['Deck'].fillna(0)
            dataset['Deck'] = dataset['Deck'].astype(int)
        # we can now drop the cabin feature
        train_df = train_df.drop(['Cabin'], axis=1)
        test_df = test_df.drop(['Cabin'], axis=1)
        data = [train_df, test_df]
        for dataset in data:
            mean = train_df["Age"].mean()
            std = test_df["Age"].std()
            is_null = dataset["Age"].isnull().sum()
            # compute random numbers between the mean, std and is_null
            rand_age = np.random.randint(mean - std, mean + std, size=is_null)
            # fill NaN values in Age column with random values generated
            age_slice = dataset["Age"].copy()
            age_slice[np.isnan(age_slice)] = rand_age
            dataset["Age"] = age_slice
            dataset["Age"] = train_df["Age"].astype(int)
        train_df["Age"].isnull().sum()
        #print(train_df['Embarked'].describe())
        common_value = 'S'
        data = [train_df, test_df]
        for dataset in data:
            dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
        #print(train_df.info())
        data = [train_df, test_df]
        for dataset in data:
            dataset['Fare'] = dataset['Fare'].fillna(0)
            dataset['Fare'] = dataset['Fare'].astype(int)
        data = [train_df, test_df]
        titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        for dataset in data:
            # extract titles
            dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
            # replace titles with a more common title or as Rare
            dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', \
                                                         'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
            # convert titles into numbers
            dataset['Title'] = dataset['Title'].map(titles)
            # filling NaN with 0, to get safe
            dataset['Title'] = dataset['Title'].fillna(0)
        train_df = train_df.drop(['Name'], axis=1)
        test_df = test_df.drop(['Name'], axis=1)
        genders = {"male": 0, "female": 1}
        data = [train_df, test_df]
        for dataset in data:
            dataset['Sex'] = dataset['Sex'].map(genders)
        train_df['Ticket'].describe()
        train_df = train_df.drop(['Ticket'], axis=1)
        test_df = test_df.drop(['Ticket'], axis=1)
        ports = {"S": 0, "C": 1, "Q": 2}
        data = [train_df, test_df]
        for dataset in data:
            dataset['Embarked'] = dataset['Embarked'].map(ports)
        data = [train_df, test_df]
        for dataset in data:
            dataset['Age'] = dataset['Age'].astype(int)
            dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
            dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
            dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
            dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
            dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
            dataset.loc[dataset['Age'] > 66, 'Age'] = 6
        # let's see how it's distributed train_df['Age'].value_counts()
        #print(train_df.head(10))
        data = [train_df, test_df]
        for dataset in data:
            dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
            dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
            dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
            dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare'] = 3
            dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare'] = 4
            dataset.loc[dataset['Fare'] > 250, 'Fare'] = 5
            dataset['Fare'] = dataset['Fare'].astype(int)
        data = [train_df, test_df]
        for dataset in data:
            dataset['Age_Class'] = dataset['Age'] * dataset['Pclass']
        for dataset in data:
            dataset['Fare_Per_Person'] = dataset['Fare'] / (dataset['relatives'] + 1)
            dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
        # Let's take a last look at the training set, before we start training the models.
        #print(train_df.head(10))
        X_train = train_df.drop("Survived", axis=1)
        Y_train = train_df["Survived"]
        X_test = test_df.drop("PassengerId", axis=1).copy()
        # Stochastic Gradient Descent (SGD):
        sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
        sgd.fit(X_train, Y_train)
        Y_pred = sgd.predict(X_test)
        sgd.score(X_train, Y_train)
        acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
        # Random Forest:
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(X_train, Y_train)
        Y_prediction = random_forest.predict(X_test)
        random_forest.score(X_train, Y_train)
        acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
        # Logistic Regression:
        logreg = LogisticRegression()
        logreg.fit(X_train, Y_train)
        Y_pred = logreg.predict(X_test)
        acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
        # K Nearest Neighbor:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
        # Gaussian Naive Bayes:
        gaussian = GaussianNB()
        gaussian.fit(X_train, Y_train)
        Y_pred = gaussian.predict(X_test)
        acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
        # Perceptron:
        perceptron = Perceptron(max_iter=5)
        perceptron.fit(X_train, Y_train)
        Y_pred = perceptron.predict(X_test)
        acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
        # Linear Support Vector Machine:
        linear_svc = LinearSVC()
        linear_svc.fit(X_train, Y_train)
        Y_pred = linear_svc.predict(X_test)
        acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

        # Decision Tree

        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, Y_train)


        from sklearn import tree
        #print(tree.export_graphviz(decision_tree))
        dotfile = open("dtree2.dot", 'w')
        tree.export_graphviz(decision_tree, out_file=dotfile)

        Y_pred = decision_tree.predict(X_test)
        # tree.plot_tree(decision_tree);

        plt.show()
        acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

        # Which is the best Model ?

        results = pd.DataFrame({
            'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                      'Random Forest', 'Naive Bayes', 'Perceptron',
                      'Stochastic Gradient Decent',
                      'Decision Tree'],
            'Score': [acc_linear_svc, acc_knn, acc_log,
                      acc_random_forest, acc_gaussian, acc_perceptron,
                      acc_sgd, acc_decision_tree]})
        result_df = results.sort_values(by='Score', ascending=False)
        result_df = result_df.set_index('Score')
        #print(result_df.head(9))
        myDict = {'Logistic Regression':acc_log,'Decision Tree':acc_decision_tree,'Decision Tree with Hypertuning':acc_random_forest,'K - Nearest Neighbors and':acc_knn,'Support Vector Machines':acc_linear_svc}
        return myDict

