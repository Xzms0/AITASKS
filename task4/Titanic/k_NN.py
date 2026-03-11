import pandas as pd
from sklearn import neighbors
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent
TRAIN_DIR = ROOT_DIR / 'data' / 'train'
SUBMISSION_DIR = ROOT_DIR / 'data' / 'submission'

class Titanic_kNN:
    def __init__(self):
        self.train_data = pd.read_csv(TRAIN_DIR / 'train.csv')
        self.test_data = pd.read_csv(TRAIN_DIR / 'test.csv')
        self.test_id = self.test_data['PassengerId']

    def preprocess(self):
        #删除不必要数据
        drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        self.train_data.drop(drop_columns, axis=1, inplace=True)
        self.test_data.drop(drop_columns, axis=1, inplace=True)

        #处理性别数据
        sex_map = {'male': 0, 'female': 1}
        self.train_data['Sex'] = self.train_data['Sex'].map(sex_map)
        self.test_data['Sex'] = self.test_data['Sex'].map(sex_map)

        #处理年龄数据
        age_mean = self.train_data['Age'].mean()
        self.train_data['Age'] = self.train_data['Age'].fillna(age_mean)
        self.test_data['Age'] = self.test_data['Age'].fillna(age_mean)
        
        #处理票价数据
        fare_mean = self.train_data['Fare'].mean()
        self.train_data['Fare'] = self.train_data['Fare'].fillna(fare_mean)
        self.test_data['Fare'] = self.test_data['Fare'].fillna(fare_mean)
        
        #处理港口数据
        embarked_dummies = pd.get_dummies(self.train_data['Embarked'], prefix='Embarked', drop_first=True)
        self.train_data = pd.concat([self.train_data, embarked_dummies], axis=1)
        self.train_data.drop(['Embarked'], axis=1, inplace=True)

        embarked_dummies = pd.get_dummies(self.test_data['Embarked'], prefix='Embarked', drop_first=True)
        self.test_data = pd.concat([self.test_data, embarked_dummies], axis=1)
        self.test_data.drop(['Embarked'], axis=1, inplace=True)


    def train(self):
        model = neighbors.KNeighborsClassifier(n_neighbors=7)
        model.fit(self.train_data.drop('Survived', axis=1),
                self.train_data['Survived'])
        return model


    def predict(self,model):
        prediction = model.predict(self.test_data)

        submission = pd.DataFrame({
            "PassengerId": self.test_id,
            "Survived": prediction
        })

        submission.to_csv(SUBMISSION_DIR / 'submission_knn.csv', index=False)
        return submission


def main():
    titanic_knn = Titanic_kNN()
    titanic_knn.preprocess()
    model = titanic_knn.train()
    submission = titanic_knn.predict(model)
    print(submission)

if __name__ == '__main__':
    main()
