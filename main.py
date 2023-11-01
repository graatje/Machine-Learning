import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)


class SpaceTitanic:
    def __init__(self):
        self.dataset_df = pd.read_csv('train.csv')
        # Get what keys are present in the dataset
        print(self.dataset_df.keys())
        # ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
        # 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        # 'Name', 'Transported']

    def clean_dataset(self):
        # Remove the columns that are not useful
        self.dataset_df = self.dataset_df.drop(['PassengerId', 'Name', 'HomePlanet', 'Destination'], axis=1)

        # Fill the dataset with 0s for the missing values in these columns
        self.dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']] = (
            self.dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']].fillna(value=0))

        # Fill the missing values in Age with the mean
        self.dataset_df['Age'] = self.dataset_df['Age'].fillna(self.dataset_df['Age'].mean())

        # Remove the rows with missing values in categorical columns
        self.dataset_df = self.dataset_df.dropna()

        # Convert the booleans to integers cause the model can't handle booleans
        self.dataset_df['Transported'] = self.dataset_df['Transported'].astype(int)
        self.dataset_df['VIP'] = self.dataset_df['VIP'].astype(int)
        self.dataset_df['CryoSleep'] = self.dataset_df['CryoSleep'].astype(int)

        # Split the Cabin column into 3 different columns
        self.dataset_df[['Deck', 'Cabin_num', 'Side']] = self.dataset_df['Cabin'].str.split("/", expand=True)

        # The deck and the side are character columns, so we need to convert them to numeric
        self.dataset_df['Deck'] = pd.Categorical(self.dataset_df['Deck'])
        self.dataset_df['Deck'] = self.dataset_df['Deck'].cat.codes
        self.dataset_df['Side'] = pd.Categorical(self.dataset_df['Side'])
        self.dataset_df['Side'] = self.dataset_df['Side'].cat.codes

        # Drop the original Cabin column
        self.dataset_df = self.dataset_df.drop(['Cabin'], axis=1)

        # Check if there are any missing values
        print(self.dataset_df.isnull().sum().sort_values(ascending=False))

        # Convert all the columns to numeric
        print(self.dataset_df.dtypes)
        self.dataset_df = self.dataset_df.apply(pd.to_numeric)

        # Print the first 5 rows
        print(self.dataset_df.head())

    def split_dataset(self, dataset, test_ratio=0.20):
        # Split the dataset into training and testing dataset
        train_ds, valid_ds = train_test_split(dataset, test_size=test_ratio, random_state=1)
        return train_ds, valid_ds

    def sigmoid(self, x):
        return np.divide(1, np.add(1, np.exp(-x)))

    def compute_cost(self):
        pass

    def gradient(self):
        pass

    """
    Train the model using logistic regression and gradient descent
    """
    def train_model(self, train_ds, valid_ds):
        # Get the features and the labels
        train_features = train_ds.drop('Transported', axis=1)
        train_labels = train_ds['Transported']
        valid_features = valid_ds.drop('Transported', axis=1)
        valid_labels = valid_ds['Transported']

        # Initialize the weights and the bias
        weights = np.zeros(train_features.shape[1])
        bias = 0

        # Set the learning rate
        learning_rate = 0.01

        # Set the number of epochs
        epochs = 1000

        # Train the model
        for epoch in range(epochs):
            # Get the predictions

            predictions = self.sigmoid(np.dot(train_features, weights) + bias)

            # Compute the cost
            cost = -np.sum(np.multiply(train_labels, np.log(predictions)) +
                           np.multiply(1 - train_labels, np.log(1 - predictions))) / train_features.shape[0]

            # Compute the gradients
            dw = np.dot(train_features.T, (predictions - train_labels)) / train_features.shape[0]
            db = np.sum(predictions - train_labels) / train_features.shape[0]

            # Update the weights and the bias
            weights -= learning_rate * dw
            bias -= learning_rate * db

            # Print the cost every 100 epochs
            if epoch % 100 == 0:
                print("Cost after {} epochs: {}".format(epoch, cost))

        # Get the predictions on the validation set
        predictions = self.sigmoid(np.dot(valid_features, weights) + bias)

        # Convert the predictions to 0s and 1s
        predictions = np.round(predictions)

        # Compute the accuracy
        accuracy = np.sum(predictions == valid_labels) / valid_features.shape[0]
        print("Accuracy: {:.2f}%".format(accuracy * 100))



# train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
# print("{} examples in training, {} examples in testing.".format(
#     len(train_ds_pd), len(valid_ds_pd)))
#
# # Random forest classifier
# rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# rf.fit(train_ds_pd.drop('Transported', axis=1), train_ds_pd['Transported'])
# print("Random Forest Accuracy: {:.2f}%".format(rf.score(valid_ds_pd.drop('Transported', axis=1),
#                                                             valid_ds_pd['Transported']) * 100))

if __name__ == "__main__":
    space_titanic = SpaceTitanic()
    space_titanic.clean_dataset()
    train_ds, valid_ds = space_titanic.split_dataset(space_titanic.dataset_df)
    space_titanic.train_model(train_ds, valid_ds)