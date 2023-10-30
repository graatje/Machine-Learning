import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
pd.set_option('display.max_columns', None)

dataset_df = pd.read_csv('train.csv')

# Get what keys are present in the dataset
print(dataset_df.keys())
# ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
# 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
# 'Name', 'Transported']

# Split the dataset into features and labels
dataset_df = dataset_df.drop(['PassengerId', 'Name'], axis=1)

dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']] = dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']].fillna(value=0)

dataset_df['Age'] = dataset_df['Age'].fillna(dataset_df['Age'].mean())

# remove the rows with missing values in categorical columns
dataset_df = dataset_df.dropna()

dataset_df['Transported'] = dataset_df['Transported'].astype(int)
dataset_df['VIP'] = dataset_df['VIP'].astype(int)
dataset_df['CryoSleep'] = dataset_df['CryoSleep'].astype(int)

dataset_df[['Deck', 'Cabin_num', 'Side']] = dataset_df['Cabin'].str.split("/", expand=True)


try:
    dataset_df = dataset_df.drop(['Cabin'], axis=1)
except KeyError:
    print("field does not exist")

print(dataset_df.isnull().sum().sort_values(ascending=False))

print(dataset_df.head())


# def split_dataset(dataset, test_ratio=0.20):
#     test_indices = np.random.rand(len(dataset)) < test_ratio
#     return dataset[~test_indices], dataset[test_indices]
#
#
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
    pass
