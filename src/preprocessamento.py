import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

def processamento(dataset):
    """
    This function processes the input dataset by encoding categorical variables and preparing them for further analysis.

    Parameters:
    dataset (pd.DataFrame): The input dataset containing the data to be processed.

    Returns:
    pd.DataFrame: The processed dataset with encoded categorical variables.

    The function performs the following steps:
    1. Encodes categorical variables using LabelEncoder for 'Gender' and 'MealType' columns.
    2. Encodes ordinal variables using OrdinalEncoder for 'VisitFrequency' and 'TimeOfVisit' columns based on the provided mappings.
    3. Encodes target variables using TargetEncoder for 'PreferredCuisine' and 'DiningOccasion' columns based on the 'HighSatisfaction' column.

    The processed dataset is returned for further analysis or modeling.
    """
    le = LabelEncoder()

    colunas_label = ['Gender', 'MealType']
    for coluna in colunas_label:
        dataset[coluna] = le.fit_transform(dataset[coluna])
        dataset[coluna] = dataset[coluna].astype('int64')

    ordinal_cols = ['VisitFrequency', 'TimeOfVisit']
    ordinal_mappings = [
        {'col': 'VisitFrequency', 'mapping': ['Rarely', 'Monthly', 'Weekly', 'Daily']},
        {'col': 'TimeOfVisit', 'mapping': ['Breakfast', 'Lunch', 'Dinner']},
    ]
    categories = [mapping['mapping'] for mapping in ordinal_mappings]
    oe = OrdinalEncoder(categories=categories)
    dataset[ordinal_cols] = oe.fit_transform(dataset[ordinal_cols])

    colunas_target = ['PreferredCuisine', 'DiningOccasion']
    te = TargetEncoder(cols=colunas_target)
    dataset[colunas_target] = te.fit_transform(dataset[colunas_target], dataset['HighSatisfaction'])

    return dataset

if __name__ == '__main__':
    df = pd.read_csv("./data/restaurant_customer_satisfaction.csv")
    processamento(df)