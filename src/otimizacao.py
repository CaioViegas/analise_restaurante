import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from preprocessamento import processamento

def modelo_otimizado(dataset, target):
    """
    This function optimizes a machine learning model using GridSearchCV and evaluates its performance.
    It applies preprocessing steps, trains a Logistic Regression model with PCA and feature selection,
    and evaluates the model using accuracy, precision, recall, and F1 score.
    It also performs cross-validation to assess the model's generalization ability.

    Parameters:
    dataset (pandas.DataFrame): The dataset containing the features and target variable.
    target (str): The name of the target variable column in the dataset.

    Returns:
    None
    """
    dataset = processamento(dataset)

    X = dataset.drop(target, axis=1)
    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('pca', PCA(n_components=16)),
        ('selector', SelectKBest(score_func=f_classif)),
        ('model', LogisticRegression())
    ])

    param_grid = {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'model__penalty': ['l1', 'l2', 'elasticnet'],
        'model__solver': ['liblinear', 'saga'],
        'model__max_iter': [100, 200, 500],
        'model__tol': [0.0001, 0.001, 0.01],
        'model__fit_intercept': [True, False],
        'model__intercept_scaling': [1, 2, 3] 
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f'Best Parameters: {best_params}')

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  
    recall = recall_score(y_test, y_pred, average='weighted')  
    f1 = f1_score(y_test, y_pred, average='weighted')  

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

    print(f'Cross-Validation Scores: {cv_scores}')
    print(f'Mean Cross-Validation Score: {cv_scores.mean()}')

    joblib.dump(best_model, 'melhor_modelo.joblib')

if __name__ == '__main__':
    df = pd.read_csv("./data/restaurant_customer_satisfaction.csv")
    modelo_otimizado(df, "HighSatisfaction")