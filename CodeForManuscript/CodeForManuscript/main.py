import pandas as pd
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("The specified file could not be found.")
        return None

def preprocess_data(data):
    le = LabelEncoder()
    data['attack_type'] = le.fit_transform(data['attack_type'])
    X = data.drop("attack_type", axis=1)
    y = data["attack_type"]
    X = pd.get_dummies(X)
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def optimize_hyperparameters(X_train, y_train):

    param_grid = {
        'n_estimators': [100, 200, 300],  
        'max_features': ['auto', 'sqrt'],  
        'max_depth': [10, 20, 30, 40, 50],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4],  
        'bootstrap': [True, False]  
    }

    
    cf = SVC(kernel='rbf', probability=True)

    
    cf_random = RandomizedSearchCV(estimator=cf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
   cf_random.fit(X_train, y_train)

    return cf_random.best_params_

def main():
# Each IDS dataset is loaded sequentially in each transaction. Our datasets are KDDCUP’99, NSL-KDD # and UNSW-B15. The KDDCUP'99 dataset is shown here as an example.
    print("Loading dataset..")
    data = load_data("KDDCUP99.csv")
    if data is None:
        return

    print("\nData")
    print("Total record: ", len(data))
    print(data.head())
    print(data.info())

    print("\nData Preprocessing… ")
    X, y = preprocess_data(data)

    print("\nDividing Data into Training and Testing Sets")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nHyperparameters Optimization")
    best_params = optimize_hyperparameters(X_train, y_train)
    

if __name__ == "__main__":
    main()
