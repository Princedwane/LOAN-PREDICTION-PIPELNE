import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("Loan_Pipeline") #experiment name

def load_data(loadpath):
    """
    The function reads a CSV file and returns the Dataframe

    parameters:
        loadpath(str): the function excpects a string which is the actusl path or name of the file

    Returns:
        a data frame is returned  

    """
    try:
        df = pd.read_csv(loadpath)
        print(f"Data is loaded sucessfully.Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{loadpath}' not found.")
        return None
    

def remove_missing(df):
    
    """
    The function: removes the missing values found in the file.

    Parameters: data frame recived and using the dropna() to automatically remove all empty values

    Returns:
        1-total number of missing values
        2- a clean data with the number of complete columns and rows

    """
    total_nulls= df.isnull().sum().sum()

    print(f"Total Nulls: {total_nulls}")

    if total_nulls >0:
        print("Nulls deleted:", total_nulls)
        return df.dropna()
    else:
        print("No missing values were actually found!")
        return df


def encode_data(df):
    """
    The function:encodes and map the columns data so that it can easly be interpreted by the model as 0 /1
    parameters: clean data frame after removing missing

    Returns: clean coded columns
    """
    df[' education'] = df[' education'].map({' Graduate': 1, ' Not Graduate': 0})
    df[' self_employed'] = df[' self_employed'].map({' Yes': 1, ' No': 0})
    df[' loan_status'] = df[' loan_status'].map({' Approved': 1, ' Rejected': 0})

    return df


def show_summary(df):
    """
    The function: Prints the summary of the data

    Parameters:
        Receives the clean df 
        Head(5) to print the first 5  elements(rows)
        shape to print the overall clean shape aftrer removing the nulls

    """
    #prints shape and the first five rows
    print("loan Shape:(columns ,rows)" ,df.shape)
    print(df.head(5))


def train_model(df,features ,test_size=0.3):
    """
    Trains a Logistic Regression model on the Loan_approval_dataset.
    Parameters:
        df: the clean dataframe received from remove_missing
        features: list of column names to use for training
        test_size: proportion of data to use for testing (default 0.2)

    Returns:
        the trained model and its accuracy score
    """
    
    X = df[features]
    y = df[" loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
     
    model = LogisticRegression()
    model.fit(X_train, y_train) #where the actual learning happen

    predictions = model.predict(X_test) #model predicts on unseen data
    accuracy = accuracy_score(y_test, predictions) #comparres rpredictions against real answers and provide percentage of accuracy

    return model, accuracy

if __name__ == "__main__":
    l = load_data("loan_approval_dataset.csv")
    if l is not None:
        c = remove_missing(l)
        e = encode_data(c)
        show_summary(e)

        # Run 1 — income annualy ,loan amount and credit score features
        with mlflow.start_run(run_name="Run 1--basic"):
            features = [" income_annum", " loan_amount", " cibil_score"]
            model, accuracy = train_model(e, features, test_size=0.3)
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("features", str(features))
            mlflow.log_param("test_size", 0.3)
            mlflow.log_metric("rows_before", len(l))
            mlflow.log_metric("rows_after", len(c))
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
            print(f"\nRun 1 Accuracy: {accuracy}")

        # Run 2 — loan term ,dependants and education features
        with mlflow.start_run(run_name="Run 2--mid"):
            features = [" loan_term", " no_of_dependents", " education"]
            model, accuracy = train_model(e, features, test_size=0.3)
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("features", str(features))
            mlflow.log_param("test_size", 0.3)
            mlflow.log_metric("rows_before", len(l))
            mlflow.log_metric("rows_after", len(c))
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
            print(f"\nRun 2 Accuracy: {accuracy}")

        # Run 3 — All features with test 0.3
        with mlflow.start_run(run_name="Run 3--All 0.3"):
            features = [" bank_asset_value"," commercial_assets_value"," residential_assets_value"," luxury_assets_value"," income_annum", " loan_amount", " self_employed"," cibil_score" ," loan_term", " no_of_dependents", " education"]
            model, accuracy = train_model(e, features, test_size=0.3)
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("features", str(features))
            mlflow.log_param("test_size", 0.3)
            mlflow.log_metric("rows_before", len(l))
            mlflow.log_metric("rows_after", len(c))
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
            print(f"\nRun 3 Accuracy: {accuracy}")

        # Run 4 — All features with test 0.5
        with mlflow.start_run(run_name="Run 4--All 0.5"):
            features = [" bank_asset_value"," commercial_assets_value"," residential_assets_value"," luxury_assets_value"," income_annum", " loan_amount", " self_employed"," cibil_score" ," loan_term", " no_of_dependents", " education"]
            model, accuracy = train_model(e, features, test_size=0.5)
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("features", str(features))
            mlflow.log_param("test_size", 0.5)
            mlflow.log_metric("rows_before", len(l))
            mlflow.log_metric("rows_after", len(c))
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
            print(f"\nRun 4 Accuracy: {accuracy}")
    


"""
features 
[ ' no_of_dependents', ' education', ' self_employed', 
' income_annum', ' loan_amount', ' loan_term', ' cibil_score', 
' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value',
 ' bank_asset_value', 
 target(y)
 ' loan_status']
"""    