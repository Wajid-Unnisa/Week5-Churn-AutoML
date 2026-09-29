import numpy as np
import pandas as pd
from pycaret.classification import load_model


def load_churn_model():
    model = load_model('churn_model')
    return model

def predict_churn_probability(df):
    model = load_churn_model()
    probabilities = model.predict_proba(df)
    churn_probabilities = probabilities[:, 1]

    return pd.Series(
        churn_probabilities,
        index=df.index,
        name='churn_probability'
    )

def prepare_churn_data(df):
    df = df.copy()

    df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})

    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    ).fillna(0)

    df['Contract_One year'] = (df['Contract'] == 'One year').astype(int)
    df['Contract_Two year'] = (df['Contract'] == 'Two year').astype(int)

    df['PaymentMethod_Credit card (automatic)'] = (
        df['PaymentMethod'] == 'Credit card (automatic)'
    ).astype(int)

    df['PaymentMethod_Electronic check'] = (
        df['PaymentMethod'] == 'Electronic check'
    ).astype(int)

    df['PaymentMethod_Mailed check'] = (
        df['PaymentMethod'] == 'Mailed check'
    ).astype(int)

    df['ChargesPerMonth'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )

    model_columns = [
        'tenure',
        'PhoneService',
        'MonthlyCharges',
        'TotalCharges',
        'Contract_One year',
        'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check',
        'ChargesPerMonth'
    ]

    return df[model_columns]

    
def prepare_modified_churn_data(df):
    df = df.copy()

    df['Contract_One year'] = (df['Contract'] == 1).astype(int)
    df['Contract_Two year'] = (df['Contract'] == 2).astype(int)

    df['PaymentMethod_Credit card (automatic)'] = (
        df['PaymentMethod'] == 0
    ).astype(int)

    df['PaymentMethod_Electronic check'] = (
        df['PaymentMethod'] == 2
    ).astype(int)

    df['PaymentMethod_Mailed check'] = (
        df['PaymentMethod'] == 1
    ).astype(int)

    df['ChargesPerMonth'] = df['charge_per_tenure']

    model_columns = [
        'tenure',
        'PhoneService',
        'MonthlyCharges',
        'TotalCharges',
        'Contract_One year',
        'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check',
        'ChargesPerMonth'
    ]

    return df[model_columns]

def add_percentiles(results, training_probabilities):
    results = results.copy()

    results['percentile'] = (
        results['churn_probability']
        .apply(
            lambda p:
            (training_probabilities <= p).mean() * 100
        )
    )

    return results

def make_predictions(df, training_probabilities=None):
    if 'charge_per_tenure' in df.columns:
        prepared_df = prepare_modified_churn_data(df)
    else:
        prepared_df = prepare_churn_data(df)

    probabilities = predict_churn_probability(prepared_df)

    results = pd.DataFrame({
        'churn_probability': probabilities,
        'predicted_churn': (probabilities >= 0.5).astype(int)
    })

    if training_probabilities is not None:
        results = add_percentiles(results, training_probabilities)

    print("Predictions:")
    print(results)

    return results

class ChurnPredictor:
    def __init__(self):
        self.model = load_churn_model()

    def predict_probability(self, df):
        probabilities = self.model.predict_proba(df)
        churn_probabilities = probabilities[:, 1]

        return pd.Series(
            churn_probabilities,
            index=df.index,
            name='churn_probability'
        )

    def prepare_data(self, df):
        if 'charge_per_tenure' in df.columns:
            return prepare_modified_churn_data(df)
        else:
            return prepare_churn_data(df)

    def predict(self, df, training_probabilities=None):
        prepared_df = self.prepare_data(df)

        probabilities = self.predict_probability(prepared_df)

        results = pd.DataFrame({
            'churn_probability': probabilities,
            'predicted_churn': (probabilities >= 0.5).astype(int)
        })

        if training_probabilities is not None:
            results = add_percentiles(
                results,
                training_probabilities
            )

        print("Predictions:")
        print(results)

        return results

    def predict_from_file(self, training_probabilities=None):
        filename = input("Enter the CSV filename: ")

        df = pd.read_csv(filename)

        return self.predict(
            df,
            training_probabilities
        )

if __name__ == "__main__":
    predictor = ChurnPredictor()
    predictor.predict_from_file()