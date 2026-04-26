import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap
from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
except Exception as e:
    class BaseTool(BaseModel):
        pass

class AMLFraudDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.explainer = None
        # These are the expected columns in your uploaded CSV
        self.features = [
            'transaction_count',
            'transaction_amount',
            'account_dormancy_days',
            'high_risk_geographic_transfers',
            'crypto_involvement'
        ]
        # Automatically train on dummy data so it works without needing a CSV immediately
        self.train_on_dummy_data()

    def train_on_dummy_data(self):
        import numpy as np
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'transaction_count': np.random.poisson(5, n),
            'transaction_amount': np.random.exponential(1000, n),
            'account_dormancy_days': np.random.poisson(30, n),
            'high_risk_geographic_transfers': np.random.binomial(5, 0.1, n),
            'crypto_involvement': np.random.binomial(1, 0.2, n)
        })
        df['is_fraud'] = ((df['transaction_amount'] > 3000) | (df['high_risk_geographic_transfers'] > 1)).astype(int)
        self.train_on_dataframe(df, target_column='is_fraud')

    def train_on_dataframe(self, df: pd.DataFrame, target_column: str = 'is_fraud'):
        """Train the model on a real Pandas DataFrame uploaded from the backend."""
        # Ensure all required features exist in the uploaded CSV
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Uploaded CSV is missing required columns: {missing_features}")
            
        if target_column not in df.columns:
            raise ValueError(f"Uploaded CSV must have a target column named '{target_column}' for training.")

        X = df[self.features]
        y = df[target_column]
        
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

    def predict_single(self, transaction_data: dict) -> dict:
        """Predict risk score and compute SHAP explanations for a single transaction."""
        if self.explainer is None:
            raise ValueError("Model is not trained yet! Please upload a training CSV first.")
            
        df = pd.DataFrame([transaction_data], columns=self.features)
        risk_score = float(self.model.predict_proba(df)[0][1])
        shap_values = self.explainer.shap_values(df)
        
        if isinstance(shap_values, list):
            contributions = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            contributions = shap_values[0, :, 1]
        else:
            contributions = shap_values[0]
        
        shap_explanation = {
            feature: float(contribution)
            for feature, contribution in zip(self.features, contributions)
        }
        shap_explanation = dict(sorted(shap_explanation.items(), key=lambda item: abs(item[1]), reverse=True))

        return {
            "risk_score": risk_score,
            "shap_explanation": shap_explanation
        }

    def predict_batch(self, df: pd.DataFrame) -> list:
        """Process an entire uploaded CSV of new transactions."""
        if self.explainer is None:
            raise ValueError("Model is not trained yet! Please upload a training CSV first.")
            
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Uploaded CSV is missing required columns: {missing_features}")

        X = df[self.features]
        risk_scores = self.model.predict_proba(X)[:, 1]
        shap_values_batch = self.explainer.shap_values(X)
        
        results = []
        for i in range(len(df)):
            if isinstance(shap_values_batch, list):
                contributions = shap_values_batch[1][i]
            elif len(shap_values_batch.shape) == 3:
                contributions = shap_values_batch[i, :, 1]
            else:
                contributions = shap_values_batch[i]
                
            shap_explanation = {
                feature: float(contribution)
                for feature, contribution in zip(self.features, contributions)
            }
            shap_explanation = dict(sorted(shap_explanation.items(), key=lambda item: abs(item[1]), reverse=True))
            
            results.append({
                "transaction_id": int(df.index[i]) if 'transaction_id' not in df.columns else str(df['transaction_id'].iloc[i]),
                "member_id": str(df['member_id'].iloc[i]) if 'member_id' in df.columns else "Unknown",
                "risk_score": float(risk_scores[i]),
                "shap_explanation": shap_explanation
            })
            
        return results

# ==========================================
# CrewAI Integration Example
# ==========================================
class AMLTransactionInput(BaseModel):
    transaction_count: int = Field(..., description="Number of transactions in a given period.")
    transaction_amount: float = Field(..., description="Total amount of transactions.")
    account_dormancy_days: int = Field(..., description="Days since last activity before current transactions.")
    high_risk_geographic_transfers: int = Field(..., description="Number of transfers to high risk geographic regions.")
    crypto_involvement: int = Field(..., description="1 if crypto is involved, 0 otherwise.")

class AMLRiskAssessmentTool(BaseTool):
    model_config = {'arbitrary_types_allowed': True}
    name: str = "AML Risk Assessment Tool"
    description: str = "Assess the AML risk of a transaction set and provide SHAP explainability."
    args_schema: type[BaseModel] = AMLTransactionInput
    detector: AMLFraudDetector = Field(default_factory=AMLFraudDetector)

    def _run(self, transaction_count: int, transaction_amount: float, account_dormancy_days: int, high_risk_geographic_transfers: int, crypto_involvement: int) -> str:
        data = {
            'transaction_count': transaction_count,
            'transaction_amount': transaction_amount,
            'account_dormancy_days': account_dormancy_days,
            'high_risk_geographic_transfers': high_risk_geographic_transfers,
            'crypto_involvement': crypto_involvement
        }
        result = self.detector.predict_single(data)
        return json.dumps(result, indent=2)
