from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import traceback
from aml_pipeline import AMLFraudDetector

app = FastAPI(title="AML Fraud Detection API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ML Model (It will auto-train on synthetic data so you can test it immediately!)
detector = AMLFraudDetector()

@app.post("/upload-training-data")
async def upload_training_data(file: UploadFile = File(...)):
    """
    Upload a CSV file containing historical data with an 'is_fraud' column to train the model.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Train the model on the uploaded dataset
        detector.train_on_dataframe(df, target_column='is_fraud')
        
        return {"message": "Model successfully trained on uploaded CSV data!", "rows_processed": len(df)}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assess-transactions-csv")
async def assess_transactions_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file of new transactions. Returns the Risk Score and SHAP explanation for each row.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Get predictions for the entire batch
        results = detector.predict_batch(df)
        
        # Identify suspects with a high risk score (e.g. >= 70% risk)
        suspect_threshold = 0.70
        suspects = [r for r in results if r["risk_score"] >= suspect_threshold]
        
        return {
            "message": f"Processed {len(results)} transactions. Found {len(suspects)} suspect(s)!",
            "total_transactions": len(results),
            "suspects_found": len(suspects),
            "suspected_members": suspects,
            "all_results": results
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Note: You can run this API using:
# uvicorn main_api:app --reload
