import pickle
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

# 1. Load the saved model and vectorizer
with open('model.bin', 'rb') as f:
    dv, model = pickle.load(f)

# 2. Initialize the FastAPI app
app = FastAPI(title="E-Commerce Delivery Prediction API", description="Predicts if a shipment will be late.")

# 3. Define the exact JSON payload schema for validation
class DeliveryData(BaseModel):
    warehouse_block: str
    mode_of_shipment: str
    customer_care_calls: int
    customer_rating: int
    cost_of_the_product: int
    prior_purchases: int
    product_importance: str
    gender: str
    discount_offered: int
    weight_in_gms: int

    # Optimization: Provide a realistic default payload for the Swagger UI
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "warehouse_block": "D",
                    "mode_of_shipment": "Ship",
                    "customer_care_calls": 4,
                    "customer_rating": 3,
                    "cost_of_the_product": 177,
                    "prior_purchases": 3,
                    "product_importance": "low",
                    "gender": "F",
                    "discount_offered": 44,
                    "weight_in_gms": 1233
                }
            ]
        }
    }

# 4. Create the prediction endpoint
@app.post('/predict')
def predict_delivery(data: DeliveryData):
    # Convert validated Pydantic object back to a dictionary
    customer_dict = data.model_dump()
    
    # Vectorize and format for XGBoost
    X = dv.transform([customer_dict])
    dMatrix = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())
    
    # Generate prediction
    prediction = model.predict(dMatrix)[0]
    is_late = prediction >= 0.5
    
    return {
        'late_probability': float(prediction),
        'will_be_late': bool(is_late)
    }