import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. Load and clean data
df = pd.read_csv('Train.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')
df = df.rename(columns={'reached_on_time_y_n': 'target'})
df.drop('id', axis=1, inplace=True)

# 2. Split data (using the full train set for final model)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
y_full_train = df_full_train.target.values
del df_full_train['target']

numerical = ['customer_care_calls', 'customer_rating', 'cost_of_the_product', 'prior_purchases', 'discount_offered', 'weight_in_gms']
categorical = ['warehouse_block', 'mode_of_shipment', 'product_importance', 'gender']

# 3. Vectorize
dv = DictVectorizer(sparse=False)
train_dict = df_full_train[categorical + numerical].to_dict(orient='records')
X_full_train = dv.fit_transform(train_dict)

# 4. Train final XGBoost model
dtrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.get_feature_names_out().tolist())
xgb_params = {
    'eta': 0.1, 
    'max_depth': 4,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 42
}
model = xgb.train(xgb_params, dtrain, num_boost_round=100)

# 5. Export model and vectorizer
with open('model.bin', 'wb') as f:
    pickle.dump((dv, model), f)

print("Final model saved successfully to model.bin")