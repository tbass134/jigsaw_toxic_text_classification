import pandas as pd
import joblib
import TokenizerTransformer, DenseTransformer


sample_submission = pd.read_csv("data/sample_submission.csv")
test_df = pd.read_csv("data/test.csv")
df_test = pd.merge(test_df, sample_submission, on = "id")

model = joblib.load("model.pkl")
y_pred = model.predict(df_test["comment_text"])
print