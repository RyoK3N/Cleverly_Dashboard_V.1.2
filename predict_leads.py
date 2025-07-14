from lp_model_training import DataModule, Trainer, Predictor
import pandas as pd

best_pth = './models_v4/leadnet_best.keras'
CSV_PATH = "./sessions/processed_data/preprocessed_data.csv"
data_module = DataModule(CSV_PATH, batch_size=256)

predictor = Predictor(best_pth, data_module)


df_scheduled = pd.read_csv('./sessions/sessions/scheduled_20250714_140422.csv')
selected_columns = df_scheduled[['Owner', 'Linkedin', 'Email', 'Company', 'Interested In', 'Plan Type', 'UTM Medium', 'UTM LINK', 'origin']]

for index, row in selected_columns.iterrows():
    sample_lead_dict = row.to_dict()
    result = predictor.predict_one(sample_lead_dict)
    print(f"Lead {index + 1}: {result[0]}, {result[1]}")
