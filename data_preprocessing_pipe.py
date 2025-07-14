from typing import Tuple
import pandas as pd
from sklearn.utils import resample
import os 


class DataPreprocessor:
    UNNEEDED = {
        "Follow Up Tracker", "Notes", "Email Template #1", "Email Template #2",
        "Send Panda Doc?", "UTM Source", "UTM Campaign", "UTM Content",
        "Lead Source", "Channel FOR FUNNEL METRICS", "Subitems",
        "text_mkr7frmd", "Item ID", "Phone", "Last updated", "Sales Call Date",
        "Deal Status Date", "Date Created", "Item Name"
    }
    HIGH_NULL = 0.50

    def load_and_clean(self, won_csv: str, lost_csv: str) -> pd.DataFrame:
        df = pd.concat(
            [pd.read_csv(won_csv), pd.read_csv(lost_csv)], ignore_index=True)
        return self._clean(df)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        hi_null_cols = df.columns[df.isna().mean() > self.HIGH_NULL]
        df = df.drop(columns=hi_null_cols, errors="ignore")
        df = df[df["Owner"].notna()]
        df = df.drop(columns=[c for c in self.UNNEEDED if c in df.columns],
                     errors="ignore")
        return df

    def remove_null_utm_links(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df['UTM LINK'].notna()]

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        won_data = df[df['Group'] == 'won']
        lost_data = df[df['Group'] == 'lost']

        lost_downsampled = resample(lost_data, 
                                   replace=False,
                                   n_samples=len(won_data),
                                   random_state=42)

        return pd.concat([won_data, lost_downsampled])

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Linkedin'] = df['Linkedin'].fillna('Unknown')
        df['Company'] = df['Company'].fillna('Unknown')
        df['UTM Medium'] = df['UTM Medium'].fillna('Unknown')
        return df

    def preprocess_pipeline(self, won_csv: str, lost_csv: str) -> pd.DataFrame:
        df = self.load_and_clean(won_csv, lost_csv)
        print(f"After initial cleaning: {df.shape}")

        df = self.remove_null_utm_links(df)
        print(f"After removing null UTM links: {df.shape}")

        df = self.balance_dataset(df)
        print(f"After balancing dataset: {df.shape}")
        print(f"Class distribution:\n{df['Group'].value_counts()}")

        df = self.fill_missing_values(df)
        print(f"Missing values after filling:\n{df.isna().sum()}")

        return df

    def split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df[~df["Deal Status"].isin(["Scheduled", "Re-Schedule"])]
        return df.drop(columns=["Deal Status"]), df["Deal Status"]


if __name__ == "__main__":
    prep = DataPreprocessor()
    df_processed = prep.preprocess_pipeline('sessions/sessions/won_20250714_140422.csv', 'sessions/sessions/lost_20250714_140422.csv')

    save_path= './sessions/processed_data/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    filename = 'preprocessed_data.csv'
    df_processed.to_csv(os.path.join(save_path, filename), index=False)
    print(f'Preprocessed data saved to {save_path}')
