# Spotify Track Popularity Prediction

A supervised machine learning project that predicts a track's Spotify popularity score (0–100) from its audio features and metadata, comparing five regression models on 114,000 tracks.

Built as the team capstone project for the IIT Delhi Online Certificate Programme in Data Science & Machine Learning.

## Dataset

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle): 114,000 tracks and 20 columns, covering 114 genres with 1,000 tracks each.

- **Target:** `popularity` (0–100)
- **Audio features:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode, time signature
- **Metadata:** duration, explicit flag, artists, genre

## Approach

1. **Exploratory analysis:** popularity distribution, correlation of audio features with popularity, and mean popularity by genre.
2. **Cleaning and feature engineering:** dropped 1 row with missing text fields, converted `explicit` to 0/1, and created `duration_min` and `num_artists` (from the `;`-separated artist list).
3. **Preprocessing:** 15 numeric features passed through as they are; genre one-hot encoded in a scikit-learn `ColumnTransformer`.
4. **Split:** 80/20 train/test split (91,199 / 22,800 tracks), random state 42.
5. **Models:** Ridge Regression, Random Forest, HistGradientBoosting, XGBoost and LightGBM, each in a scikit-learn `Pipeline`.
6. **Evaluation:** RMSE, MAE and R² on the held-out test set, plus 5-fold cross-validated RMSE on the training set.

## Results

| Model | Test RMSE | Test MAE | Test R² | CV RMSE (5-fold) |
|---|---|---|---|---|
| **Random Forest** | **15.25** | **10.27** | **0.531** | **15.78** |
| XGBoost | 17.85 | 13.08 | 0.358 | 18.00 |
| LightGBM | 18.08 | 13.35 | 0.341 | 18.17 |
| HistGradientBoosting | 18.12 | 13.38 | 0.338 | 18.18 |
| Ridge Regression | 19.14 | 14.09 | 0.262 | 19.24 |

**Best model: Random Forest.** On average, its predictions are about 10 points off on the 0–100 popularity scale.

## Key insights

- **Audio features alone barely track popularity.** Every audio feature's correlation with popularity is below 0.1 in absolute terms; the strongest are instrumentalness (−0.095) and loudness (+0.050).
- **Genre carries much of the signal.** Mean popularity ranges from 2.2 (iranian) to 59.3 (pop-film).
- **Non-linear models help.** Moving from Ridge (R² 0.26) to Random Forest (R² 0.53) roughly doubles the variance explained, which suggests that features interact rather than act independently.
- **About half the variance is still unexplained.** Popularity also depends on factors this dataset doesn't capture, such as artist fame, marketing and release timing.

## Limitations and next steps

- **Possible duplicate tracks across splits.** The dataset lists some tracks under more than one genre, and the split was not deduplicated by `track_id`. A track could therefore appear in both training and test sets, which may flatter models like Random Forest. Next step: deduplicate or group by `track_id` before splitting.
- **Boosting models were not tuned.** They used fixed settings (depth 8, 200 estimators), so the ranking reflects these configurations, not each model's best performance. Next step: hyperparameter tuning.
- **No artist-level features.** Adding artist popularity or follower counts would likely explain more of the variance.

## How to run

1. Install the dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`.
2. Download the dataset CSV and set `DATA_PATH` in the notebook to its location.
3. Run all cells.

## My contribution

This was a team project. I worked on data cleaning and preprocessing, exploratory analysis and feature engineering, and contributed to the model comparison.

## Author

**Dr. Dhruv Gupta**, BDS, clinician working at the intersection of clinical care, healthcare operations and data
[LinkedIn](https://www.linkedin.com/in/dr-dhruv-gupta)
