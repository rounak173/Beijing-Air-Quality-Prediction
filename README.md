# Beijing Air Quality Prediction

Predicting hourly concentrations of **PM10, SO₂, and NO₂** in Beijing from meteorological data, using an **LSTM** (primary) and **XGBRegressor** (alternative) model.

Project for the "Machine Learning for Physicists" module, TU Dortmund University.

## Overview

Air pollution contributes to roughly 8 million premature deaths a year (WHO), and 99% of the world's population breathes air that doesn't meet WHO standards. This project predicts hourly PM10, SO₂, and NO₂ levels in Beijing from historical pollutant and meteorological data — most prior work on this dataset targets PM2.5 instead. Two models were built and compared: an **LSTM** to capture temporal dependencies, and an **XGBRegressor** as a tabular, feature-based alternative.

## Built With

![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

![Pandas](https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-0092CA?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

![Git](https://img.shields.io/badge/Git-F05032.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=github&logoColor=white)

## Dataset

- **Source:** [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) · **License:** CC BY 4.0
- Hourly data of 6 air pollutants + 6 meteorological variables from 12 monitoring sites — **420,768 rows × 17 columns**, March 2013 – Feb 2017.
- **Key features:** `day`, `hour`, `TEMP`, `PRES`, `DEWP`, `station`, `WSPM` · **Targets:** `PM10`, `SO2`, `NO2`

Full feature descriptions and missing-value counts are in `ML - Report.pdf` (Table 2.1).

## Repository Structure

```
Beijing-Air-Quality-Prediction/
├── Data/                      # Raw station-level CSVs (UCI dataset)
├── ML.ipynb                   # Preprocessing, LSTM & XGBRegressor training/evaluation
├── Initial Presentation.pdf   # Early-stage project proposal slides
├── Final Presentation.pdf     # Final results presentation slides
├── ML - Report.pdf            # Full written report
└── README.md
```

## Approach

**Preprocessing:** rolling-mean imputation for pollutants, datetime interpolation for weather features, cyclical (sin/cos) encoding for month/day/hour, `log1p` + `RobustScaler` for skewed features, `StandardScaler` for the rest, and `log1p` + `MinMaxScaler` for LSTM targets.

**LSTM (primary):** a separate 2-layer LSTM per pollutant, trained on 24-hour input sequences to predict the next hour's concentration, tuned via `RandomizedSearchCV`.

**XGBRegressor (alternative):** chosen for its robustness to noisy/missing tabular data and non-temporal drivers (temperature, pressure, wind, station), tuned via `GridSearchCV`.

Full architectures and hyperparameter tables are in `ML - Report.pdf` (Sections 3.1–3.2).

## Results

| Pollutant | LSTM MAE | LSTM RMSE | LSTM R² | XGB MAE | XGB RMSE | XGB R² |
|---|---|---|---|---|---|---|
| PM10 | 0.36 | 0.49 | 0.75 | 15.67 | 3.97 | 0.91 |
| SO2  | 0.46 | 0.61 | 0.56 | 3.99  | 2.00 | 0.89 |
| NO2  | 0.44 | 0.55 | 0.57 | 6.77  | 2.60 | 0.93 |

*LSTM errors are on scaled targets, XGBRegressor's are on the raw scale — R² is the fairer cross-model comparison. XGBRegressor scored higher R² on all three pollutants and tracked spikes/dips more sharply; LSTM captured general trends but smoothed over sharp peaks.*

## Key Takeaways

- XGBRegressor outperformed LSTM on R² across all three pollutants and was more responsive to short-term fluctuations.
- LSTM is theoretically well-suited to this sequential task but is more sensitive to noisy, spike-heavy signals.
- Model choice here depends as much on data quality/structure as on the temporal nature of the task.

## Future Work

- Add traffic, industrial-activity, or satellite-based features
- Try longer input windows (48–72h), Bidirectional LSTM/GRU, or attention mechanisms
- Ensemble LSTM (temporal) with XGBoost (tabular) predictions

## Authors

- **Md Rounak Jahan Raj** — rounak.raj@tu-dortmund.de

TU Dortmund University, IMAPP · Submitted to Dr. Carsten Burgard and Dr. Cornelius Grunwald, July 2025

## License

Dataset licensed under CC BY 4.0.
