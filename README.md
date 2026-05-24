# Player Engagement Prediction — Online Gaming Behavior

End-to-end machine learning project that predicts a player's engagement level (**Low / Medium / High**) using the public [Predict Online Gaming Behavior dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset).

The project covers data cleaning, exploratory analysis, feature engineering, two trained models compared with cross-validation, and a **FastAPI prediction service** that serves the best model over HTTP — complete with personalized retention-action recommendations.

**Domain:** Game development / player analytics  
**Task:** Multi-class classification (3 classes)  
**Dataset:** ~40,000 players × 13 columns

---

## Why this matters for game studios

Studios that can identify a player's engagement level in real time can:
- Target **Low**-engagement players with re-engagement campaigns before they churn
- Push **Medium**-engagement players over the hump with personalized content
- Reward **High**-engagement players (whales) with loyalty perks

The API returns a recommended retention action with every prediction.

---

## Project structure

```
gaming_project/
├── data/
│   └── online_gaming_behavior_dataset.csv     <- download from Kaggle (gitignored)
├── notebooks/
│   └── gaming_engagement_analysis.ipynb       <- full training pipeline
├── models/
│   ├── engagement_pipeline.pkl                <- saved by the notebook
│   ├── label_encoder.pkl                      <- maps 0/1/2 ↔️ Low/Medium/High
│   └── metadata.json                          <- features + test metrics
├── api/
│   └── app.py                                 <- FastAPI prediction service
├── docs/
│   └── HELPBOOK.md                            <- detailed walkthrough
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick start

### 1. Setup

```bash
git clone https://github.com/<your-username>/gaming-engagement-prediction.git
cd gaming-engagement-prediction
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the data

Download `online_gaming_behavior_dataset.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset) and place it in `data/`.

### 3. Train the model

```bash
jupyter notebook notebooks/gaming_engagement_analysis.ipynb
```

Run all cells. This produces:
- `models/engagement_pipeline.pkl`
- `models/label_encoder.pkl`
- `models/metadata.json`

### 4. Start the API

```bash
uvicorn api.app:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

### 5. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 28, "Gender": "Male", "Location": "USA", "GameGenre": "RPG",
    "PlayTimeHours": 18.5, "InGamePurchases": 1, "GameDifficulty": "Hard",
    "SessionsPerWeek": 14, "AvgSessionDurationMinutes": 95,
    "PlayerLevel": 78, "AchievementsUnlocked": 42
  }'
```

Response:

```json
{
  "engagement_level": "High",
  "confidence": 1.0,
  "probabilities": {"High": 1.0, "Low": 0.0, "Medium": 0.0},
  "retention_action": "Reward loyalty: exclusive content, early access, or VIP perks. These players are your best advocates — invite to beta tests."
}
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET  | `/`              | Service info |
| GET  | `/health`        | Liveness check |
| GET  | `/metadata`      | Model name, classes, feature list, test metrics |
| POST | `/predict`       | Predict for one player |
| POST | `/predict_batch` | Predict for up to 1,000 players |
| GET  | `/docs`          | Interactive Swagger UI |

---

## What the project does

| Stage | Tool / Approach |
|---|---|
| Data cleaning | Drop `PlayerID`, verify no missing values |
| EDA | KDE plots, categorical bar charts, stacked-bar engagement-by-difficulty, correlation heatmap |
| Feature engineering | `weekly_play_minutes`, `achievements_per_level`, `age_group`, `is_power_user`, `session_intensity` |
| Preprocessing | `ColumnTransformer` → median impute + scale for numbers, mode impute + one-hot for categories |
| Target encoding | `LabelEncoder` (Low/Medium/High → numeric, saved for API decoding) |
| Models trained | **Logistic Regression**, **Random Forest** |
| Selection metric | 5-fold stratified cross-validated **Macro F1** |
| Evaluation | Accuracy, macro/weighted precision/recall/F1, per-class report, confusion matrix |
| Persistence | `joblib.dump` for pipeline + label encoder |
| Serving | FastAPI with Pydantic input validation + business retention actions |

---

## Decisions explained

**Why these two models?**  
- **Logistic Regression** — a fast, interpretable linear baseline. Always train one to make sure a heavier model is actually doing better than the simplest sensible thing.
- **Random Forest** — an ensemble of decision trees that captures non-linear interactions, which matter for engagement (e.g., *high SessionsPerWeek × long AvgSessionDuration × made InGamePurchases* together predict "High" engagement much better than each feature alone).

Comparing a simple baseline against a strong non-linear model is the minimum responsible model-selection setup. If Logistic Regression wins, the relationships in the data are mostly linear and you've saved compute. If Random Forest wins, you know the non-linearities are real.

**Why Macro F1 instead of accuracy?**  
The three engagement classes are roughly balanced (~33% each). Macro F1 weights each class equally, which matches the business priority: identifying the *Low*-engagement players (who churn) and the *High*-engagement players (the revenue drivers) is at least as important as the majority Medium class.

**How is overfitting handled?**
- 80/20 stratified train/test split — test set touched only at the end
- 5-fold stratified cross-validation on the training set for model selection
- `max_depth=12` capped on the Random Forest to prevent trees from memorizing
- Preprocessing inside the pipeline so scaling/encoding never see test data

