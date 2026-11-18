# Fieldwise Crop Recommendations

Flask application for crop recommendations based on soil nutrients and weather conditions. Successful recommendations and their input values are stored in a local SQLite database.

## Run locally

```powershell
.\RUBYFATH\Scripts\python.exe -m pip install -r requirements.txt
.\RUBYFATH\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`.

The database is created at `instance/crop_recommendations.sqlite3` by default. Set `CROP_DATABASE_PATH` to use another SQLite file. Set `CROP_MODEL_PATH` to use a model artifact at another path.

## Model requirements

The model must be a trusted joblib artifact containing a fitted estimator with `predict_proba`, at least three trained classes in `classes_`, and a probability output for each class. It must accept these input features, in order: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`. The app ranks the estimator's probabilities and displays the three most likely crop classes with their percentages. It also stores the three recommendations with each successful input in SQLite.

The bundled `crop-recommendation (3).pkl` is a joblib-serialized scikit-learn Pipeline. Load it with joblib rather than Python's `pickle` module. It was serialized with scikit-learn 1.7.0; the dependency is pinned to that version for compatibility. The artifact should only be loaded if it comes from a trusted source.

## Tests

```powershell
.\RUBYFATH\Scripts\python.exe -m unittest discover -s tests -v
```
