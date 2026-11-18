import json
import logging
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, current_app, render_template, request


FEATURES = {
    "N": (0, 140),
    "P": (5, 145),
    "K": (5, 205),
    "temperature": (0, 44),
    "humidity": (14, 99),
    "ph": (3.5, 9.9),
    "rainfall": (20, 298),
}
FEATURE_NAMES = list(FEATURES)
DEFAULT_MODEL_FILENAME = "crop-recommendation (3).pkl"


def load_model(model_path):
    try:
        model = joblib.load(model_path)
    except (OSError, EOFError, AttributeError, ImportError, ValueError) as error:
        logging.getLogger(__name__).error("Could not load crop recommendation model: %s", error)
        return None

    if not callable(getattr(model, "predict_proba", None)):
        logging.getLogger(__name__).error(
            "The configured crop recommendation artifact has no callable predict_proba method."
        )
        return None

    classes = getattr(model, "classes_", None)
    if classes is None or np.asarray(classes).ndim != 1 or len(classes) < 3:
        logging.getLogger(__name__).error(
            "The configured crop recommendation model needs at least three trained classes."
        )
        return None

    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is not None and feature_count != len(FEATURE_NAMES):
        logging.getLogger(__name__).error(
            "The crop recommendation model expects %s features; expected %s.",
            feature_count,
            len(FEATURE_NAMES),
        )
        return None

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != FEATURE_NAMES:
        logging.getLogger(__name__).error(
            "The crop recommendation model feature names/order do not match the input form."
        )
        return None

    return model


def connect_database():
    connection = sqlite3.connect(current_app.config["DATABASE"])
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(database_path):
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nitrogen REAL NOT NULL,
                phosphorus REAL NOT NULL,
                potassium REAL NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                ph REAL NOT NULL,
                rainfall REAL NOT NULL,
                crop TEXT NOT NULL,
                confidence REAL,
                top_three TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(predictions)")
        }
        if "top_three" not in columns:
            connection.execute(
                "ALTER TABLE predictions ADD COLUMN top_three TEXT NOT NULL DEFAULT '[]'"
            )


def recent_predictions(limit=8):
    with closing(connect_database()) as connection:
        rows = connection.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    predictions = []
    for row in rows:
        prediction = dict(row)
        prediction["top_three"] = json.loads(prediction["top_three"])
        predictions.append(prediction)
    return predictions


def render_home(
    *,
    form_data=None,
    error_message=None,
    field_errors=None,
    result=None,
    status=200,
):
    model_ready = current_app.config["CROP_MODEL"] is not None
    return (
        render_template(
            "index.html",
            form_data=form_data or {},
            error_message=error_message,
            field_errors=field_errors or {},
            result=result,
            history=recent_predictions(),
            model_ready=model_ready,
            model_message=(
                "Model ready — top recommendations are ranked by the trained estimator."
                if model_ready
                else "Prediction is unavailable: add a compatible trained model with class probabilities."
            ),
        ),
        status,
    )


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_mapping(
        DATABASE=os.environ.get(
            "CROP_DATABASE_PATH",
            str(Path(app.instance_path) / "crop_recommendations.sqlite3"),
        ),
        MODEL_PATH=os.environ.get(
            "CROP_MODEL_PATH",
            str(Path(app.root_path) / DEFAULT_MODEL_FILENAME),
        ),
    )
    if test_config:
        app.config.update(test_config)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    initialize_database(app.config["DATABASE"])
    app.config["CROP_MODEL"] = load_model(app.config["MODEL_PATH"])

    @app.get("/")
    def index():
        return render_home()

    @app.post("/predict")
    def predict_crop():
        form_data = {name: request.form.get(name, "").strip() for name in FEATURE_NAMES}
        values = {}
        field_errors = {}

        for name, (minimum, maximum) in FEATURES.items():
            raw_value = form_data[name]
            try:
                value = float(raw_value)
            except ValueError:
                field_errors[name] = "Enter a valid number."
                continue

            if not math.isfinite(value):
                field_errors[name] = "Enter a finite number."
            elif not minimum <= value <= maximum:
                field_errors[name] = f"Enter a value from {minimum:g} to {maximum:g}."
            else:
                values[name] = value

        if field_errors:
            return render_home(
                form_data=form_data,
                error_message="Check the highlighted values and try again.",
                field_errors=field_errors,
                status=400,
            )

        model = current_app.config["CROP_MODEL"]
        if model is None:
            return render_home(
                form_data=form_data,
                error_message="No prediction was made. Add a compatible trained model to enable recommendations.",
                status=503,
            )

        input_frame = pd.DataFrame([values], columns=FEATURE_NAMES)
        try:
            probability_rows = np.asarray(model.predict_proba(input_frame), dtype=float)
            classes = np.asarray(model.classes_)
            if probability_rows.shape != (1, len(classes)):
                raise ValueError("Model probability output does not match its classes.")
            probabilities = probability_rows[0]
            if (
                not np.all(np.isfinite(probabilities))
                or not np.all((probabilities >= 0) & (probabilities <= 1))
                or not np.isclose(probabilities.sum(), 1.0, rtol=1e-4, atol=1e-6)
            ):
                raise ValueError("Model returned invalid class probabilities.")
            ranked_indices = np.argsort(-probabilities, kind="stable")[:3]
            recommendations = [
                {
                    "crop": str(classes[index]),
                    "probability": float(probabilities[index]),
                }
                for index in ranked_indices
            ]
            crop_name = recommendations[0]["crop"]
            confidence = recommendations[0]["probability"]
        except (ValueError, TypeError, AttributeError, IndexError, RuntimeError) as error:
            current_app.logger.exception("Crop model prediction failed: %s", error)
            return render_home(
                form_data=form_data,
                error_message="The model could not process these inputs. Check the model file and try again.",
                status=500,
            )

        with closing(connect_database()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO predictions
                        (nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall,
                         crop, confidence, top_three)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        values["N"],
                        values["P"],
                        values["K"],
                        values["temperature"],
                        values["humidity"],
                        values["ph"],
                        values["rainfall"],
                        crop_name,
                        confidence,
                        json.dumps(recommendations, separators=(",", ":")),
                    ),
                )

        result = {"crop": crop_name, "confidence": confidence, "recommendations": recommendations}
        return render_home(form_data=form_data, result=result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="127.0.0.1")
