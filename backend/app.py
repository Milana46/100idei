from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Корень проекта: .../курсач
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import addRecommender

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(BASE_DIR, "saved_model_ab")
EXCEL_PATH = os.path.join(BASE_DIR, "prepared_all.xlsx")

recommender = addRecommender.Recommender(model_dir=MODEL_DIR)

# 1) пробуем загрузить модель
try:
    recommender.load()
    # важно: чтобы был список антибиотиков
    recommender.load_data(excel_path=EXCEL_PATH, sheet_name=None)
    print("Модель загружена успешно.")
except Exception as e:
    print("Не удалось загрузить модель, начинаем обучение. Причина:", e)

    # 2) если модели нет — обучаем (НО сначала грузим данные!)
    recommender.load_data(excel_path=EXCEL_PATH, sheet_name=None)
    recommender.train(epochs=15, batch_size=256, validation_split=0.15, learning_rate=2e-3)
    recommender.save()
    print("Модель обучена и сохранена.")
    
    

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    diagnosis = (data.get("diagnosis") or "").strip()
    microorganism = (data.get("microorganism") or "").strip()

    if not diagnosis or not microorganism:
        return jsonify({"error": "diagnosis and microorganism are required"}), 400

    try:
        df = recommender.recommend(diag=diagnosis, microbe=microorganism, top_k=15)
        # for col in ["P(S)", "P(I)", "P(R)", "Ожидаемый_балл"]:
        #  if col in df.columns:
        #   df[col] = df[col].astype(float).round(3)
       
        return jsonify({
            "diagnosis": diagnosis,
            "microorganism": microorganism,
            "top15": df.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
