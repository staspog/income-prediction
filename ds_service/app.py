import json
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from catboost import CatBoostRegressor

# Инициализация приложения
app = FastAPI(
    title="Bank Income Prediction AI",
    description="Сервис прогноза доходов клиентов с объяснением (SHAP) и рекомендациями.",
    version="1.0.0"
)

# Глобальные переменные для хранения модели и метаданных
model = None
features_config = None
explainer = None

@app.on_event("startup")
def load_artifacts():
    """
    Загрузка модели и конфигурации при старте контейнера.
    Это происходит один раз, чтобы запросы обрабатывались быстро.
    """
    global model, features_config, explainer
    
    try:
        print("Loading configuration...")
        with open('features.json', 'r') as f:
            features_config = json.load(f)
            
        print("Loading CatBoost model...")
        model = CatBoostRegressor()
        model.load_model("model.cbm")
        
        print("Initializing SHAP explainer...")
        # TreeExplainer оптимизирован для деревьев, работает быстро
        explainer = shap.TreeExplainer(model)
        
        print("Service is ready to accept requests!")
        
    except Exception as e:
        print(f"CRITICAL ERROR during startup: {e}")
        raise e

# --- Pydantic модели для валидации данных ---

class PredictionRequest(BaseModel):
    # Клиент может прислать неполные данные, мы сами заполним пропуски
    features: Dict[str, Any]

class FeatureImpact(BaseModel):
    feature: str
    impact: float
    description: str

class Recommendation(BaseModel):
    product_name: str
    reason: str

class PredictionResponse(BaseModel):
    predicted_income: float
    currency: str = "RUB"
    segment: str
    shap_explanation: List[FeatureImpact]
    recommendations: List[Recommendation]

# --- Основная логика ---

@app.post("/predict", response_model=PredictionResponse)
def predict_income(request: PredictionRequest):
    global model, features_config, explainer
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # 1. Подготовка DataFrame
        input_data = request.features
        df = pd.DataFrame([input_data])
        
        # Получаем список ожидаемых признаков из конфига
        required_features = features_config['feature_names']
        cat_features_list = features_config['cat_features']
        
        # 2. Выравнивание признаков (Feature Alignment)
        # Если каких-то признаков нет во входном JSON, заполняем их дефолтными значениями
        for col in required_features:
            if col not in df.columns:
                if col in cat_features_list:
                    df[col] = "MISSING"
                else:
                    df[col] = 0.0
        
        # Оставляем только нужные колонки в правильном порядке
        df = df[required_features]
        
        # 3. Приведение типов (важно для CatBoost)
        for col in cat_features_list:
            # Конвертируем все категориальные в строки, заменяем NaN
            df[col] = df[col].astype(str).replace('nan', 'MISSING').replace('None', 'MISSING')
            
        # 4. Предсказание
        # Модель возвращает логарифм дохода, так как мы учили её на log1p
        log_prediction = model.predict(df)[0]
        
        # Конвертируем обратно в рубли: exp(x) - 1
        prediction = np.expm1(log_prediction)
        prediction = max(0.0, float(prediction)) # Доход не может быть < 0
        
        # 5. Объяснение (SHAP Values)
        shap_values = explainer.shap_values(df)
        
        # Собираем топ-5 влияющих факторов
        # shap_values[0] - массив значений для первой (и единственной) строки
        feature_importance = list(zip(required_features, shap_values[0]))
        # Сортируем по абсолютному значению влияния
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanation = []
        for feat, impact in feature_importance[:5]:
            explanation.append(FeatureImpact(
                feature=feat,
                impact=round(float(impact), 4),
                description=f"Влияние признака {feat} на логарифм дохода"
            ))

        # 6. Бизнес-логика (Сегментация и рекомендации)
        recommendations = []
        segment = "Mass"
        
        if prediction > 150_000:
            segment = "Premium"
            recommendations.append(Recommendation(
                product_name="Alfa Travel Premium",
                reason="Идеально для частых путешествий с высоким доходом."
            ))
            recommendations.append(Recommendation(
                product_name="Персональный брокер",
                reason="Индивидуальные стратегии инвестирования."
            ))
        elif prediction > 60_000:
            segment = "Middle"
            recommendations.append(Recommendation(
                product_name="Кредитная карта 'Целый год без %'",
                reason="Вам доступен повышенный кредитный лимит."
            ))
        else:
            segment = "Mass"
            recommendations.append(Recommendation(
                product_name="Альфа-Карта с кэшбэком",
                reason="Бесплатное обслуживание и кэшбэк на супермаркеты."
            ))

        return PredictionResponse(
            predicted_income=round(prediction, 2),
            segment=segment,
            shap_explanation=explanation,
            recommendations=recommendations
        )

    except Exception as e:
        # Логируем ошибку и возвращаем 500
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}