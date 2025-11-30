import json
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from catboost import CatBoostRegressor

app = FastAPI(
    title="Bank Income Prediction AI",
    description="Сервис прогноза доходов (Native NaN support + Smart Features)",
    version="2.0.0"
)

# Глобальные переменные
model = None
features_config = None
explainer = None

# Список признаков, которые мы удаляли в ноутбуке (чтобы сервис тоже их игнорировал)
USELESS_FEATURES = [
    'addrref', 'city_smart_name', 'dp_ewb_last_employment_position', 
    'client_active_flag', 'vert_has_app_ru_tinkoff_investing', 
    'dp_ewb_dismissal_due_contract_violation_by_lb_cnt', 'period_last_act_ad', 
    'ovrd_sum', 'businessTelSubs', 'dp_ils_days_ip_share_5y', 
    'nonresident_flag', 'vert_has_app_ru_vtb_invest', 
    'hdb_bki_total_pil_cnt', 'accountsalary_out_flag',
    'id', 'dt', 'w', 'target'
]

@app.on_event("startup")
def load_artifacts():
    global model, features_config, explainer
    try:
        print("Loading configuration...")
        with open('features.json', 'r') as f:
            features_config = json.load(f)
            
        print("Loading CatBoost model...")
        model = CatBoostRegressor()
        model.load_model("model.cbm")
        
        print("Initializing SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        print("Service Ready!")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise e

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class FeatureImpact(BaseModel):
    feature: str
    impact: float
    description: str

class PredictionResponse(BaseModel):
    predicted_income: float
    segment: str
    shap_explanation: List[FeatureImpact]

# --- ФУНКЦИЯ ПРЕДОБРАБОТКИ (Копия логики из ноутбука) ---
def preprocess_input(input_data: Dict[str, Any], required_features: List[str]) -> pd.DataFrame:
    # 1. Создаем DataFrame
    df = pd.DataFrame([input_data])
    
    # 2. Удаляем мусор, если клиент его прислал
    cols_to_drop = [c for c in USELESS_FEATURES if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 3. Приведение типов текстовых чисел
    for col in df.select_dtypes(include='object').columns:
         try:
            temp_col = df[col].astype(str).str.replace(' ', '').str.replace(',', '.')
            df[col] = pd.to_numeric(temp_col, errors='coerce')
         except:
            pass

    # 4. Smart Features (Генерация новых признаков)
    # Флаги пропусков
    important_nans = ['salary_6to12m_avg', 'first_salary_income']
    for col in important_nans:
        # Если колонки нет в запросе, считаем, что она NaN
        if col not in df.columns:
            df[f'{col}_is_missing'] = 1
        else:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)

    # Proxy Income
    s = df.get('salary_6to12m_avg', pd.Series(0, index=df.index)).fillna(0)
    t = df.get('turn_cur_cr_avg_act_v2', pd.Series(0, index=df.index)).fillna(0)
    
    df['income_proxy_max'] = np.maximum(s, t)
    df['salary_turnover_diff'] = s - t
    # Data Completeness (сколько полей пришло не пустыми)
    df['data_completeness'] = df.notna().sum(axis=1)

    # Limit Ratio
    lim = df.get('hdb_bki_total_max_limit', pd.Series(0, index=df.index)).fillna(0)
    df['limit_to_turnover_ratio'] = lim / (t + 1.0)

    # 5. Выравнивание признаков (Feature Alignment)
    # Добавляем все колонки, которых ждет модель, но которых нет в df
    # ВНИМАНИЕ: Логика заполнения должна совпадать с ноутбуком
    
    zero_fill_keywords = ['sum', 'count', 'cnt', 'amount', 'turn', 'limit', 'outstanding', 'balance']
    
    for col in required_features:
        if col not in df.columns:
            # Логика заполнения отсутствующих полей
            if any(k in col.lower() for k in zero_fill_keywords):
                df[col] = 0.0
            elif col in features_config['cat_features']:
                df[col] = "MISSING"
            else:
                # Остальные числовые оставляем NaN (CatBoost съест)
                df[col] = np.nan

    # Оставляем только нужные и в нужном порядке
    df = df[required_features]
    
    # 6. Финальная зачистка
    # Нули
    cols_to_zero = [c for c in df.columns if any(k in c.lower() for k in zero_fill_keywords) and df[c].dtype != 'object']
    df[cols_to_zero] = df[cols_to_zero].fillna(0)
    
    # Категории
    cat_cols = [c for c in df.columns if c in features_config['cat_features']]
    df[cat_cols] = df[cat_cols].fillna("MISSING")
    for c in cat_cols:
        df[c] = df[c].astype(str)

    return df

@app.post("/predict", response_model=PredictionResponse)
def predict_income(request: PredictionRequest):
    global model, features_config, explainer
    
    if not model:
        raise HTTPException(status_code=503, detail="Model loading...")

    try:
        # Препроцессинг
        df_clean = preprocess_input(request.features, features_config['feature_names'])
        
        # Предикт (получаем логарифм)
        log_pred = model.predict(df_clean)[0]
        
        # Конвертация в рубли
        prediction = np.expm1(log_pred)
        prediction = max(0.0, float(prediction))
        
        # SHAP
        shap_values = explainer.shap_values(df_clean)
        feature_importance = list(zip(features_config['feature_names'], shap_values[0]))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanation = []
        for feat, impact in feature_importance[:5]:
            explanation.append(FeatureImpact(
                feature=feat,
                impact=round(float(impact), 4),
                description="Влияние на прогноз"
            ))
            
        segment = "Premium" if prediction > 150000 else "Middle" if prediction > 60000 else "Mass"

        return PredictionResponse(
            predicted_income=round(prediction, 2),
            segment=segment,
            shap_explanation=explanation
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))