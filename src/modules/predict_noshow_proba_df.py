import numpy as np
import torch
import pandas as pd

feature_cols = ['gender',
 'age',
 'under_12_years_old',
 'over_60_years_old',
 'patient_needs_companion',
 'days_since_first_visit',
 'average_temp_day',
 'max_temp_day',
 'average_rain_day',
 'max_rain_day',
 'storm_day_before',
 'appointment_date_Fri',
 'appointment_date_Mon',
 'appointment_date_Sat',
 'appointment_date_Sun',
 'appointment_date_Thu',
 'appointment_date_Tue',
 'appointment_date_Wed',
 'appointment_datetime_Jan',
 'appointment_datetime_Feb',
 'appointment_datetime_Mar',
 'appointment_datetime_Apr',
 'appointment_datetime_May',
 'appointment_datetime_Jun',
 'appointment_datetime_Jul',
 'appointment_datetime_Aug',
 'appointment_datetime_Sep',
 'appointment_datetime_Oct',
 'appointment_datetime_Nov',
 'appointment_datetime_Dec',
 'disability_intellectual',
 'disability_motor',
 'disability_null',
 'specialty_physiotherapy',
 'specialty_psychotherapy',
 'specialty_speech therapy',
 'specialty_occupational therapy',
 'specialty_unknown',
 'specialty_enf',
 'specialty_assist',
 'specialty_pedagogo',
 'specialty_sem especialidade',
 'icd_d43.4',
 'icd_f06.7',
 'icd_f68',
 'icd_f71',
 'icd_f71.2',
 'icd_f80',
 'icd_f80.8ef84',
 'icd_f80.9',
 'icd_f83',
 'icd_f83.0',
 'icd_f84',
 'icd_f84.,9',
 'icd_f84.0',
 'icd_f84.1',
 'icd_f84.5',
 'icd_f84.9',
 'icd_f84.ef91',
 'icd_f90',
 'icd_f90.',
 'icd_g09',
 'icd_g11',
 'icd_g20',
 'icd_g21.3',
 'icd_g37',
 'icd_g40',
 'icd_g40.1',
 'icd_g45',
 'icd_g45.8',
 'icd_g71.3',
 'icd_g80.9',
 'icd_g81',
 'icd_g81.1',
 'icd_g91.es06',
 'icd_g93.4',
 'icd_h90',
 'icd_i64',
 'icd_i67',
 'icd_i69',
 'icd_i69.4',
 'icd_i73',
 'icd_j06',
 'icd_k02',
 'icd_k68',
 'icd_p21',
 'icd_q05.2',
 'icd_q05.9',
 'icd_q99.9ef84.0',
 'icd_r13',
 'icd_r26',
 'icd_r46.3',
 'icd_r47.1',
 'icd_r58',
 'icd_r61.0',
 'icd_r62',
 'icd_r68',
 'icd_r68.0',
 'icd_s14',
 'icd_s24',
 'icd_s72.3',
 'icd_t00',
 'icd_t91.3',
 'icd_z00',
 'icd_z00.1',
 'icd_z11.2',
 'icd_z71.2',
 'icd_z89.4',
 'icd_null']

@torch.no_grad()
def predict_noshow_proba_df(
    model,
    scaler,
    df,
    feature_cols = feature_cols,
    target_col="no_show",
    device=None,
    batch_size=4096,   # 메모리/속도에 맞게 조절
):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    X_df = df.copy()

    # 타겟 제거
    if target_col in X_df.columns:
        X_df = X_df.drop(columns=[target_col])

    # 컬럼 정렬/누락 보정
    X_df = X_df.reindex(columns=feature_cols, fill_value=0)

    # float32 + scaler
    X = X_df.values.astype(np.float32)
    X_scaled = scaler.transform(X)

    # 배치 추론
    probs = np.empty((X_scaled.shape[0],), dtype=np.float32)

    for start in range(0, X_scaled.shape[0], batch_size):
        end = start + batch_size
        xb = torch.tensor(X_scaled[start:end], dtype=torch.float32, device=device)
        logits = model(xb).view(-1)
        probs[start:end] = torch.sigmoid(logits).detach().cpu().numpy()

    df_with_prob = df.copy()
    df_with_prob["no_show_prob"] = probs
    
    return df_with_prob