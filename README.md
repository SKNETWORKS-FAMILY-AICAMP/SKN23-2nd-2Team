## 📌 프로젝트 명  

#### 노쇼

<p align="left" style="display: flex; align-items: center;">
  <img src="image/스포티파이.svg" width="150" />
  <img src="image/team.png" width="200" style="margin-left: 15px;" />
</p>
      
![header](https://capsule-render.vercel.app/api?type=waving&color=0db954&height=180&text=Spotify%20Churn%20Prediction%20App&fontSize=50&fontColor=ffffff)

---

## 📅 프로젝트 기간  
#### **2026. 01. 02. ~ 01. 15 (총 14일)**

---

## 👥 프로젝트 팀 및 역할  

#### 팀명: Team ㅇㅇ

| 이름       | 역할                                                  | Git                                                                                                                                                  |
| ---------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **정유선** | 팀장, PM, 화면설계 \| Streamlit Frontend + Integrator | <a href="https://github.com/sbpark2930-ui"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/></a> |
| **정희영** | DB 설계 및 관리                                       | <a href="https://github.com/sjy361872"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/></a>     |
| **정용희** |                                                       | <a href="https://github.com/ykgstar37-lab"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/></a> |
| **유헌상** |                                                       | <a href="https://github.com/wjdtpdus25"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/></a>    |
| **김도영** | 발표                                                  | <a href="https://github.com/silentkit12"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/></a>   |

---

## 🚗 프로젝트 주제  
**의료 기반 데이터셋 수집 후 ML/DL을 통한 모델 기반 진료부결(이하 노쇼) 예측 및 솔루션


**

---


## 📝 개요

<p align="center">
  <img src="image/spotify_naver.png" alt="네이버-스포티파이기사" width="45%">
  <img src="image/spotify_naver2.png" alt="사진" width="37%">
</p>


· 문제 정의 단계 : 분석하고자 하는 분야를 이해하고, 해결해야 할 문제를 객관적이고 구체적으로 정의한다. · 데이터 수집 단계 : 분석에 필요한 데이터 요건을 정의하고, 데이터를 확보한다. • 데이터 전처리 단계 : 수집한 데이터에 존재하는 결측값이나 오류를 수정/보완한다. 경우에 따라서 데이터 구조나 특성을 변경한다. • 데이터 모델링 단계: 하나의 테이블(데이터셋)이 아닌 다수의 테이블을 이용하여 분석을 하는 경우가 있다. 이러한 경우, 데이터 모델링이 필요하다. • 시각화 및 탐색 단계 : 다양한 도구를 이용하여 데이터를 시각화하고, 탐색을 통하여 문제를 해결한다.

헬스케어 ERP 시장의 확대 https://www.metatechinsights.com/ko/industry-insights/healthcare-erp-market-2674
노쇼 심각 https://www.whosaeng.com/155771
병원 예약부도(No-show) 감소를 위한 예약관리 방안 https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201708034065753 노쇼 문자보다 전화가 유용

<br>

본 프로젝트는 제한된 데이터를 기반으로, 사용자 행동 데이터를 분석하여 **이탈 가능성**을 예측한 후 해당 데이터를 통한 노쇼 리스크로서 병원 경영에 유의미한 솔루션을 제공하는 것을 목표로 하였다. 

- ** 기상 정보와 병명 서비스 시작일을 기반으로 진료계약 부결률을 낮춤과 동시에 단순 문자 발송 시스템이 아닌 예측형 모델로서 ERP 혹은 CRM, ACRM적 분석을 할 수 있는 것을 목표**
- **📊 사용자 행동 기반 부결 위험도 분석 및 시각화 대시보드 제공**
- **👩‍💻 관리자가 전체적인 부결률을 확인 할 수 있는 관리 시스템 구현**
- **🔍 다양한 모델을 기반을 최적의 모델을 찾아 적용**

<br>

<h2>👤❓ WBS (업무 분담 체계)</h2>
<p align="center">
  <img src="image/역할분담.png" alt="역할분담" width="850">
</p>

---

## 🛠 기술스택  

### 🎛 Backend (API 서버)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![PyMySQL](https://img.shields.io/badge/PyMySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![bcrypt](https://img.shields.io/badge/bcrypt-3388FF?style=for-the-badge&logo=security&logoColor=white)

### 📊 Data Processing & Analysis
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

### 📈 Data Visualization
![matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

### 🖥️ Frontend (Streamlit)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### ⚙️ Dev Environment
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![VSCode](https://img.shields.io/badge/VSCode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)

```plaintext### 폴더 및 파일 구조

your_project_name/
├── .venv/                         # 가상 환경 (git X)
├── assets                         # 이미지, 데이터 파일등을 위한 폴더
│   ├── images/                    # 이미지 폴더
│   ├── fonts/                     # font용 woff2 파일이 저장되어 있는 폴더
│		├── no_show_feature_correlations.csv
│		├── noshow_weather.csv         # weather 테이블 입력용 csv
│		├── noshow_appt.csv            # appointment 테이블 입력용 csv
│   └── medical-appointments-no-show-en.csv # 원본 데이터
│   
│── api/
│   ├── weather_api.py             # 날씨 정보 API
│   └── weather_week.py            # 날씨 정보 DB에 반영
│── artifacts/                     # 모델 및 학습 데이터 정보
│   ├── feature_columns.json       # 학습에 사용한 피처 컬럼 이름
│   ├── lg_feature_columns.json    # 로지스틱 피쳐 컬럼
│   ├── lg_model.joblib            # 로지스틱 모델 평가
│   ├── lgbm_feature_columns.json  # lightGBM 피쳐 컬럼
│   ├── lgbm_model.joblib          # lightGBM 모델 평가
│   ├── mlp_model.pt               # 김도영 딥러닝 모델 정보
│   ├── scaler.joblib              # 데이터 스케일러 정보
│   ├── rf_feature_columns.json    # 랜덤포레스트 피쳐 컬럼
│   ├── rf_metrics.json            # 랜덤포레스트 평가
│   ├── rf_model.joblib            # 랜덤포레스트 모델 정보
│   ├── rf_threshold.json          # 랜덤포레스트 판단 값
│   ├── xgb_feature_columns.json   # XBBoost 피쳐 컬럼
│   ├── xgb_metrics.json           # XBBoost 평가
│   ├── xgb_model.joblib           # XBBoost 모델 저정보
│   └── xgb_threshold.json         # XBBoost 판단 값
│── services/
│   └── customerService.py         # 스트림릿 모델 불러오기 및 확률 계산 함수
│
├── src/
│   ├── modules/                   # 모듈 파일들로 구성된 폴더
│   │   ├── connect_db_module.py    # DB 연결 모듈
│   │   ├── machine_module.py       # 머신러닝 모듈
│   │   ├── one_hot_module.py       # 컬럼 원핫인코딩용 모듈
│   │   ├── predict_noshow_logistic.py    # 로지스틱 모델 확률 예측 모듈
│   │   ├── predict_noshow_lightgbm.py    # lightGBM 모델 확률 예측 모듈
│   │   ├── DL_KDY.py               # 김도영 딥러닝 학습 파일
│   │   ├── xgboost.py              # xgboost 머신러닝
│   │   └── randomForest.py         # 랜덤포레스트 머신러닝
│   └── views/                     # 화면 페이지 정의 폴더
│       ├── CustomerList.py        # 고객 관리 페이지
│       ├── Dashboard.py           # 대시보드 페이지
│       ├── MoedelAnalytics.py     # 모델 성능 확인 페이지
│       ├── .streamlit/
│       │   └── config.toml        # streamlit 테마 설정 파일 
│       ├── modals/                
│       │   ├── editInfoModal.py   # 메세지 전송 모덜
│       │   ├── messageModal.py    # 메세지 전송 모덜
│       │   └── weatherModal.py    # 날씨 유형별 노쇼 예측 비율 모덜
│       └── tabs/
│           ├── deepTap.py         # 딥러닝 모델 성능 확인 탭
│           └── machineTap.py      # 머신러닝 모델 성능 확인 탭
├── .env                           # (git X) 환경변수 파일
├── .gitignore
├── Main.py                        # 앱의 메인 시작 파일
├── README.md
├── requirements.txt               # 프로젝트 의존성 목록
└── user_flow.drawio               # 화면흐름도 drawio 파일
```
<br>


## 📊 데이터 및 모델링 개요 (Data & Modeling)

### 1) Data Setup
- **데이터 구조**: 49593회의 환자의 간단한 진료 정보 및 날짜 데이터가 보관됨.
- **Feature Exploration**:
- specialty (진료과)

appointment_time (예약 시간)

gender (성별)

appointment_date (예약 날짜)

no_show (노쇼 여부 - Target)

no_show_reason (노쇼 사유)

disability (장애 유형)

date_of_birth (생년월일)

entry_service_date (최초 내원일)

city (거주 도시)

icd (질병 코드)

appointment_month (예약 월)

appointment_year (예약 년도)

appointment_shift (진료 시간대 - 오전/오후 등)

age (연령)

under_12_years_old (12세 미만 여부)

over_60_years_old (60세 이상 여부)

patient_needs_companion (보호자 동반 필요 여부)

average_temp_day (당일 평균 기온)

average_rain_day (당일 평균 강수량)

max_temp_day (당일 최고 기온)

max_rain_day (당일 최고 강수량)

rainy_day_before (전날 강우 여부)

storm_day_before (전날 폭풍/강풍 여부)

rain_intensity (강수 강도)

heat_intensity (더위 강도)

1. 원핫인코딩은 별도 모듈로 진행할 예정이므로 해당하는 컬럼은 원상태 유지
2. 결측치 자체가 의미있는 컬럼은 결측치 유지
3. no_show_reason 컬럼 삭제
4. 상기 삭제 제외 컬럼 수 유지
5. 날짜 / 시간 컬럼은 표준 문자열 포맷으로 변환
6. 그 외에는 타입 기반을 채움
7. 문자열 표준화
8. 숫자형 데이터의 결측치는 중간값, 문자형 데이터의 결측치는 unknown으로 채움
9. 나이, 월 >> 정수형으로 바꾸기
10. 성별 i 제거하고 gender, appointment_shift 컬럼 0 / 1로 맵핑
11. appointment_datetime, weather_date 이라는 새 컬럼 만들기

### 2) Model Selection (Why HGB?)
| Model | F1 Score   | AUC    | 설명                                                           |
| ----- | ---------- | ------ | -------------------------------------------------------------- |
| ****  | **0.6427** | 0.8093 | Precision/Recall 균형이 가장 우수하며, 서비스 운영에 최적화됨. |
| LGBM  | 0.6414     | 0.8158 | 성능은 유사하나 HGB가 더 안정적인 예측 패턴을 보임.            |

HGB(Histogram-based Gradient Boosting) 모델을 최종 선정하여 **메인 이탈 예측 엔진**으로 탑재하였습니다.

머신러닝 (할당)

1. Logistic Regression (유헌상)
2. RandomForest (정희영)
3. XGBoost (정희영)
4. LightGBM (유헌상)

딥러닝

tapnet		

ft transformer	

DNN	

MLP	

XGB	

CGB
---

## 💻 개발 화면 (Streamlit Application Features)

### 👩‍💻 관리자 페이지

#### 🏠 대시보드 및 사용자 관리
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="image/gif/01_.gif" width="48%" />
  <img src="image/gif/04_.gif" width="48%" />
</div>
> 전체 사용자 통계, 이탈 위험도 모니터링, 데이터셋 관리 기능을 제공합니다.

#### 🎯 이탈 예측 (개별/배치)
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="image/gif/06_.gif" width="48%" />
  <img src="image/gif/07_.gif" width="48%" />
</div>
> 특정 사용자의 이탈 확률을 실시간으로 예측하거나, CSV 업로드를 통해 대량 예측을 수행할 수 있습니다.

### 🙋‍♂️ 사용자 페이지

#### 🔐 로그인 및 내 정보
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="image/gif/11_.gif" width="48%" />
  <img src="image/gif/33_.gif" width="48%" />
</div>

#### 🏆 도전과제 및 음악 플레이어
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="image/gif/66_.gif" width="48%" />
  <img src="image/gif/22_.gif" width="48%" />
</div>
> 개인화된 도전과제를 수행하고 보상을 받으며, Spotify API와 연동된 음악 스트리밍을 즐길 수 있습니다.

---

## 📄 참고 자료

- **데이터 원본**: [Cervical Cancer Risk Classification Dataset (Mendeley Data)](https://data.mendeley.com/datasets/wm6w2fvkfj/1?utm_source=chatgpt.com)
- **데이터 산업 동향**: [데이터 안심구역 및 산업 활성화 기본 계획 자료](https://www.data1window.kr/dbPcyEvlSys/detail?pstSn=74)
- **시장 분석**: [글로벌 헬스케어 ERP 시장 규모 및 확대 전망](https://www.metatechinsights.com/ko/industry-insights/healthcare-erp-market-2674)
- **사회 현상 분석**: [병원 내 예약부도(No-show) 발생 현황 및 심각성](https://www.whosaeng.com/155771)
- **기술적 대안 연구**: [예약부도 감소를 위한 예약관리 방안 연구 - 전화 통화의 유용성 분석](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201708034065753)
- **비용 최적화 설계**: [AI Voice Assistant 서비스(Vapi) 요금 체계 및 단가](https://vapi.ai/pricing)
- **인프라 확보 공고**: [과학기술정보통신부 GPU 컴퓨팅 자원 지원 사업 안내](https://www.msit.go.kr/bbs/view.do?sCode=user&mId=311&mPid=121&pageIndex=&bbsSeqNo=100&nttSeqNo=3186359&searchOpt=ALL&searchTxt=)
- **인력 운용 가이드**: [외국인 환자 유치를 위한 의료 코디네이터 직무 정리](https://blog.naver.com/78dydxo/223382302014)
- **채용 비용 산정**: [의료 코디네이터 구직 현황 및 잡코리아 채용 공고 분석](https://www.jobkorea.co.kr/Recruit/GI_Read/48372827?Oem_Code=C1&logpath=1&stext=%EC%99%B8%EA%B5%AD%EC%9D%B8+%EC%BD%94%EB%94%94%EB%84%A4%EC%9D%B4%ED%84%B0&listno=1&sc=551)
- **통계 데이터**: [피부과 등 진료 과목별 노쇼율 현황 기사](https://v.daum.net/v/kYlXwbNbVc)
- **예약 부도 방지 사례**: [인도 철도청의 예약 취소 수수료 및 오버부킹 정책(WL)](https://cpro95.tistory.com/1496)
- **기관 정보**: [SAS CER - 재활 전문 센터 운영 및 서비스 개요](https://www.sas-seconci.org.br/cer)

## 💬 팀원별 회고

| 🧑💼 이름    | 🛠 역할 | 💬 소감                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **정유선** |        | 주제 선정에 있어, 괜찮은 주제라고 선정했으나, DataSet 에 대한 이해도가 부족해, 결국 Data를 합성 하여 사용하게 된게 상당히 아쉽습니다. 좀 더 세심하게 업무 리딩을 했다면 이러한 사항이 없었을거 같은 데, 그럼에도 다들 맡은 바를 열심히 해주셔서 무사히 마무리 된거같습니다.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **정희영** |        | 작업을 하면서 느낀 건, 모델 성능을 끌어올리는 것도 중요하지만 **“데이터가 어떤 이야기를 하고 있는지 이해하고, 그 데이터를 믿고 다시 쓸 수 있게 만드는 구조”**가 훨씬 더 큰 기반이 된다는 사실이었습니다. 처음엔 같은 베이스라인 데이터에서 모델과 파라미터만 계속 바꿔봤는데, 기대만큼 변화가 없었습니다. 하지만 사용자 행동 로그, 시계열 흐름, 고객 접점 피처를 직접 설계하고, 합성 데이터를 통해 먼저 방향성을 검증하기 시작하면서, 무엇이 의미 있는 피처인지, 모델이 어디까지 해낼 수 있는지가 훨씬 또렷하게 보였습니다. 그 순간부터 실험은 숫자가 아니라 근거와 맥락을 갖기 시작했습니다. 또한 preprocessing_pipeline.py → train_experiments.py → inference.py → app.py 로 흐름을 정리하며, 팀원 누구나 이해하고 다시 학습하고 그대로 API에 붙여 쓸 수 있는 형태로 만드는 데 집중했습니다. 한 번 돌려본 코드가 아니라, 재현되고 확장되는 코드가 되도록 묶어둔 이 구조는 개인적으로도 큰 성취였고, 그 점에서는 스스로 꽤 만족하고 있습니다. 물론 한편으로 아쉬움도 남습니다. 이번에는 실제 운영 로그가 아닌 합성 피처 기반으로 실험을 진행했기에, 이탈(churn) 패턴의 방향성은 확인했지만, 실제 운영 환경에서의 성능과 안정성 검증은 아직 더 필요합니다. 그래서 앞으로는 이번에 설계한 피처들과 데이터 스키마를 실제 로그 형태에 최대한 맞춰 정렬하고, 운영 데이터 기준으로 End-to-End(전처리 → 학습 → 평가) 재검증 단계를 꼭 밟아보고 싶습니다. 이번 프로젝트는 끝이 아니라, 언젠가 실제 데이터를 올려도 흔들리지 않을 구조를 만든 출발점이라고 생각하며 마무리했습니다. 그 구조 위에 다음에는 진짜 로그가 차곡차곡 쌓이기를 기대하고 있습니다. |
| **정용희** |        | 데이터의 중요성을 느낄 수 있는 프로젝트였던거 같습니다. 특히 FE 생성과 삭제.. 데이터 병합 등 주요 작업 과정에서 많은 시행 착오가 있었지만 그 과정 속에서 문제 해결 방식을 알게 되었습니다! 프로젝트 덕분에 다양한 모델을 접할 수 있어 좋았고 팀원들의 적극적인 참여와 협업 덕분에 더 큰 시너지를 만들 수 있었던 뜻깊은 경험이었습니다! 즐거웠습니다 !!                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **유헌상** |        | 여러 사정으로 사실상 첫 프로젝트나 다름없어 걱정이 많았는데, 좋은 팀원분들을 만나 무사히 마무리할 수 있었습니다. 예상치 못한 문제들도 함께 해결해 나가며 많은 것을 배울 수 있어 좋았습니다!!                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **김도영** |        | 이번 프로젝트를 통해서 머신러닝에 대해 보다 깊이있게 이해할 수 있었습니다. 정확히는 데이터 전처리한 내용을 파이프라인으로 엮는 업무를 진행했는데, 이를 위해 앞서 분석한 내용을 해석하고 필요한 데이터를 추리는 것, 머신러닝에 활용할 수 있는 데이터로 가공하는 방법을 구체적으로 실행해 보면서 수업시간에 배운 내용을 제대로 이해할 수 있는데 도움이 되었습니다. 프로젝트를 진행하며 특히 지난 번 프로젝트 때, 프론트엔드를 맡았기 때문에 비교가 많이 되었었는데. 스트림릿으로 이런 페이지까지 구현을 할 수 있다는 것을 알게 되는 기회가 되었습니다. 담당 팀원이 만들어내는 페이지를 보며 실제 서비스에 활용할만한 방안을 팀원들과 함께 구상하면서 즐거웠습니다. 앞으로 남은 프로젝트에서도 많이 보고 배워갈 수 있도록 하겠습니다.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
