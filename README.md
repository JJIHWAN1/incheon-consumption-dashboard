# Incheon Consumption Dashboard (Demo)

인천 구별 소비 패턴을 **구 선택 → 시각화(결제액/결제건수) → 한 줄 요약**으로 보여주는 Streamlit 데모 대시보드입니다.

현재는 실제 데이터가 없는 상태를 가정해 **더미(합성) 데이터**로 동작합니다.
또한 **실시간(모의) 소비 API**를 `TTL=300s(5분)` 캐시로 흉내 내서, 5분마다 값이 갱신되는 것처럼 보이도록 구성했습니다.


```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 기능

- 구 선택(인천 기본 리스트)
- 기간 선택
- 차트
  - 시간대별 평균 결제액/결제건수
  - 요일별 평균 결제액/결제건수
  - 일자별 결제액/결제건수 추이
- 핵심 요약(규칙 기반 문장)
- 실시간(모의) 소비 카드
  - 최근 5분 결제액/결제건수
  - `st.cache_data(ttl=300)`로 5분마다 갱신

## 실제 데이터/API로 바꾸는 포인트

- 과거 데이터(CSV 등): `load_dummy_history()`를 실제 `pd.read_csv()` 기반 로더로 교체
- 실시간 API: `fetch_realtime_stub()` 내부를 `requests.get(...)` 호출로 교체
  - API 키는 `.env` 같은 환경변수로 관리하는 것을 권장
