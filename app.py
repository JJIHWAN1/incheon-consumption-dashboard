import os
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_GU_LIST = [
    "강화군",
    "계양구",
    "남동구",
    "동구",
    "미추홀구",
    "부평구",
    "서구",
    "연수구",
    "옹진군",
    "중구",
]


@dataclass(frozen=True)
class MetricConfig:
    amount_col: str = "amount"
    tx_col: str = "tx_count"


METRIC = MetricConfig()


def _local_now() -> datetime:
    return datetime.now()


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col])
    return out


@st.cache_data(show_spinner=False)
def load_dummy_history(
    seed: int,
    start_dt: datetime,
    end_dt: datetime,
    gu_list: Tuple[str, ...],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # 5-minute frequency gives us a realistic "streaming" feel and supports 5-minute realtime rollups.
    times = pd.date_range(start=start_dt, end=end_dt, freq="5min", inclusive="left")
    n_t = len(times)

    records = []

    # Different profile per gu (baseline + peak hour + weekend lift)
    base_levels = rng.uniform(0.8, 1.4, size=len(gu_list))
    peak_hours = rng.integers(17, 21, size=len(gu_list))
    weekend_lifts = rng.uniform(1.05, 1.25, size=len(gu_list))

    for i, gu in enumerate(gu_list):
        base = base_levels[i]
        peak = int(peak_hours[i])
        wk_lift = weekend_lifts[i]

        # Create a smooth daily curve (hourly) then expand to 5-min.
        hours = times.hour + times.minute / 60.0
        # peak around 'peak' with another smaller lunch peak
        dinner = np.exp(-0.5 * ((hours - peak) / 2.0) ** 2)
        lunch = 0.55 * np.exp(-0.5 * ((hours - 12.5) / 1.8) ** 2)
        late = 0.25 * np.exp(-0.5 * ((hours - 22.0) / 2.5) ** 2)
        curve = 0.35 + 1.25 * dinner + lunch + late

        dow = times.dayofweek  # Mon=0
        is_weekend = (dow >= 5).astype(float)
        weekend_factor = 1.0 + (wk_lift - 1.0) * is_weekend

        # Add slow trend (seasonality-ish) + random noise
        t = np.linspace(0, 1, n_t)
        trend = 1.0 + 0.10 * np.sin(2 * np.pi * t)  # gentle wave
        noise = rng.normal(0, 0.10, size=n_t)

        intensity = base * curve * weekend_factor * trend * (1.0 + noise)
        intensity = np.clip(intensity, 0.05, None)

        # tx_count as Poisson
        lam = 22.0 * intensity
        tx = rng.poisson(lam=lam).astype(int)

        # amount = tx * avg_ticket, avg_ticket varies by time and gu.
        avg_ticket = (8500 + 2500 * dinner + 1500 * lunch) * rng.uniform(0.9, 1.1)
        amount = (tx * avg_ticket * rng.uniform(0.85, 1.15, size=n_t)).astype(int)

        df_gu = pd.DataFrame(
            {
                "datetime": times,
                "gu": gu,
                METRIC.tx_col: tx,
                METRIC.amount_col: amount,
            }
        )
        records.append(df_gu)

    df = pd.concat(records, ignore_index=True)
    df = _ensure_datetime(df, "datetime")
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()
    df["is_weekend"] = df["datetime"].dt.dayofweek >= 5
    return df


@st.cache_data(show_spinner=False, ttl=300)
def fetch_realtime_stub(gu: str, seed: int) -> dict:
    """Simulates a realtime API call. Cached with TTL=300s (=5min)."""
    rng = np.random.default_rng(seed + (hash(gu) % 10_000))
    now = _local_now()

    # Make the realtime number coherent with time-of-day
    h = now.hour + now.minute / 60
    dinner = np.exp(-0.5 * ((h - 19.0) / 2.0) ** 2)
    lunch = 0.55 * np.exp(-0.5 * ((h - 12.5) / 1.8) ** 2)
    curve = 0.35 + 1.35 * dinner + lunch

    base_tx = 18 + 10 * curve
    tx_5m = int(max(0, rng.poisson(lam=base_tx)))

    avg_ticket = 9000 + 4000 * dinner + 2000 * lunch
    amount_5m = int(max(0, tx_5m * avg_ticket * rng.uniform(0.8, 1.2)))

    return {
        "gu": gu,
        "as_of": now.isoformat(timespec="seconds"),
        "tx_count_5m": tx_5m,
        "amount_5m": amount_5m,
    }


def compute_summary(df_gu: pd.DataFrame, amount_col: str, tx_col: str) -> str:
    if df_gu.empty:
        return "선택한 조건에 해당하는 데이터가 없습니다."

    # Peak hour
    hourly = df_gu.groupby("hour")[[amount_col, tx_col]].mean(numeric_only=True)
    peak_hour = int(hourly[amount_col].idxmax())

    # Weekend vs weekday
    wk = df_gu[df_gu["is_weekend"]]
    wd = df_gu[~df_gu["is_weekend"]]
    if not wk.empty and not wd.empty:
        wk_amt = wk[amount_col].mean()
        wd_amt = wd[amount_col].mean()
        weekend_pct = (wk_amt - wd_amt) / max(wd_amt, 1e-9) * 100
    else:
        weekend_pct = np.nan

    # Recent change: last 7 days vs previous 7 days
    last_date = df_gu["date"].max()
    last_7_start = last_date - timedelta(days=6)
    prev_7_start = last_7_start - timedelta(days=7)
    prev_7_end = last_7_start - timedelta(days=1)

    last_7 = df_gu[(df_gu["date"] >= last_7_start) & (df_gu["date"] <= last_date)]
    prev_7 = df_gu[(df_gu["date"] >= prev_7_start) & (df_gu["date"] <= prev_7_end)]

    if not last_7.empty and not prev_7.empty:
        last_7_amt = last_7[amount_col].mean()
        prev_7_amt = prev_7[amount_col].mean()
        wow_pct = (last_7_amt - prev_7_amt) / max(prev_7_amt, 1e-9) * 100
    else:
        wow_pct = np.nan

    parts = [f"피크 시간대는 {peak_hour:02d}시입니다."]

    if not np.isnan(weekend_pct):
        parts.append(f"주말 평균 결제액이 평일 대비 {weekend_pct:+.0f}%입니다.")

    if not np.isnan(wow_pct):
        parts.append(f"최근 7일은 직전 7일 대비 {wow_pct:+.0f}%입니다.")

    return " ".join(parts)


def format_currency(v: float) -> str:
    try:
        return f"₩{int(round(v)):,}"
    except Exception:
        return "-"


def main() -> None:
    st.set_page_config(page_title="Incheon Consumption Dashboard", layout="wide")

    st.title("인천 구별 소비 패턴 대시보드 (Demo)")

    with st.sidebar:
        st.header("필터")

        gu = st.selectbox("구 선택", DEFAULT_GU_LIST, index=DEFAULT_GU_LIST.index("동구") if "동구" in DEFAULT_GU_LIST else 0)

        seed = st.number_input("데모 시드", min_value=0, max_value=9999, value=42, step=1)

        today = date.today()
        default_start = today - timedelta(days=60)
        start_date, end_date = st.date_input(
            "기간 선택",
            value=(default_start, today),
            min_value=today - timedelta(days=365),
            max_value=today,
        )

        st.caption("실시간(모의) 소비는 5분 TTL 캐시로 갱신됩니다.")
        refresh = st.button("실시간 데이터 새로고침")

        if refresh:
            st.cache_data.clear()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    df = load_dummy_history(seed=seed, start_dt=start_dt, end_dt=end_dt, gu_list=tuple(DEFAULT_GU_LIST))
    df_gu = df[df["gu"] == gu].copy()

    rt = fetch_realtime_stub(gu=gu, seed=seed)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("실시간(최근 5분) 결제액", format_currency(rt["amount_5m"]))
    with kpi2:
        st.metric("실시간(최근 5분) 결제건수", f"{rt['tx_count_5m']:,}건")
    with kpi3:
        st.metric("선택 기간 총 결제액", format_currency(df_gu[METRIC.amount_col].sum()))
    with kpi4:
        st.metric("선택 기간 총 결제건수", f"{int(df_gu[METRIC.tx_col].sum()):,}건")

    st.caption(f"실시간 데이터 기준 시각: {rt['as_of']} (TTL=300s)")

    summary = compute_summary(df_gu, amount_col=METRIC.amount_col, tx_col=METRIC.tx_col)
    st.subheader("핵심 요약")
    st.info(summary)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("시간대별 평균 소비")
        hourly = (
            df_gu.groupby("hour")[[METRIC.amount_col, METRIC.tx_col]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("hour")
        )
        fig_hour_amt = px.line(hourly, x="hour", y=METRIC.amount_col, markers=True, labels={"hour": "시간", METRIC.amount_col: "평균 결제액"})
        st.plotly_chart(fig_hour_amt, use_container_width=True)

        fig_hour_tx = px.line(hourly, x="hour", y=METRIC.tx_col, markers=True, labels={"hour": "시간", METRIC.tx_col: "평균 결제건수"})
        st.plotly_chart(fig_hour_tx, use_container_width=True)

    with c2:
        st.subheader("요일별 평균 소비")
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wd = (
            df_gu.groupby("weekday")[[METRIC.amount_col, METRIC.tx_col]]
            .mean(numeric_only=True)
            .reindex(weekday_order)
            .reset_index()
            .rename(columns={"weekday": "요일"})
        )

        fig_wd_amt = px.bar(wd, x="요일", y=METRIC.amount_col, labels={METRIC.amount_col: "평균 결제액"})
        st.plotly_chart(fig_wd_amt, use_container_width=True)

        fig_wd_tx = px.bar(wd, x="요일", y=METRIC.tx_col, labels={METRIC.tx_col: "평균 결제건수"})
        st.plotly_chart(fig_wd_tx, use_container_width=True)

    st.subheader("일자별 추이")
    daily = (
        df_gu.groupby("date")[[METRIC.amount_col, METRIC.tx_col]]
        .sum(numeric_only=True)
        .reset_index()
        .sort_values("date")
    )
    fig_daily = px.line(
        daily,
        x="date",
        y=[METRIC.amount_col, METRIC.tx_col],
        labels={"value": "값", "date": "날짜", "variable": "지표"},
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    with st.expander("데이터 미리보기"):
        st.dataframe(df_gu.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
