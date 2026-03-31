from __future__ import annotations

import os
import json
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

load_dotenv()

ENERGY_TICKERS = {
    "Brent (ICE)": "BZ=F",
    "WTI (NYMEX)": "CL=F",
    "Henry Hub Natural Gas": "NG=F",
    "TTF Natural Gas": "TTF=F",
    "JKM LNG (Platts, proxy)": "JKM=F",
    "LPG (Saudi CP, proxy)": "BZ=F",
    "China PP Futures (proxy)": "PP=F",
    "China PG/LPG Futures (manual ticker)": "",
}

MARKET_DIMENSIONS = {
    "JKM LNG (Platts, proxy)": [
        "JKM seasonal structure",
        "JKM vs TTF spread",
        "JKM vs Brent slope",
        "Asia LNG shipping bottlenecks",
    ],
    "LPG (Saudi CP, proxy)": [
        "Saudi CP trend / month-over-month",
        "MB FEI vs CP spread",
        "LPG cracking & substitution economics",
        "Arb routes (USG/AG to Asia)",
    ],
    "China PP Futures (proxy)": [
        "PP margin and feedstock sensitivity",
        "PP/PG inter-product spread",
        "China domestic basis and inventory",
    ],
    "China PG/LPG Futures (manual ticker)": [
        "China PG futures structure",
        "Import parity vs domestic quotes",
        "PG-PP cross-spread monitoring",
    ],
}

ISOLATED_UPLOAD_INSTRUMENTS = {
    "JKM LNG (Platts, proxy)",
    "LPG (Saudi CP, proxy)",
    "China PP Futures (proxy)",
    "China PG/LPG Futures (manual ticker)",
}

PROVIDER_MODEL_DEFAULTS = {
    "openai": ("gpt-5.2", "gpt-5-mini"),
    "deepseek": ("deepseek-reasoner", "deepseek-chat"),
    "kimi": ("moonshot-v1-32k", "moonshot-v1-8k"),
    "xai": ("grok-4-0709", "grok-4-fast-non-reasoning"),
    "openrouter": ("openai/gpt-5", "openai/gpt-5-mini"),
    "google": ("gemini-2.5-pro", "gemini-2.5-flash"),
    "anthropic": ("claude-sonnet-4-5", "claude-haiku-4-5"),
    "ollama": ("qwen3:latest", "qwen3:latest"),
}


def build_config(
    llm_provider: str,
    deep_think_llm: str,
    quick_think_llm: str,
    max_debate_rounds: int,
    backend_url: str | None = None,
    api_key: str | None = None,
    uploaded_market_context: str | None = None,
) -> dict[str, Any]:
    """Build runtime config for TradingAgentsGraph."""

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = llm_provider
    config["deep_think_llm"] = deep_think_llm
    config["quick_think_llm"] = quick_think_llm
    config["max_debate_rounds"] = max_debate_rounds
    if backend_url:
        config["backend_url"] = backend_url
    if api_key:
        config["api_key"] = api_key
    if uploaded_market_context:
        config["uploaded_market_context"] = uploaded_market_context

    # Keep yfinance as the default vendor chain for all categories.
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    return config


def summarize_uploaded_excel(uploaded_file) -> str:
    """Summarize uploaded Excel into a compact text context."""
    try:
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
    except Exception as exc:  # noqa: BLE001
        return f"[Excel parse failed] {type(exc).__name__}: {exc}"

    segments: list[str] = []
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        cleaned = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if cleaned.empty:
            continue
        preview = cleaned.head(8).astype(str)
        numeric_cols = cleaned.select_dtypes(include="number")
        stats_text = ""
        if not numeric_cols.empty:
            desc = numeric_cols.describe().T[["mean", "std", "min", "max"]].round(3)
            stats_text = f"\nStats:\n{desc.to_string()}"
        segments.append(
            f"[Sheet: {sheet_name}] rows={len(cleaned)} cols={len(cleaned.columns)}\n"
            f"Columns: {', '.join(map(str, cleaned.columns[:30]))}\n"
            f"Preview:\n{preview.to_string(index=False)}{stats_text}"
        )

    if not segments:
        return "[Excel parsed but no usable non-empty sheets found]"
    return "\n\n".join(segments)


def normalize_models_for_provider(
    llm_provider: str,
    deep_think_llm: str,
    quick_think_llm: str,
) -> tuple[str, str]:
    """Normalize obviously incompatible model defaults by provider."""
    default_deep, default_quick = PROVIDER_MODEL_DEFAULTS.get(
        llm_provider, ("gpt-5.2", "gpt-5-mini")
    )
    if llm_provider == "deepseek" and deep_think_llm.startswith("gpt-"):
        deep_think_llm = default_deep
    if llm_provider == "deepseek" and quick_think_llm.startswith("gpt-"):
        quick_think_llm = default_quick
    if llm_provider == "kimi" and deep_think_llm.startswith("gpt-"):
        deep_think_llm = default_deep
    if llm_provider == "kimi" and quick_think_llm.startswith("gpt-"):
        quick_think_llm = default_quick
    return deep_think_llm, quick_think_llm


def render_reports(final_state: dict[str, Any], decision: str) -> None:
    """Render output panels."""

    st.subheader("Final Decision")
    st.code(decision)

    with st.expander("Market Report", expanded=False):
        st.write(final_state.get("market_report", "N/A"))

    with st.expander("Sentiment Report", expanded=False):
        st.write(final_state.get("sentiment_report", "N/A"))

    with st.expander("News Report", expanded=False):
        st.write(final_state.get("news_report", "N/A"))

    with st.expander("Fundamentals Report", expanded=False):
        st.write(final_state.get("fundamentals_report", "N/A"))


def _score_report(text: str | None) -> int:
    if not text:
        return 0
    positive_tokens = [
        "bull", "buy", "uptrend", "improving", "strong", "上涨", "看多", "改善"
    ]
    negative_tokens = [
        "bear", "sell", "downtrend", "weak", "risk", "下跌", "看空", "走弱"
    ]
    lower = text.lower()
    score = sum(lower.count(token) for token in positive_tokens)
    score -= sum(lower.count(token) for token in negative_tokens)
    return int(score)


def build_factor_dataframe(final_state: dict[str, Any]) -> pd.DataFrame:
    factors = [
        ("Market", _score_report(final_state.get("market_report"))),
        ("Sentiment", _score_report(final_state.get("sentiment_report"))),
        ("News", _score_report(final_state.get("news_report"))),
        ("Fundamentals", _score_report(final_state.get("fundamentals_report"))),
    ]
    df = pd.DataFrame(factors, columns=["factor", "score"])
    df["contribution"] = df["score"].cumsum()
    return df


def render_decision_overview_cards(decision: str, factor_df: pd.DataFrame) -> None:
    decision_upper = decision.upper()
    confidence = min(95, 50 + int(factor_df["score"].abs().sum() * 3))
    risk_level = (
        "High"
        if factor_df["score"].std() > 4
        else "Medium"
        if factor_df["score"].std() > 2
        else "Low"
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Decision", decision_upper)
    col2.metric("Confidence (heuristic)", f"{confidence}%")
    col3.metric("Risk Level", risk_level)


def render_factor_waterfall(factor_df: pd.DataFrame) -> None:
    st.subheader("因子贡献瀑布图（启发式）")
    st.caption("注：当前为文本打分启发式版本，后续可替换为模型结构化打分。")
    st.bar_chart(factor_df.set_index("factor")["score"])


def render_timeline_chart() -> None:
    st.subheader("分析时间线图")
    timeline_df = pd.DataFrame(
        [
            {"phase": "Market Analyst", "order": 1},
            {"phase": "Social Analyst", "order": 2},
            {"phase": "News Analyst", "order": 3},
            {"phase": "Fundamentals Analyst", "order": 4},
            {"phase": "Debate & Risk", "order": 5},
            {"phase": "Portfolio Decision", "order": 6},
        ]
    )
    st.line_chart(timeline_df.set_index("phase")["order"])


def export_report_payload(
    *,
    ticker: str,
    trade_date: str,
    decision: str,
    final_state: dict[str, Any],
    factor_df: pd.DataFrame,
    uploaded_market_context: str | None = None,
) -> tuple[str, str]:
    factor_table = factor_df.to_string(index=False)
    markdown = f"""# TradingAgents Report

- Ticker: `{ticker}`
- Trade Date: `{trade_date}`
- Final Decision: `{decision}`

## Factor Scores
```
{factor_table}
```

## Reports
### Market
{final_state.get("market_report", "N/A")}

### Sentiment
{final_state.get("sentiment_report", "N/A")}

### News
{final_state.get("news_report", "N/A")}

### Fundamentals
{final_state.get("fundamentals_report", "N/A")}

## Uploaded Context
{uploaded_market_context or "N/A"}
"""
    json_payload = {
        "ticker": ticker,
        "trade_date": trade_date,
        "decision": decision,
        "factor_scores": factor_df.to_dict(orient="records"),
        "reports": {
            "market_report": final_state.get("market_report", "N/A"),
            "sentiment_report": final_state.get("sentiment_report", "N/A"),
            "news_report": final_state.get("news_report", "N/A"),
            "fundamentals_report": final_state.get("fundamentals_report", "N/A"),
        },
        "uploaded_market_context": uploaded_market_context or "",
    }
    return markdown, json.dumps(json_payload, ensure_ascii=False, indent=2)


def main() -> None:
    st.set_page_config(page_title="TradingAgents Energy Desk", layout="wide")
    st.title("TradingAgents Energy Desk (Streamlit)")
    st.caption(
        "Focus contracts: Brent, WTI, Henry Hub, and TTF. "
        "Data vendors are configured to yfinance by default."
    )

    provider_env_key_map = {
        "openai": "OPENAI_API_KEY",
        "xai": "XAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "kimi": "MOONSHOT_API_KEY",
    }
    provider_backend_map = {
        "openai": "https://api.openai.com/v1",
        "xai": "https://api.x.ai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "kimi": "https://api.moonshot.cn/v1",
        "ollama": "http://localhost:11434/v1",
    }

    def _sync_ticker_from_instrument() -> None:
        selected = st.session_state.get("instrument_name")
        if selected in ENERGY_TICKERS:
            st.session_state["ticker_input"] = ENERGY_TICKERS[selected]

    with st.sidebar:
        st.header("Run Setup")
        instrument_name = st.selectbox(
            "Energy Instrument",
            list(ENERGY_TICKERS.keys()),
            key="instrument_name",
            on_change=_sync_ticker_from_instrument,
        )
        if "ticker_input" not in st.session_state:
            st.session_state["ticker_input"] = ENERGY_TICKERS[instrument_name]
        ticker = st.text_input(
            "Ticker (editable)",
            key="ticker_input",
            help="选择新标的时会自动同步默认 ticker；你也可以手动覆盖。",
        )
        trade_date = st.date_input("Trade Date", value=date.today())

        llm_provider = st.selectbox(
            "LLM Provider",
            ["openai", "google", "anthropic", "xai", "deepseek", "kimi", "openrouter", "ollama"],
            index=0,
        )
        api_key_input = st.text_input(
            "API Key (optional, 覆盖环境变量)",
            type="password",
            help="建议优先使用环境变量；这里输入后仅在当前会话生效。",
        )
        backend_url = st.text_input(
            "Backend URL",
            provider_backend_map.get(llm_provider, ""),
            help="兼容 OpenAI 协议的服务可自定义地址。",
        )
        default_deep, default_quick = PROVIDER_MODEL_DEFAULTS.get(llm_provider, ("gpt-5.2", "gpt-5-mini"))
        deep_think_llm = st.text_input("Deep-Think Model", default_deep)
        quick_think_llm = st.text_input("Quick-Think Model", default_quick)
        max_debate_rounds = st.slider("Max Debate Rounds", 1, 3, 1)
        st.divider()
        st.subheader("LPG/JKM 补充数据")
        uploaded_excel = st.file_uploader(
            "上传 Excel（可多 sheet）",
            type=["xlsx", "xls"],
            accept_multiple_files=False,
            help="用于补充 yfinance 缺失的 LPG/套利链路数据。",
        )
        uploaded_images = st.file_uploader(
            "上传报价/产业链图片（可多张）",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="图片暂不做 OCR，建议配合手工要点说明。",
        )
        manual_notes = st.text_area(
            "手工补充逻辑（支持中英文）",
            value="",
            height=120,
            placeholder="例如：MB FEI/CP倒挂、PG/PP套利阈值、月差逻辑、现货贴水结构等。",
        )

        run_button = st.button("Run Analysis", type="primary")

    st.info(
        "建议：对能源品种可额外接入 EIA / ICE / ENTSOG 等结构化数据源，"
        "再结合 yfinance 做多源校验，以提升信号稳定性。"
    )
    st.markdown("**当前品种扩展分析维度（可用于后续增强）**")
    for dim in MARKET_DIMENSIONS.get(instrument_name, []):
        st.markdown(f"- {dim}")

    if not run_button:
        return

    ticker = ticker.strip()
    if not ticker:
        st.error("Ticker 不能为空。请填写一个可识别的交易代码（例如 LPG 相关代理代码或你自定义的内部代码）。")
        return

    uploaded_context_parts: list[str] = []
    if uploaded_excel is not None:
        excel_summary = summarize_uploaded_excel(uploaded_excel)
        uploaded_context_parts.append(f"[Uploaded Excel Summary]\n{excel_summary}")
        with st.expander("已解析 Excel 摘要（发送给分析代理）", expanded=False):
            st.text(excel_summary[:8000])
    if uploaded_images:
        image_names = ", ".join(file.name for file in uploaded_images)
        uploaded_context_parts.append(f"[Uploaded Image Filenames]\n{image_names}")
        st.caption(f"已上传图片: {image_names}")
    if manual_notes.strip():
        uploaded_context_parts.append(f"[Manual Notes]\n{manual_notes.strip()}")
    uploaded_market_context = "\n\n".join(uploaded_context_parts).strip()
    if instrument_name not in ISOLATED_UPLOAD_INSTRUMENTS:
        if uploaded_market_context:
            st.info("当前品种处于隔离模式：上传数据不会注入分析代理，仅用于本地展示/导出。")
        uploaded_market_context = ""
    if len(uploaded_market_context) > 6000:
        st.warning("上传上下文较长，已自动截断到 6000 字符以避免模型超长请求失败。")
        uploaded_market_context = uploaded_market_context[:6000]

    had_gpt_model_name = deep_think_llm.startswith("gpt-") or quick_think_llm.startswith("gpt-")
    deep_think_llm, quick_think_llm = normalize_models_for_provider(
        llm_provider, deep_think_llm, quick_think_llm
    )
    if llm_provider in {"deepseek", "kimi"} and had_gpt_model_name:
        st.info("检测到与当前 Provider 不匹配的 GPT 模型名，已自动切换为该 Provider 的默认模型。")

    config = build_config(
        llm_provider=llm_provider,
        deep_think_llm=deep_think_llm,
        quick_think_llm=quick_think_llm,
        max_debate_rounds=max_debate_rounds,
        backend_url=backend_url,
        api_key=api_key_input or None,
        uploaded_market_context=uploaded_market_context or None,
    )

    env_key = provider_env_key_map.get(llm_provider)
    if api_key_input and env_key:
        os.environ[env_key] = api_key_input
    elif llm_provider in provider_env_key_map and not os.environ.get(env_key or ""):
        st.warning(
            f"未检测到 {env_key}，且未在页面输入 API Key，调用 {llm_provider} 可能会失败。"
        )

    try:
        with st.spinner("Trading agents are discussing the market..."):
            ta = TradingAgentsGraph(debug=False, config=config)
            final_state, decision = ta.propagate(ticker, trade_date.strftime("%Y-%m-%d"))
    except Exception as exc:
        st.error("模型调用失败，请检查 provider / model / API key / backend URL 是否匹配。")
        st.code(
            f"provider={llm_provider}\n"
            f"deep_model={deep_think_llm}\n"
            f"quick_model={quick_think_llm}\n"
            f"backend_url={backend_url}\n"
            f"error={type(exc).__name__}: {exc}"
        )
        return

    factor_df = build_factor_dataframe(final_state)
    render_decision_overview_cards(decision, factor_df)
    render_factor_waterfall(factor_df)
    render_timeline_chart()
    render_reports(final_state, decision)

    markdown_report, json_report = export_report_payload(
        ticker=ticker,
        trade_date=trade_date.strftime("%Y-%m-%d"),
        decision=decision,
        final_state=final_state,
        factor_df=factor_df,
        uploaded_market_context=uploaded_market_context or None,
    )
    col_md, col_json = st.columns(2)
    col_md.download_button(
        "导出 Markdown 报告",
        data=markdown_report,
        file_name=f"tradingagents_report_{ticker}_{trade_date}.md",
        mime="text/markdown",
    )
    col_json.download_button(
        "导出 JSON 报告",
        data=json_report,
        file_name=f"tradingagents_report_{ticker}_{trade_date}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
