from __future__ import annotations

import os
from datetime import date
from typing import Any

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

    # Keep yfinance as the default vendor chain for all categories.
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    return config


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

    with st.sidebar:
        st.header("Run Setup")
        instrument_name = st.selectbox("Energy Instrument", list(ENERGY_TICKERS.keys()))
        ticker = st.text_input("Ticker (editable)", ENERGY_TICKERS[instrument_name])
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

        run_button = st.button("Run Analysis", type="primary")

    st.info(
        "建议：对能源品种可额外接入 EIA / ICE / ENTSOG 等结构化数据源，"
        "再结合 yfinance 做多源校验，以提升信号稳定性。"
    )

    if not run_button:
        return

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

    render_reports(final_state, decision)


if __name__ == "__main__":
    main()
