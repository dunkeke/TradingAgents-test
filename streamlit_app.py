from __future__ import annotations

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


def build_config(
    llm_provider: str,
    deep_think_llm: str,
    quick_think_llm: str,
    max_debate_rounds: int,
) -> dict[str, Any]:
    """Build runtime config for TradingAgentsGraph."""

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = llm_provider
    config["deep_think_llm"] = deep_think_llm
    config["quick_think_llm"] = quick_think_llm
    config["max_debate_rounds"] = max_debate_rounds

    # Keep yfinance as the default vendor chain for all categories.
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    return config


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

    with st.sidebar:
        st.header("Run Setup")
        instrument_name = st.selectbox("Energy Instrument", list(ENERGY_TICKERS.keys()))
        ticker = st.text_input("Ticker (editable)", ENERGY_TICKERS[instrument_name])
        trade_date = st.date_input("Trade Date", value=date.today())

        llm_provider = st.selectbox(
            "LLM Provider",
            ["openai", "google", "anthropic", "xai", "openrouter", "ollama"],
            index=0,
        )
        deep_think_llm = st.text_input("Deep-Think Model", "gpt-5.2")
        quick_think_llm = st.text_input("Quick-Think Model", "gpt-5-mini")
        max_debate_rounds = st.slider("Max Debate Rounds", 1, 3, 1)

        run_button = st.button("Run Analysis", type="primary")

    st.info(
        "建议：对能源品种可额外接入 EIA / ICE / ENTSOG 等结构化数据源，"
        "再结合 yfinance 做多源校验，以提升信号稳定性。"
    )

    if not run_button:
        return

    config = build_config(
        llm_provider=llm_provider,
        deep_think_llm=deep_think_llm,
        quick_think_llm=quick_think_llm,
        max_debate_rounds=max_debate_rounds,
    )

    with st.spinner("Trading agents are discussing the market..."):
        ta = TradingAgentsGraph(debug=False, config=config)
        final_state, decision = ta.propagate(ticker, trade_date.strftime("%Y-%m-%d"))

    render_reports(final_state, decision)


if __name__ == "__main__":
    main()
