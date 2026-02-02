# MOMENTUM-X Research Bibliography

**Protocol**: TR-P §III.1 — Every implementation must trace to an entry here.

---

## Core Framework References

### [REF-001] TradingAgents: Multi-Agents LLM Financial Trading Framework
- **arXiv**: 2412.20138
- **Authors**: UCLA/MIT Research Group
- **Key Results**: Sharpe 8.21 on AAPL, 26.62% cumulative return (Jun-Nov 2024)
- **Relevance**: Bull/bear debate architecture, structured agent roles
- **Used In**: `/src/agents/debate_engine.py`, ADR-001

### [REF-002] MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents
- **arXiv**: 2502.00415
- **Key Results**: 125.9% cumulative return on S&P 100, Sharpe 4.00
- **Relevance**: Chain-of-Thought + RAG for actionable stock selection
- **Used In**: `/src/agents/coordinator.py`

### [REF-003] Sentiment Trading with LLMs (Kirtac & Germano)
- **arXiv**: 2412.19245
- **Key Results**: 3.05 Sharpe, 74.4% accuracy on 965K articles
- **Relevance**: OPT model sentiment extraction, long-short strategy design
- **Used In**: `/src/agents/news_agent.py`

### [REF-004] ChatGPT-Informed Graph Neural Network for Stock Prediction
- **arXiv**: 2306.03763
- **Key Results**: Superior cumulative returns with reduced volatility on DOW 30
- **Relevance**: Dynamic inter-company relationship graphs from news
- **Used In**: `/src/data/relationship_graph.py`

### [REF-005] DeepSeek-R1: Incentivizing Reasoning in LLMs via RL
- **arXiv**: 2501.12948
- **Key Results**: 72.6 AIME, 94.3 MATH-500 (32B distilled variant)
- **Relevance**: Primary reasoning kernel, native <think> blocks
- **Used In**: `/src/agents/reasoning_kernel.py`, ADR-001

### [REF-006] FinLoRA: Benchmarking LoRA for Financial LLMs
- **arXiv**: 2505.19819
- **Key Results**: 36% average improvement, <$300 training cost
- **Relevance**: Fine-tuning pipeline for domain specialization
- **Used In**: `/docs/decisions/ADR_003_FINE_TUNING.md` (future)

### [REF-007] Advances in Financial Machine Learning (Lopez de Prado)
- **ISBN**: 978-1119482086
- **Chapter 7**: Purged and Embargoed Cross-Validation
- **Chapter 12**: Backtesting through CPCV
- **Relevance**: Mandatory backtesting methodology per TR-P §I.2
- **Used In**: `/src/core/backtester.py`

### [REF-008] TradExpert: Mixture of Expert LLMs for Trading
- **arXiv**: 2411.00782
- **Relevance**: Specialized expert routing for multi-factor analysis
- **Used In**: `/src/agents/expert_router.py`

### [REF-009] DSPy: Programming—not Prompting—Foundation Models
- **URL**: https://github.com/stanfordnlp/dspy
- **Relevance**: Automated prompt optimization via MIPROv2
- **Used In**: `/src/agents/prompt_arena.py`

### [REF-010] TextGrad: Automatic Differentiation via Text
- **arXiv**: Published in Nature
- **URL**: https://github.com/zou-group/textgrad
- **Relevance**: Gradient-based prompt optimization
- **Used In**: `/src/agents/prompt_arena.py`

### [REF-011] Alpha Arena Season 1 Trading Competition Results
- **URL**: https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/
- **Key Results**: Qwen3 MAX +22.32%, DeepSeek V3.1 +4.89%, GPT-5 -53.29%
- **Relevance**: Model selection for live trading, risk discipline patterns
- **Used In**: `/docs/decisions/ADR_002_MODEL_SELECTION.md`

### [REF-012] Adaptive-OPRO for Financial Trading
- **Authors**: Papadakis et al., October 2025
- **Relevance**: Online prompt optimization with delayed rewards
- **Used In**: `/src/agents/prompt_arena.py`

### [REF-013] GAN-Based Synthetic Data for Backtest Robustness
- **arXiv**: 2209.04895
- **Relevance**: Synthetic price paths for overfitting prevention
- **Used In**: `/src/core/backtester.py`

---

## Data Source References

### [DATA-001] Alpaca Markets Trading API
- **URL**: https://docs.alpaca.markets/docs/trading-api
- **Latency**: ~1.5ms (live), ~731ms (paper)
- **Rate Limit**: 200 req/min
- **Used In**: `/src/execution/alpaca_client.py`

### [DATA-002] SEC EDGAR API
- **URL**: https://www.sec.gov/search-filings
- **Used In**: `/src/data/sec_client.py`

### [DATA-003] sec-api.io (Float, Filings, WebSocket)
- **URL**: https://sec-api.io/docs
- **Used In**: `/src/data/sec_client.py`, `/src/scanners/float_scanner.py`

### [DATA-004] Alpha Vantage News & Sentiment API
- **URL**: https://www.alphavantage.co/documentation/
- **Used In**: `/src/data/news_client.py`

### [DATA-005] Finnhub Financial Data API
- **URL**: https://finnhub.io/docs/api
- **Used In**: `/src/data/finnhub_client.py`

---

## Methodology Notes

- All REF-IDs must be cited in docstrings of implementing modules
- New references require a PR-style update to this file before code merge
- DATA-IDs track API dependencies for monitoring and failover
