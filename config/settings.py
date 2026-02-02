"""
MOMENTUM-X Configuration

### ARCHITECTURAL CONTEXT
Type-safe configuration using pydantic-settings. All secrets load from
environment variables or .env file. Thresholds are derived from
docs/mathematics/MOMENTUM_LOGIC.md.

### DESIGN DECISIONS
- pydantic-settings over raw os.environ for validation at startup (ADR-005)
- Nested models for logical grouping (broker, models, thresholds)
- All threshold defaults match the LaTeX definitions in MOMENTUM_LOGIC.md
- .env file auto-loaded for developer convenience (ADR-005 §2)
- SIP feed enforced as default (ADR-004 §1, CONSTRAINT-001)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve .env relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class AlpacaConfig(BaseSettings):
    """Alpaca broker configuration. Ref: DATA-001, DATA-001-EXT."""

    api_key: str = Field(default="", description="Alpaca API key ID")
    secret_key: str = Field(default="", description="Alpaca API secret key")
    base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Paper trading by default. NEVER default to live.",
    )
    data_url: str = Field(
        default="https://data.alpaca.markets",
        description="Market data endpoint (shared across paper/live)",
    )
    feed: str = Field(
        default="sip",
        description="Data feed: 'sip' (required for production per ADR-004) or 'iex' (testing only)",
    )

    model_config = SettingsConfigDict(env_prefix="ALPACA_", env_file=str(_ENV_FILE), extra="ignore")


class ModelConfig(BaseSettings):
    """LLM model configuration. Ref: ADR-001 Model Tiering."""

    # Tier 1: Reasoning (Debate, News, Risk agents)
    tier1_model: str = Field(
        default="deepseek/deepseek-r1-distill-qwen-32b",
        description="Primary reasoning model via Together AI / Fireworks",
    )
    tier1_provider: str = Field(
        default="together_ai",
        description="LiteLLM provider prefix",
    )

    # Tier 2: Extraction (Technical, Fundamental, Institutional agents)
    tier2_model: str = Field(
        default="qwen/qwen-2.5-14b-instruct",
        description="Extraction model for structured data analysis",
    )
    tier2_provider: str = Field(
        default="together_ai",
        description="LiteLLM provider prefix",
    )

    # Tier 3: Validation (Final conviction, used sparingly)
    tier3_model: str = Field(
        default="deepseek/deepseek-r1",
        description="Full R1 671B for highest-conviction calls only",
    )
    tier3_provider: str = Field(
        default="together_ai",
        description="LiteLLM provider prefix",
    )

    # Inference
    default_temperature: float = Field(default=0.3)
    max_tokens: int = Field(default=4096)
    timeout_seconds: int = Field(default=120)

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=str(_ENV_FILE), extra="ignore")


class ScannerThresholds(BaseSettings):
    """
    Scanning thresholds derived from MOMENTUM_LOGIC.md §1-§4.
    Every value here maps to a LaTeX definition.
    """

    # RVOL thresholds (MOMENTUM_LOGIC.md §2)
    rvol_premarket_min: float = Field(
        default=2.0,
        description="τ_rvol for pre-market. MOMENTUM_LOGIC.md §1",
    )
    rvol_intraday_min: float = Field(
        default=3.0,
        description="τ_rvol for intraday. MOMENTUM_LOGIC.md §1",
    )
    rvol_breakout_min: float = Field(
        default=2.0,
        description="Minimum RVOL for valid breakout. MOMENTUM_LOGIC.md §4",
    )

    # Gap thresholds (MOMENTUM_LOGIC.md §3)
    gap_pct_min: float = Field(
        default=0.05,
        description="τ_gap minimum (5%). MOMENTUM_LOGIC.md §1",
    )
    gap_pct_explosive: float = Field(
        default=0.20,
        description="Explosive gap threshold (20%). MOMENTUM_LOGIC.md §3",
    )

    # ATR ratio (MOMENTUM_LOGIC.md §4)
    atr_ratio_min: float = Field(
        default=1.5,
        description="τ_atr minimum. MOMENTUM_LOGIC.md §1",
    )
    atr_lookback_days: int = Field(
        default=14,
        description="ATR calculation period. MOMENTUM_LOGIC.md §4",
    )

    # Float constraints
    float_max_shares: int = Field(
        default=20_000_000,
        description="Max float for high-volatility focus (20M shares)",
    )
    float_ideal_shares: int = Field(
        default=10_000_000,
        description="Ideal float for maximum move potential (10M shares)",
    )

    # Volume
    premarket_volume_min_7am: int = Field(
        default=50_000,
        description="Min pre-market volume by 7:00 AM ET",
    )
    premarket_volume_min_9am: int = Field(
        default=100_000,
        description="Min pre-market volume by 9:00 AM ET",
    )

    # Price range
    price_min: float = Field(default=0.50, description="Min price filter")
    price_max: float = Field(default=20.00, description="Max price filter")

    model_config = SettingsConfigDict(env_prefix="SCAN_", env_file=str(_ENV_FILE), extra="ignore")


class ScoringWeights(BaseSettings):
    """
    Multi-Factor Composite Score weights. MOMENTUM_LOGIC.md §5.
    Must sum to 1.0.
    """

    catalyst_news: float = Field(default=0.30)
    technical: float = Field(default=0.20)
    volume_rvol: float = Field(default=0.20)
    float_structure: float = Field(default=0.15)
    institutional: float = Field(default=0.10)
    deep_search: float = Field(default=0.05)
    risk_aversion_lambda: float = Field(
        default=0.3,
        description="λ risk penalty. MOMENTUM_LOGIC.md §5",
    )

    model_config = SettingsConfigDict(env_prefix="SCORE_", env_file=str(_ENV_FILE), extra="ignore")


class ExecutionConfig(BaseSettings):
    """Position sizing and risk management. MOMENTUM_LOGIC.md §6."""

    max_positions: int = Field(default=3, description="Max concurrent positions")
    max_position_pct: float = Field(
        default=0.05,
        description="Max 5% of portfolio per position. MOMENTUM_LOGIC.md §6",
    )
    kelly_fraction: float = Field(
        default=0.5,
        description="Half-Kelly for estimation error. MOMENTUM_LOGIC.md §6",
    )
    stop_loss_pct: float = Field(default=0.07, description="Default 7% stop-loss")
    daily_loss_limit_pct: float = Field(
        default=0.05,
        description="Circuit breaker: halt if daily P&L < -5%",
    )
    close_positions_by: str = Field(
        default="15:45",
        description="Close all intraday positions by 3:45 PM ET",
    )

    model_config = SettingsConfigDict(env_prefix="EXEC_", env_file=str(_ENV_FILE), extra="ignore")


class DebateConfig(BaseSettings):
    """Debate engine parameters. ADR-001, MOMENTUM_LOGIC.md §10."""

    divergence_high_threshold: float = Field(
        default=0.6,
        description="DIV > 0.6 → full position. MOMENTUM_LOGIC.md §10",
    )
    divergence_low_threshold: float = Field(
        default=0.3,
        description="DIV < 0.3 → no trade. MOMENTUM_LOGIC.md §10",
    )
    mfcs_debate_threshold: float = Field(
        default=0.6,
        description="Minimum MFCS to trigger debate engine",
    )

    model_config = SettingsConfigDict(env_prefix="DEBATE_", env_file=str(_ENV_FILE), extra="ignore")


class Settings(BaseSettings):
    """Root configuration aggregating all sub-configs."""

    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    thresholds: ScannerThresholds = Field(default_factory=ScannerThresholds)
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)

    # Global
    mode: str = Field(
        default="paper",
        description="'paper' or 'live'. NEVER default to live.",
    )
    log_level: str = Field(default="INFO")
    max_candidates_per_scan: int = Field(default=20)


def load_settings() -> Settings:
    """Load settings from environment variables with validation."""
    return Settings()
