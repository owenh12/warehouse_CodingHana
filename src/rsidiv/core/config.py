"""설정 파일(config/*.yaml) 스키마와 로더.

전략·비용·리스크 등 모든 파라미터는 ``config/`` 아래 YAML에서만 정의하고 코드에
하드코딩하지 않는다. YAML 파일 하나가 :class:`Settings` 의 필드 하나에 대응하며,
로드 시 pydantic으로 타입·범위·파일 간 정합성을 검증한다. 정의되지 않은 키(오타)는
오류로 처리한다.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

AssetClassName = Literal["stock_kr", "crypto"]
ASSET_CLASSES: tuple[AssetClassName, ...] = ("stock_kr", "crypto")

#: Settings 필드명 → config 디렉터리 내 파일명
CONFIG_FILES: dict[str, str] = {
    "base": "base.yaml",
    "universe": "universe.yaml",
    "data": "data.yaml",
    "markets": "markets.yaml",
    "strategy": "strategy.yaml",
    "costs": "costs.yaml",
    "risk": "risk.yaml",
    "backtest": "backtest.yaml",
    "optimize": "optimize.yaml",
    "live": "live.yaml",
}

#: 설정 디렉터리 기본값. 환경변수 RSIDIV_CONFIG_DIR 로 바꿀 수 있다.
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

M = TypeVar("M", bound=BaseModel)


# ---------------------------------------------------------------------------
# 공통 타입
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    """모든 설정 모델의 기반: 미정의 키 금지, 불변."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _parse_clock(value: Any) -> dt.time:
    if isinstance(value, dt.time):
        return value
    if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}", value):
        raise ValueError(f"시각은 따옴표로 감싼 'HH:MM' 문자열이어야 합니다: {value!r}")
    return dt.time.fromisoformat(value)


def _check_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"알 수 없는 타임존: {value!r}") from exc
    return value


_TIMEFRAME_RE = re.compile(r"^(\d+)(m|h|d)$")
_TIMEFRAME_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440}


def timeframe_minutes(timeframe: str) -> int:
    """'15m', '4h', '1d' 같은 타임프레임 문자열을 분 단위 정수로 변환한다."""
    match = _TIMEFRAME_RE.fullmatch(timeframe)
    if match is None or int(match.group(1)) == 0:
        raise ValueError(f"타임프레임 형식 오류: {timeframe!r} (예: 1m, 15m, 4h, 1d)")
    return int(match.group(1)) * _TIMEFRAME_UNIT_MINUTES[match.group(2)]


def _check_timeframe(value: str) -> str:
    timeframe_minutes(value)
    return value


ClockTime = Annotated[dt.time, BeforeValidator(_parse_clock)]
TimeZoneName = Annotated[str, AfterValidator(_check_timezone)]
Timeframe = Annotated[str, AfterValidator(_check_timeframe)]
Period = Annotated[str, Field(pattern=r"^\d+[DWMY]$")]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
OpenFraction = Annotated[float, Field(gt=0.0, lt=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
RsiLevel = Annotated[float, Field(gt=0.0, lt=100.0)]


# ---------------------------------------------------------------------------
# base.yaml
# ---------------------------------------------------------------------------


class ProjectCfg(_Model):
    name: str
    base_currency: Literal["USD"]
    storage_timezone: Literal["UTC"]
    display_timezone: TimeZoneName


class CapitalCfg(_Model):
    initial: PositiveFloat
    allocation: dict[AssetClassName, Fraction]
    sleeve_mode: Literal["isolated"]

    @model_validator(mode="after")
    def _allocation_sums_to_one(self) -> CapitalCfg:
        total = sum(self.allocation.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"capital.allocation 합계가 1.0이 아닙니다: {total}")
        return self


class PathsCfg(_Model):
    cache_dir: Path
    state_dir: Path
    log_dir: Path
    report_dir: Path


class BaseCfg(_Model):
    project: ProjectCfg
    capital: CapitalCfg
    paths: PathsCfg


# ---------------------------------------------------------------------------
# universe.yaml
# ---------------------------------------------------------------------------


class MembershipCfg(_Model):
    point_in_time: bool
    source: Literal["krx"]
    include_delisted: bool


class StockUniverseCfg(_Model):
    enabled: bool
    venue: Literal["KRX"]
    provider: Literal["kis"]
    account_currency: Literal["KRW"]
    index: Literal["KOSPI200"]
    index_code: str
    membership: MembershipCfg
    include_symbols: list[str]
    exclude_symbols: list[str]


class CryptoUniverseCfg(_Model):
    enabled: bool
    venue: Literal["binance"]
    provider: Literal["binance"]
    account_currency: Literal["USDT"]
    market_type: Literal["spot", "usdm_futures"]
    symbols: list[str] = Field(min_length=1)


class UniverseCfg(_Model):
    stock_kr: StockUniverseCfg
    crypto: CryptoUniverseCfg


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------


class BacktestPeriodCfg(_Model):
    start: dt.date
    end: dt.date | None

    @model_validator(mode="after")
    def _ordered(self) -> BacktestPeriodCfg:
        if self.end is not None and self.end <= self.start:
            raise ValueError("backtest_period.end 는 start 보다 뒤여야 합니다")
        return self


KisEnv = Literal["real", "virtual"]


class KisProviderCfg(_Model):
    env: KisEnv
    base_url: dict[KisEnv, str]
    max_requests_per_sec: dict[KisEnv, PositiveFloat]
    market_division: Literal["J"]
    source_timeframe: Timeframe
    adjust_prices: bool


class BinanceProviderCfg(_Model):
    history_source: Literal["rest", "vision"]
    source_timeframe: Timeframe
    enable_rate_limit: bool
    page_limit: Annotated[int, Field(ge=1, le=1000)]
    timeout_sec: PositiveFloat
    max_retries: NonNegativeInt
    retry_backoff_sec: Annotated[float, Field(ge=0.0)]
    spot_public_api: str | None
    vision_base_url: str
    vision_verify_checksum: bool


class ProvidersCfg(_Model):
    kis: KisProviderCfg
    binance: BinanceProviderCfg


class FxCfg(_Model):
    pair: Literal["USD/KRW"]
    source: Literal["fred", "ecos", "yfinance"]
    fred_series: str
    fill_method: Literal["ffill"]
    usdt_per_usd: PositiveFloat


class QualityCfg(_Model):
    intraday_missing_bar: Literal["ffill_flat", "keep_gap"]
    halted_day: Literal["skip"]
    max_consecutive_missing_warn: PositiveInt
    dividend_handling: Literal["ignore", "cashflow"]


class CacheCfg(_Model):
    compression: Literal["zstd", "snappy", "gzip", "none"]
    refresh_recent_days: NonNegativeInt


class DataCfg(_Model):
    timeframe: Timeframe
    backtest_period: BacktestPeriodCfg
    providers: ProvidersCfg
    fx: FxCfg
    quality: QualityCfg
    cache: CacheCfg

    @model_validator(mode="after")
    def _source_divides_timeframe(self) -> DataCfg:
        target = timeframe_minutes(self.timeframe)
        for name in ("kis", "binance"):
            source = timeframe_minutes(getattr(self.providers, name).source_timeframe)
            if source > target or target % source != 0:
                raise ValueError(
                    f"providers.{name}.source_timeframe 은 timeframe({self.timeframe})의 약수여야 합니다"
                )
        return self


# ---------------------------------------------------------------------------
# markets.yaml
# ---------------------------------------------------------------------------


class TimeWindow(_Model):
    start: ClockTime
    end: ClockTime

    @model_validator(mode="after")
    def _ordered(self) -> TimeWindow:
        if self.start >= self.end:
            raise ValueError("start 는 end 보다 앞서야 합니다")
        return self


class SpecialSession(_Model):
    date: dt.date
    open: ClockTime
    close: ClockTime


class TickRule(_Model):
    below: PositiveFloat | None
    tick: PositiveFloat


class KrxCfg(_Model):
    calendar: str
    timezone: TimeZoneName
    regular_open: ClockTime
    regular_close: ClockTime
    opening_auction: TimeWindow
    closing_auction: TimeWindow
    special_sessions: list[SpecialSession]
    lot_size: PositiveInt
    daily_price_limit: OpenFraction
    tick_table: list[TickRule] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> KrxCfg:
        if self.regular_open >= self.regular_close:
            raise ValueError("krx.regular_open 은 regular_close 보다 앞서야 합니다")
        bounds = [rule.below for rule in self.tick_table]
        if bounds[-1] is not None or any(b is None for b in bounds[:-1]):
            raise ValueError("krx.tick_table: 마지막 구간만 below: null 이어야 합니다")
        finite = [b for b in bounds[:-1] if b is not None]
        if finite != sorted(finite) or len(set(finite)) != len(finite):
            raise ValueError("krx.tick_table: below 값은 오름차순이어야 합니다")
        return self

    def tick_size(self, price: float) -> float:
        """가격대별 호가단위를 반환한다."""
        for rule in self.tick_table:
            if rule.below is None or price < rule.below:
                return rule.tick
        raise AssertionError("unreachable: 마지막 구간은 below=None")


class SymbolFilter(_Model):
    min_qty: PositiveFloat
    qty_step: PositiveFloat
    min_notional: Annotated[float, Field(ge=0.0)]
    price_tick: PositiveFloat


CryptoMarketType = Literal["spot", "usdm_futures"]


class BinanceCfg(_Model):
    timezone: TimeZoneName
    always_open: bool
    filters_source: Literal["exchange_info", "fallback_only"]
    fallback_filters: dict[CryptoMarketType, dict[str, SymbolFilter]]


class MarketsCfg(_Model):
    krx: KrxCfg
    binance: BinanceCfg


# ---------------------------------------------------------------------------
# strategy.yaml
# ---------------------------------------------------------------------------


class RsiCfg(_Model):
    period: Annotated[int, Field(ge=2)]
    method: Literal["wilder"]
    source: Literal["close"]


class PivotCfg(_Model):
    left: PositiveInt
    right: PositiveInt
    price_source: Literal["low", "close"]
    strict: bool


class DivergenceCfg(_Model):
    t1_selection: Literal["previous_pivot", "scan_window"]
    p2_selection: Literal["highest_pivot", "max_high"]


class GapBarsFilter(_Model):
    enabled: bool
    min_bars: PositiveInt
    max_bars: PositiveInt

    @model_validator(mode="after")
    def _ordered(self) -> GapBarsFilter:
        if self.min_bars >= self.max_bars:
            raise ValueError("filters.gap_bars: min_bars < max_bars 여야 합니다")
        return self


class RsiOversoldFilter(_Model):
    enabled: bool
    threshold: RsiLevel


class RsiDiffFilter(_Model):
    enabled: bool
    min_diff: Annotated[float, Field(ge=0.0, lt=100.0)]


class PriceDropFilter(_Model):
    enabled: bool
    min_pct: Annotated[float, Field(ge=0.0, lt=1.0)]


class ToggleFilter(_Model):
    enabled: bool


class TrendFilter(_Model):
    enabled: bool
    mode: Literal["htf_ma", "ltf_ma"]
    htf: Timeframe
    ma_type: Literal["sma", "ema"]
    ma_period: Annotated[int, Field(ge=2)]
    condition: Literal["close_above", "slope_up", "close_above_and_slope_up"]


class VolumeFilter(_Model):
    enabled: bool
    mode: Literal["t3_vs_average", "t3_vs_t1"]
    bars_before: NonNegativeInt
    bars_after: NonNegativeInt
    lookback: PositiveInt
    min_ratio: PositiveFloat


class SessionFilterKR(_Model):
    enabled: bool
    exclude_first_bars: NonNegativeInt
    exclude_last_bars: NonNegativeInt
    exclude_auction_bars: bool
    applies_to: Literal["signal_bar", "t3_bar", "both"]


class FiltersCfg(_Model):
    gap_bars: GapBarsFilter
    rsi_t1_oversold: RsiOversoldFilter
    rsi_diff_min: RsiDiffFilter
    price_drop_min: PriceDropFilter
    no_lower_low_between: ToggleFilter
    trend: TrendFilter
    volume: VolumeFilter
    session_kr: SessionFilterKR


class EntryCfg(_Model):
    mode: Literal["A", "B", "C"]
    confirm_window_bars: PositiveInt
    rsi_cross_level: RsiLevel
    cancel_if_below_stop: bool
    order_type: Literal["market", "limit"]


class StopCfg(_Model):
    mode: Literal["atr", "pct"]
    atr_period: PositiveInt
    atr_mult: PositiveFloat
    pct: OpenFraction


class TakeProfitCfg(_Model):
    mode: Literal["p2", "r_multiple", "trailing", "none"]
    r_multiple: PositiveFloat
    trailing_atr_period: PositiveInt
    trailing_atr_mult: PositiveFloat
    p2_below_entry: Literal["skip", "fallback_r_multiple"]


class TimeExitCfg(_Model):
    enabled: bool
    max_bars: PositiveInt


class OvernightKRCfg(_Model):
    hold_overnight: bool
    flatten_minutes_before_close: NonNegativeInt


class ExitCfg(_Model):
    stop: StopCfg
    take_profit: TakeProfitCfg
    time_exit: TimeExitCfg
    overnight_kr: OvernightKRCfg


class StrategyParams(_Model):
    """한 자산군에 적용되는 전략 파라미터 전체 (strategy.yaml 의 default + override)."""

    rsi: RsiCfg
    pivot: PivotCfg
    divergence: DivergenceCfg
    filters: FiltersCfg
    entry: EntryCfg
    exit: ExitCfg

    @model_validator(mode="after")
    def _no_lookahead(self) -> StrategyParams:
        if self.filters.volume.bars_after > self.pivot.right:
            raise ValueError(
                "filters.volume.bars_after 는 pivot.right 이하여야 합니다 "
                "(신호 시점 t3+R 이후의 거래량을 쓰면 미래참조)"
            )
        return self


class StrategyCfg(_Model):
    default: StrategyParams
    overrides: dict[AssetClassName, dict[str, Any]]

    @model_validator(mode="after")
    def _overrides_valid(self) -> StrategyCfg:
        for asset_class in self.overrides:
            self.for_asset_class(asset_class)
        return self

    def for_asset_class(self, asset_class: AssetClassName) -> StrategyParams:
        """default 에 해당 자산군 override 를 병합한 파라미터를 반환한다."""
        merged = deep_merge(self.default.model_dump(), self.overrides.get(asset_class, {}))
        return StrategyParams.model_validate(merged)


# ---------------------------------------------------------------------------
# costs.yaml
# ---------------------------------------------------------------------------

FeeRate = Annotated[float, Field(ge=0.0, le=0.01)]


class CommissionCfg(_Model):
    rate: FeeRate
    rounding: Literal["floor_krw"]
    verify_with_broker: bool


class TaxRate(_Model):
    effective_from: dt.date
    rate: Annotated[float, Field(ge=0.0, le=0.01)]


class SellTaxCfg(_Model):
    rounding: Literal["floor_krw"]
    schedule: list[TaxRate] = Field(min_length=1)

    @model_validator(mode="after")
    def _sorted(self) -> SellTaxCfg:
        dates = [entry.effective_from for entry in self.schedule]
        if dates != sorted(dates) or len(set(dates)) != len(dates):
            raise ValueError("sell_tax.schedule 은 effective_from 오름차순, 중복 없음이어야 합니다")
        return self

    def rate_on(self, day: dt.date) -> float:
        """해당 일자 매도분에 적용되는 세율."""
        applicable = [entry.rate for entry in self.schedule if entry.effective_from <= day]
        if not applicable:
            raise ValueError(f"{day} 에 적용할 매도세율이 schedule 에 없습니다")
        return applicable[-1]


class StockSlippageCfg(_Model):
    market_ticks: NonNegativeInt
    limit_ticks: NonNegativeInt


class StockKRCostsCfg(_Model):
    commission: CommissionCfg
    sell_tax: SellTaxCfg
    slippage: StockSlippageCfg


class BnbDiscountCfg(_Model):
    enabled: bool
    rate: Annotated[float, Field(ge=0.0, lt=1.0)]


Liquidity = Literal["maker", "taker"]


class FeeTierCfg(_Model):
    maker: FeeRate
    taker: FeeRate
    bnb_discount: BnbDiscountCfg

    def rate(self, liquidity: Liquidity) -> float:
        """BNB 할인 반영 후 실효 수수료율."""
        base = self.maker if liquidity == "maker" else self.taker
        return base * (1.0 - self.bnb_discount.rate) if self.bnb_discount.enabled else base


class FundingCfg(_Model):
    enabled: bool
    source: Literal["historical"]


class FuturesFeeTierCfg(FeeTierCfg):
    fee_basis: Literal["notional"]
    funding: FundingCfg


class CryptoSlippageCfg(_Model):
    taker_pct: Annotated[float, Field(ge=0.0, lt=0.05)]
    maker_pct: Annotated[float, Field(ge=0.0, lt=0.05)]


class CryptoCostsCfg(_Model):
    vip_level: NonNegativeInt
    spot: FeeTierCfg
    usdm_futures: FuturesFeeTierCfg
    slippage: CryptoSlippageCfg


class OrderLiquidityCfg(_Model):
    market: Liquidity
    stop_market: Liquidity
    limit_resting: Liquidity
    limit_marketable: Liquidity


class CostScenariosCfg(_Model):
    slippage_multipliers: list[Annotated[float, Field(ge=0.0)]] = Field(min_length=1)
    include_gross: bool


class CostsCfg(_Model):
    stock_kr: StockKRCostsCfg
    crypto: CryptoCostsCfg
    order_liquidity: OrderLiquidityCfg
    scenarios: CostScenariosCfg


# ---------------------------------------------------------------------------
# risk.yaml
# ---------------------------------------------------------------------------


class SizingCfg(_Model):
    risk_per_trade: OpenFraction
    capital_base: Literal["sleeve_equity", "sleeve_initial"]
    reference_price: Literal["signal_close"]
    include_costs_in_risk: bool
    on_below_min: Literal["skip"]
    on_cap_exceeded: Literal["cap", "skip"]
    cash_buffer: Annotated[float, Field(ge=0.0, lt=0.2)]


class AssetLimitsCfg(_Model):
    max_positions: PositiveInt
    max_weight: Annotated[float, Field(gt=0.0, le=1.0)]


class LimitsCfg(_Model):
    stock_kr: AssetLimitsCfg
    crypto: AssetLimitsCfg
    one_position_per_symbol: bool


class ResetRule(_Model):
    type: Literal["market_open", "clock"]
    time: ClockTime | None = None
    timezone: TimeZoneName | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ResetRule:
        has_clock = self.time is not None and self.timezone is not None
        if self.type == "clock" and not has_clock:
            raise ValueError("reset type=clock 은 time 과 timezone 이 필요합니다")
        if self.type == "market_open" and (self.time is not None or self.timezone is not None):
            raise ValueError("reset type=market_open 에는 time/timezone 을 지정하지 않습니다")
        return self


class DailyLossCfg(_Model):
    enabled: bool
    threshold: OpenFraction
    basis: Literal["day_start_equity"]
    scope: Literal["per_account"]
    keep_protective_orders: bool
    reset: dict[AssetClassName, ResetRule]

    @model_validator(mode="after")
    def _all_asset_classes(self) -> DailyLossCfg:
        missing = set(ASSET_CLASSES) - set(self.reset)
        if missing:
            raise ValueError(f"daily_loss.reset 에 자산군이 빠졌습니다: {sorted(missing)}")
        return self


class KillSwitchCfg(_Model):
    enabled: bool
    max_drawdown: OpenFraction
    basis: Literal["peak_equity_since_start"]
    scope: Literal["portfolio_usd", "per_account"]
    on_trip: Literal["flatten_all", "keep_stops"]
    auto_resume: bool
    release_ack: str | None

    @model_validator(mode="after")
    def _no_auto_resume(self) -> KillSwitchCfg:
        if self.auto_resume:
            raise ValueError("kill_switch.auto_resume 은 false 만 허용됩니다 (수동 해제만 가능)")
        return self


class ApiErrorsCfg(_Model):
    enabled: bool
    max_consecutive: PositiveInt
    reset_on_success: bool
    record_open_orders: bool
    release_ack: str | None


class RiskCfg(_Model):
    sizing: SizingCfg
    limits: LimitsCfg
    daily_loss: DailyLossCfg
    kill_switch: KillSwitchCfg
    api_errors: ApiErrorsCfg


# ---------------------------------------------------------------------------
# backtest.yaml
# ---------------------------------------------------------------------------


class FillCfg(_Model):
    entry_price: Literal["next_open"]
    same_bar_stop_and_target: Literal["stop_first"]
    gap_through_stop: Literal["fill_at_open"]
    limit_fill_rule: Literal["touch", "through"]
    trailing_update: Literal["on_bar_close"]


class BenchmarksCfg(_Model):
    stock_kr: Literal["kospi200_index", "equal_weight_members"]
    crypto: Literal["equal_weight_symbols"]
    portfolio: Literal["allocation_weighted"]


class MetricsCfg(_Model):
    risk_free_rate: Annotated[float, Field(ge=0.0, lt=1.0)]
    return_frequency: Literal["daily"]


class OutputsCfg(_Model):
    trades_csv: bool
    signals_csv: bool
    equity_csv: bool
    charts: list[Literal["equity", "drawdown", "signal_samples"]]
    signal_sample_count: NonNegativeInt


class BacktestCfg(_Model):
    fill: FillCfg
    risk_rules_variants: list[Literal["with_live_rules", "without_live_rules"]] = Field(min_length=1)
    benchmarks: BenchmarksCfg
    metrics: MetricsCfg
    outputs: OutputsCfg


# ---------------------------------------------------------------------------
# optimize.yaml
# ---------------------------------------------------------------------------

MetricName = Literal[
    "expectancy_r", "sharpe", "sortino", "calmar", "cagr", "total_return", "profit_factor"
]


class IntParam(_Model):
    type: Literal["int"]
    low: int
    high: int
    step: PositiveInt = 1

    @model_validator(mode="after")
    def _ordered(self) -> IntParam:
        if self.low >= self.high:
            raise ValueError("low < high 여야 합니다")
        return self


class FloatParam(_Model):
    type: Literal["float"]
    low: float
    high: float
    step: PositiveFloat | None = None
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> FloatParam:
        if self.low >= self.high:
            raise ValueError("low < high 여야 합니다")
        return self


class CategoricalParam(_Model):
    type: Literal["categorical"]
    choices: list[int | float | str | bool] = Field(min_length=1)


ParamSpec = Annotated[IntParam | FloatParam | CategoricalParam, Field(discriminator="type")]


class ObjectiveCfg(_Model):
    metric: MetricName
    min_trades: NonNegativeInt
    report_also: list[MetricName]


class WalkForwardCfg(_Model):
    train: Period
    test: Period
    step: Period
    anchored: bool


class StabilityCfg(_Model):
    heatmap_pairs: list[tuple[str, str]]
    neighborhood_tolerance: OpenFraction


class MonteCarloCfg(_Model):
    n_sims: PositiveInt
    method: Literal["shuffle", "bootstrap"]
    kill_switch_drawdown: OpenFraction
    seed: int


class CrossAssetCfg(_Model):
    report_per_symbol: bool
    report_per_asset_class: bool


class OptimizeCfg(_Model):
    method: Literal["optuna", "grid"]
    n_trials: PositiveInt
    seed: int
    objective: ObjectiveCfg
    search_space: dict[str, ParamSpec]
    walk_forward: WalkForwardCfg
    stability: StabilityCfg
    monte_carlo: MonteCarloCfg
    cross_asset: CrossAssetCfg

    @model_validator(mode="after")
    def _check(self) -> OptimizeCfg:
        for pair in self.stability.heatmap_pairs:
            unknown = [key for key in pair if key not in self.search_space]
            if unknown:
                raise ValueError(f"stability.heatmap_pairs 키가 search_space 에 없습니다: {unknown}")
        if self.method == "grid":
            continuous = [
                key
                for key, spec in self.search_space.items()
                if isinstance(spec, FloatParam) and spec.step is None
            ]
            if continuous:
                raise ValueError(f"grid 탐색은 float 파라미터에 step 이 필요합니다: {continuous}")
        return self


# ---------------------------------------------------------------------------
# live.yaml
# ---------------------------------------------------------------------------


class SchedulerCfg(_Model):
    bar_close_delay_sec: dict[AssetClassName, Annotated[float, Field(ge=0.0)]]
    stock_kr_calendar: str


class StopExecutionCfg(_Model):
    stock_kr: Literal["client_monitor"]
    crypto: Literal["exchange_native", "client_monitor"]


class OrdersCfg(_Model):
    unfilled_timeout_sec: PositiveFloat
    max_retries: NonNegativeInt
    retry_backoff_sec: list[PositiveFloat]
    partial_fill: Literal["resize_protective"]
    stop_execution: StopExecutionCfg
    stop_monitor_interval_sec: PositiveFloat

    @model_validator(mode="after")
    def _backoff_len(self) -> OrdersCfg:
        if len(self.retry_backoff_sec) < self.max_retries:
            raise ValueError("orders.retry_backoff_sec 길이는 max_retries 이상이어야 합니다")
        return self


class DailyReportCfg(_Model):
    time: ClockTime
    timezone: TimeZoneName


class NotificationsCfg(_Model):
    channels: list[Literal["telegram", "slack"]] = Field(min_length=1)
    events: list[Literal["signal", "fill", "error", "risk_event", "daily_report"]]
    daily_report: DailyReportCfg


class LoggingCfg(_Model):
    format: Literal["json"]
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    redact_keys: list[str]


class PaperCfg(_Model):
    min_duration_weeks: PositiveInt
    fill_model: Literal["backtest_rules"]


class LiveCfg(_Model):
    mode: Literal["paper", "live"]
    live_confirm: bool
    scheduler: SchedulerCfg
    orders: OrdersCfg
    notifications: NotificationsCfg
    logging: LoggingCfg
    paper: PaperCfg

    @model_validator(mode="after")
    def _live_requires_confirm(self) -> LiveCfg:
        if self.mode == "live" and not self.live_confirm:
            raise ValueError("mode: live 는 live_confirm: true 일 때만 허용됩니다")
        return self


# ---------------------------------------------------------------------------
# 전체 설정 + 파일 간 정합성
# ---------------------------------------------------------------------------


class Settings(_Model):
    base: BaseCfg
    universe: UniverseCfg
    data: DataCfg
    markets: MarketsCfg
    strategy: StrategyCfg
    costs: CostsCfg
    risk: RiskCfg
    backtest: BacktestCfg
    optimize: OptimizeCfg
    live: LiveCfg

    @model_validator(mode="after")
    def _cross_file_checks(self) -> Settings:
        errors: list[str] = []

        for asset_class in ASSET_CLASSES:
            enabled = getattr(self.universe, asset_class).enabled
            if enabled and self.base.capital.allocation.get(asset_class, 0.0) <= 0.0:
                errors.append(f"{asset_class} 가 활성화되었지만 capital.allocation 이 0 입니다")

        market_type = self.universe.crypto.market_type
        filters = self.markets.binance.fallback_filters.get(market_type, {})
        missing = [s for s in self.universe.crypto.symbols if s not in filters]
        if missing:
            errors.append(f"markets.binance.fallback_filters.{market_type} 에 없는 심볼: {missing}")

        base_minutes = timeframe_minutes(self.data.timeframe)
        for asset_class in ASSET_CLASSES:
            trend = self.strategy.for_asset_class(asset_class).filters.trend
            htf = timeframe_minutes(trend.htf)
            if trend.mode == "htf_ma" and (htf <= base_minutes or htf % base_minutes != 0):
                errors.append(
                    f"[{asset_class}] filters.trend.htf({trend.htf}) 는 "
                    f"data.timeframe({self.data.timeframe}) 의 배수이며 더 커야 합니다"
                )

        default_params = self.strategy.default
        for key, spec in self.optimize.search_space.items():
            probes: list[Any] = (
                list(spec.choices) if isinstance(spec, CategoricalParam) else [spec.low, spec.high]
            )
            for value in probes:
                try:
                    apply_dotted(default_params, {key: value})
                except (KeyError, ValueError) as exc:
                    errors.append(f"optimize.search_space.{key}={value!r} 적용 불가: {exc}")
                    break

        if errors:
            raise ValueError("설정 파일 간 정합성 오류:\n  - " + "\n  - ".join(errors))
        return self


# ---------------------------------------------------------------------------
# 로더와 유틸리티
# ---------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """중첩 dict 병합. update 의 값이 우선하며, 리스트는 통째로 교체한다."""
    merged: dict[str, Any] = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_dotted(model: M, updates: Mapping[str, Any]) -> M:
    """'pivot.left' 같은 점 표기 경로의 값을 바꾼 새 모델을 검증 후 반환한다.

    최적화 시도마다 파라미터를 바꿀 때 사용한다. 존재하지 않는 경로는 KeyError.
    """
    data = model.model_dump()
    for dotted, value in updates.items():
        node = data
        *parents, leaf = dotted.split(".")
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"설정 경로 없음: {dotted}")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"설정 경로 없음: {dotted}")
        node[leaf] = value
    return type(model).model_validate(data)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError(f"{path}: 최상위는 매핑이어야 합니다")
    return content


def resolve_config_dir(config_dir: str | Path | None = None) -> Path:
    """인자 → 환경변수 RSIDIV_CONFIG_DIR → 저장소 config/ 순으로 설정 디렉터리를 정한다."""
    if config_dir is not None:
        return Path(config_dir)
    env_dir = os.environ.get("RSIDIV_CONFIG_DIR")
    return Path(env_dir) if env_dir else DEFAULT_CONFIG_DIR


def load_settings(
    config_dir: str | Path | None = None,
    overrides: Sequence[str | Path] = (),
) -> Settings:
    """설정 디렉터리의 YAML 전체를 읽어 검증된 :class:`Settings` 를 반환한다.

    Args:
        config_dir: config 디렉터리. None 이면 :func:`resolve_config_dir` 규칙을 따른다.
        overrides: 실험용 덮어쓰기 YAML 경로들. 최상위 키는 Settings 필드명
            (예: ``strategy: {default: {pivot: {left: 4}}}``)이며 순서대로 병합된다.
    """
    directory = resolve_config_dir(config_dir)
    raw: dict[str, Any] = {key: _read_yaml(directory / name) for key, name in CONFIG_FILES.items()}
    for override_path in overrides:
        patch = _read_yaml(Path(override_path))
        unknown = set(patch) - set(CONFIG_FILES)
        if unknown:
            raise ValueError(f"{override_path}: 알 수 없는 최상위 키 {sorted(unknown)}")
        raw = deep_merge(raw, patch)
    return Settings.model_validate(raw)
