"""설정 스키마와 로더.

``config/*.yaml`` 9개 파일을 읽어 :class:`Settings` 로 검증한다.
- 정의되지 않은 키(오타)는 거부한다.
- 따옴표 없이 쓴 시각(YAML 이 ``00:00`` 을 60진수 정수로 읽는 경우)을 거부한다.
- 파일 간 정합성(타임프레임 배수 관계, 순위 봉 = 수집 봉, 탐색 공간 키 존재 등)을 검사한다.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = PROJECT_DIR / "config"
CONFIG_FILES = ("base", "universe", "data", "strategy", "costs", "risk", "backtest", "optimize", "live")

_TF_UNITS = {"m": 1, "h": 60, "d": 1440}


def timeframe_minutes(timeframe: str) -> int:
    """'5m' → 5, '4h' → 240, '1d' → 1440."""
    match = re.fullmatch(r"(\d+)([mhd])", timeframe)
    if not match or int(match.group(1)) <= 0:
        raise ValueError(f"타임프레임 형식 오류: {timeframe!r} (예: 1m, 5m, 15m, 1h, 4h, 1d)")
    return int(match.group(1)) * _TF_UNITS[match.group(2)]


def _check_timeframe(value: str) -> str:
    minutes = timeframe_minutes(value)
    if 1440 % minutes != 0 and minutes % 1440 != 0:
        raise ValueError(f"{value}: 하루(1440분)를 나누어떨어지게 하는 타임프레임만 지원합니다")
    return value


def _check_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"알 수 없는 시간대: {value!r}") from exc
    return value


def _parse_clock(value: Any) -> dt.time:
    if isinstance(value, int):
        raise ValueError("시각은 따옴표로 감싸야 합니다 (YAML 이 00:00 을 숫자로 읽음). 예: \"00:00\"")
    if isinstance(value, dt.time):
        return value
    return dt.time.fromisoformat(str(value))


Timeframe = Annotated[str, AfterValidator(_check_timeframe)]
TimeZoneName = Annotated[str, AfterValidator(_check_timezone)]
ClockTime = Annotated[dt.time, BeforeValidator(_parse_clock)]
Fraction = Annotated[float, Field(gt=0.0, lt=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
RsiLevel = Annotated[float, Field(gt=0.0, lt=100.0)]
FeeRate = Annotated[float, Field(ge=0.0, le=0.01)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# base.yaml
# ---------------------------------------------------------------------------


class ProjectCfg(_Model):
    name: str
    storage_timezone: Literal["UTC"]
    display_timezone: TimeZoneName


class AccountCfg(_Model):
    initial_capital: PositiveFloat
    quote_asset: Literal["USDT"]


class ExchangeCfg(_Model):
    venue: Literal["binance_usdm"]
    position_mode: Literal["one_way"]
    margin_mode: Literal["isolated"]
    leverage: Annotated[int, Field(ge=1, le=125)]


class PathsCfg(_Model):
    cache_dir: Path
    state_dir: Path
    log_dir: Path
    report_dir: Path


class BaseCfg(_Model):
    project: ProjectCfg
    account: AccountCfg
    exchange: ExchangeCfg
    paths: PathsCfg


# ---------------------------------------------------------------------------
# universe.yaml
# ---------------------------------------------------------------------------


class RankingCfg(_Model):
    metric: Literal["quote_volume"]
    window_hours: PositiveInt
    bar_timeframe: Timeframe
    aggregate_by: Literal["base_asset"]
    rank_quote_assets: list[str] = Field(min_length=1)
    usd_per_quote: dict[str, PositiveFloat]

    @model_validator(mode="after")
    def _rates_cover_quotes(self) -> RankingCfg:
        missing = set(self.rank_quote_assets) - set(self.usd_per_quote)
        if missing:
            raise ValueError(f"ranking.usd_per_quote 에 환산 비율이 없습니다: {sorted(missing)}")
        return self


class TradingUniverseCfg(_Model):
    quote_asset: Literal["USDT"]
    contract_type: Literal["perpetual"]


class ExcludeCfg(_Model):
    stablecoin_bases: list[str]
    tradfi_bases: list[str]
    reviewed_not_tradfi: list[str]  # 판별식이 TradFi 로 잡았지만 확인 결과 암호화폐인 코인 (제외하지 않음)
    symbols: list[str]

    @model_validator(mode="after")
    def _disjoint(self) -> ExcludeCfg:
        both = set(self.tradfi_bases) & set(self.reviewed_not_tradfi)
        if both:
            raise ValueError(f"tradfi_bases 와 reviewed_not_tradfi 에 모두 있는 코인: {sorted(both)}")
        return self


class CandidatesCfg(_Model):
    hourly_rank_margin: NonNegativeInt
    include_delisted: bool


class UniverseCfg(_Model):
    top_n: PositiveInt
    ranking: RankingCfg
    trading: TradingUniverseCfg
    exclude: ExcludeCfg
    candidates: CandidatesCfg

    @model_validator(mode="after")
    def _check(self) -> UniverseCfg:
        if self.trading.quote_asset not in self.ranking.rank_quote_assets:
            raise ValueError("trading.quote_asset 은 ranking.rank_quote_assets 에 포함되어야 합니다")
        return self


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------


class TimeframesCfg(_Model):
    collect: Timeframe
    signal: list[Timeframe] = Field(min_length=1)
    execution: Timeframe
    precise: Timeframe

    @model_validator(mode="after")
    def _multiples(self) -> TimeframesCfg:
        base = timeframe_minutes(self.collect)
        for tf in (*self.signal, self.execution):
            if timeframe_minutes(tf) % base != 0:
                raise ValueError(f"{tf} 는 수집 봉 {self.collect} 의 정수배여야 합니다")
        if base % timeframe_minutes(self.precise) != 0:
            raise ValueError(f"정밀 봉 {self.precise} 는 수집 봉 {self.collect} 를 나누어떨어지게 해야 합니다")
        if timeframe_minutes(self.execution) > min(timeframe_minutes(tf) for tf in self.signal):
            raise ValueError("실행 봉은 가장 짧은 신호 봉보다 길 수 없습니다")
        if len(set(self.signal)) != len(self.signal):
            raise ValueError("signal 타임프레임이 중복되었습니다")
        return self


class BacktestPeriodCfg(_Model):
    start: dt.date
    end: dt.date | Literal["last_month_end", "now"]  # 날짜는 그날까지 포함

    @model_validator(mode="after")
    def _ordered(self) -> BacktestPeriodCfg:
        if isinstance(self.end, dt.date) and self.end < self.start:
            raise ValueError("backtest_period.end 는 start 이후여야 합니다")
        return self

    def bounds(self, now: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
        """백테스트 구간 [시작, 끝) (UTC).

        - ``last_month_end``: 이번 달 1일 00:00 UTC 직전까지 (지난달 말일 포함). 펀딩비 월 파일이 있는 마지막 달.
        - ``now``: 현재 시각을 정시로 내림.
        - 날짜: 그날 24:00 UTC 까지.
        """
        start = dt.datetime.combine(self.start, dt.time(), tzinfo=dt.UTC)
        now = now.astimezone(dt.UTC)
        if self.end == "last_month_end":
            end = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif self.end == "now":
            end = now.replace(minute=0, second=0, microsecond=0)
        else:
            end = dt.datetime.combine(self.end + dt.timedelta(days=1), dt.time(), tzinfo=dt.UTC)
        if end <= start:
            raise ValueError(f"백테스트 구간이 비어 있습니다: {start} ~ {end}")
        return start, end


class BinanceProviderCfg(_Model):
    history_source: Literal["vision", "rest"]
    vision_base_url: str
    vision_listing_url: str
    verify_checksum: bool
    enable_rate_limit: bool
    page_limit: Annotated[int, Field(ge=1, le=1500)]
    timeout_sec: PositiveFloat
    max_retries: NonNegativeInt
    retry_backoff_sec: Annotated[float, Field(ge=0.0)]
    download_workers: Annotated[int, Field(ge=1, le=64)]


class ProvidersCfg(_Model):
    binance: BinanceProviderCfg


class InactiveBarCfg(_Model):
    rule: Literal["zero_trades"]
    halt_min_bars: PositiveInt


class QualityCfg(_Model):
    duplicate_bar: Literal["keep_last_if_identical"]
    missing_bar: Literal["keep_gap"]
    inactive_bar: InactiveBarCfg
    max_gap_warn_bars: PositiveInt


class CacheCfg(_Model):
    compression: Literal["zstd", "snappy", "gzip", "none"]
    refresh_recent_days: NonNegativeInt


class FundingSourceCfg(_Model):
    source: Literal["vision", "rest"]


class DataCfg(_Model):
    timeframes: TimeframesCfg
    backtest_period: BacktestPeriodCfg
    warmup_days: NonNegativeInt
    providers: ProvidersCfg
    quality: QualityCfg
    cache: CacheCfg
    funding: FundingSourceCfg


# ---------------------------------------------------------------------------
# strategy.yaml
# ---------------------------------------------------------------------------


class RsiCfg(_Model):
    period: Annotated[int, Field(ge=2)]
    source: Literal["close"]
    method: Literal["wilder"]


class AtrCfg(_Model):
    period: Annotated[int, Field(ge=1)]
    method: Literal["wilder"]


class PivotCfg(_Model):
    left: PositiveInt
    right: PositiveInt
    tie_rule: Literal["strict", "first"]


class BullishCfg(_Model):
    enabled: bool
    oversold: RsiLevel


class BearishCfg(_Model):
    enabled: bool
    overbought: RsiLevel


class GapBarsCfg(_Model):
    max_enabled: bool
    max: PositiveInt
    min_enabled: bool
    min: PositiveInt

    @model_validator(mode="after")
    def _ordered(self) -> GapBarsCfg:
        if self.max_enabled and self.min_enabled and self.min >= self.max:
            raise ValueError("gap_bars: min < max 여야 합니다")
        return self


class StructureCfg(_Model):
    anchor_candidates: Literal["all"]
    same_bar_signals: Literal["one_latest_anchor"]
    anchor_must_be_extreme: bool  # true: t1 = min Low[t1..p2] (약세 p1 = max High[p1..t2]) 이어야 신호
    gap_bars: GapBarsCfg


class StopCfg(_Model):
    atr_mult: PositiveFloat
    order_type: Literal["stop_market"]


class TakeProfitCfg(_Model):
    mode: Literal["structure", "r_multiple", "trailing"]
    r_multiple: PositiveFloat
    trailing_atr_mult: PositiveFloat


class TimeExitCfg(_Model):
    enabled: bool
    bars: PositiveInt


class ExitCfg(_Model):
    evaluation_timeframe: Timeframe
    stop: StopCfg
    take_profit: TakeProfitCfg
    time_exit: TimeExitCfg


class PositioningCfg(_Model):
    opposite_signal: Literal["ignore", "switch"]
    priority: list[Literal["timeframe_desc", "rank_asc"]]

    @model_validator(mode="after")
    def _priority(self) -> PositioningCfg:
        if self.priority != ["timeframe_desc", "rank_asc"]:
            raise ValueError("positioning.priority 는 [timeframe_desc, rank_asc] 만 지원합니다")
        return self


class StrategyCfg(_Model):
    rsi: RsiCfg
    atr: AtrCfg
    pivot: PivotCfg
    bullish: BullishCfg
    bearish: BearishCfg
    structure: StructureCfg
    exit: ExitCfg
    positioning: PositioningCfg

    @model_validator(mode="after")
    def _check(self) -> StrategyCfg:
        if not (self.bullish.enabled or self.bearish.enabled):
            raise ValueError("bullish 와 bearish 가 모두 꺼져 있습니다")
        if self.bullish.oversold >= self.bearish.overbought:
            raise ValueError("bullish.oversold 는 bearish.overbought 보다 작아야 합니다")
        return self


# ---------------------------------------------------------------------------
# costs.yaml
# ---------------------------------------------------------------------------

Liquidity = Literal["maker", "taker"]


class BnbDiscountCfg(_Model):
    enabled: bool
    rate: Annotated[float, Field(ge=0.0, lt=1.0)]


class FeesCfg(_Model):
    vip_level: NonNegativeInt
    maker: FeeRate
    taker: FeeRate
    basis: Literal["notional"]
    bnb_discount: BnbDiscountCfg

    def rate(self, liquidity: Liquidity) -> float:
        base = self.maker if liquidity == "maker" else self.taker
        return base * (1 - self.bnb_discount.rate) if self.bnb_discount.enabled else base


class OrderLiquidityCfg(_Model):
    market: Liquidity
    stop_market: Liquidity
    limit_resting: Liquidity
    limit_marketable: Liquidity


class FundingCostCfg(_Model):
    enabled: bool


class SlippageCfg(_Model):
    taker_pct: Annotated[float, Field(ge=0.0, lt=0.05)]
    maker_pct: Annotated[float, Field(ge=0.0, lt=0.05)]


class CostScenariosCfg(_Model):
    slippage_multipliers: list[Annotated[float, Field(ge=0.0)]] = Field(min_length=1)
    include_gross: bool


class CostsCfg(_Model):
    fees: FeesCfg
    order_liquidity: OrderLiquidityCfg
    funding: FundingCostCfg
    slippage: SlippageCfg
    scenarios: CostScenariosCfg


# ---------------------------------------------------------------------------
# risk.yaml
# ---------------------------------------------------------------------------


class SizingCfg(_Model):
    equity_fraction: Annotated[float, Field(gt=0.0, le=1.0)]
    fee_buffer: Annotated[float, Field(ge=0.0, lt=0.1)]
    reference_price: Literal["signal_close"]


class PositionsCfg(_Model):
    max_concurrent: Literal[1]


class LiquidationCfg(_Model):
    enabled: bool
    maintenance_margin_rate: Annotated[float, Field(gt=0.0, lt=0.5)]


class DailyLossCfg(_Model):
    enabled: bool
    threshold: Fraction
    reset_time: ClockTime
    reset_timezone: TimeZoneName
    keep_protective_orders: bool


class KillSwitchCfg(_Model):
    enabled: bool
    max_drawdown: Fraction
    on_trip: Literal["flatten_all", "keep_stops"]
    auto_resume: bool
    release_ack: str | None

    @model_validator(mode="after")
    def _no_auto_resume(self) -> KillSwitchCfg:
        if self.auto_resume:
            raise ValueError("kill_switch.auto_resume 은 false 만 허용합니다 (사람이 release_ack 로 해제)")
        return self


class ApiErrorsCfg(_Model):
    enabled: bool
    max_consecutive: PositiveInt
    reset_on_success: bool
    record_open_orders: bool
    release_ack: str | None


class RiskCfg(_Model):
    sizing: SizingCfg
    positions: PositionsCfg
    liquidation: LiquidationCfg
    daily_loss: DailyLossCfg
    kill_switch: KillSwitchCfg
    api_errors: ApiErrorsCfg


# ---------------------------------------------------------------------------
# backtest.yaml
# ---------------------------------------------------------------------------


class FillCfg(_Model):
    entry: Literal["next_open"]
    same_bar_stop_and_target: Literal["stop_first"]
    intrabar_precision: bool
    gap_through_stop: Literal["fill_at_open"]
    limit_fill_rule: Literal["touch", "through"]
    delisting: Literal["close_at_last_trade"]


class BenchmarkCfg(_Model):
    symbol: str


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
    benchmark: BenchmarkCfg
    metrics: MetricsCfg
    outputs: OutputsCfg


# ---------------------------------------------------------------------------
# optimize.yaml
# ---------------------------------------------------------------------------

MetricName = Literal["expectancy_r", "sharpe", "sortino", "calmar", "cagr", "total_return", "profit_factor"]


class IntParam(_Model):
    type: Literal["int"]
    low: int
    high: int
    step: PositiveInt = 1


class FloatParam(_Model):
    type: Literal["float"]
    low: float
    high: float
    step: PositiveFloat | None = None


class CategoricalParam(_Model):
    type: Literal["categorical"]
    choices: list[Any] = Field(min_length=1)


SearchParam = Annotated[IntParam | FloatParam | CategoricalParam, Field(discriminator="type")]


class ObjectiveCfg(_Model):
    metric: MetricName
    min_trades: PositiveInt
    report_also: list[MetricName]


class WalkForwardCfg(_Model):
    train: Annotated[str, Field(pattern=r"^\d+M$")]
    test: Annotated[str, Field(pattern=r"^\d+M$")]
    step: Annotated[str, Field(pattern=r"^\d+M$")]
    anchored: bool


class StabilityCfg(_Model):
    heatmap_pairs: list[tuple[str, str]]
    neighborhood_tolerance: Fraction


class MonteCarloCfg(_Model):
    n_sims: PositiveInt
    method: Literal["shuffle", "bootstrap"]
    kill_switch_drawdown: Fraction
    seed: int


class CrossSectionCfg(_Model):
    per_symbol: bool
    per_timeframe: bool


class OptimizeCfg(_Model):
    method: Literal["optuna", "grid"]
    n_trials: PositiveInt
    seed: int
    objective: ObjectiveCfg
    search_space: dict[str, SearchParam]
    walk_forward: WalkForwardCfg
    stability: StabilityCfg
    monte_carlo: MonteCarloCfg
    cross_section: CrossSectionCfg

    @model_validator(mode="after")
    def _ranges(self) -> OptimizeCfg:
        for key, param in self.search_space.items():
            if isinstance(param, IntParam | FloatParam) and param.low > param.high:
                raise ValueError(f"search_space.{key}: low > high")
        for a, b in self.stability.heatmap_pairs:
            for key in (a, b):
                if key not in self.search_space:
                    raise ValueError(f"stability.heatmap_pairs 의 {key} 가 search_space 에 없습니다")
        return self


# ---------------------------------------------------------------------------
# live.yaml
# ---------------------------------------------------------------------------


class SchedulerCfg(_Model):
    bar_close_delay_sec: Annotated[float, Field(ge=0.0, le=60.0)]
    max_bar_staleness_sec: PositiveFloat


class OrdersCfg(_Model):
    entry_type: Literal["market", "limit"]
    limit_timeout_sec: PositiveFloat
    limit_fallback_to_market: bool
    max_retries: NonNegativeInt
    retry_backoff_sec: Annotated[float, Field(ge=0.0)]
    protective_stop: Literal["reduce_only_stop_market"]


class StateCfg(_Model):
    sqlite_file: str
    reconcile_on_start: bool


class TelegramCfg(_Model):
    enabled: bool
    events: list[Literal["signal", "fill", "error", "risk", "daily_report"]]
    daily_report_time: ClockTime


class NotificationsCfg(_Model):
    telegram: TelegramCfg


class LoggingCfg(_Model):
    format: Literal["json"]
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class LiveCfg(_Model):
    mode: Literal["paper", "live"]
    live_confirm: bool
    scheduler: SchedulerCfg
    orders: OrdersCfg
    state: StateCfg
    notifications: NotificationsCfg
    logging: LoggingCfg

    @model_validator(mode="after")
    def _confirm(self) -> LiveCfg:
        if self.mode == "live" and not self.live_confirm:
            raise ValueError("mode: live 는 live_confirm: true 와 함께 설정해야 합니다")
        return self


# ---------------------------------------------------------------------------
# 전체
# ---------------------------------------------------------------------------


class Settings(_Model):
    base: BaseCfg
    universe: UniverseCfg
    data: DataCfg
    strategy: StrategyCfg
    costs: CostsCfg
    risk: RiskCfg
    backtest: BacktestCfg
    optimize: OptimizeCfg
    live: LiveCfg

    @model_validator(mode="after")
    def _cross_file(self) -> Settings:
        tfs = self.data.timeframes
        if self.universe.ranking.bar_timeframe != tfs.collect:
            raise ValueError("universe.ranking.bar_timeframe 은 data.timeframes.collect 와 같아야 합니다")
        if self.strategy.exit.evaluation_timeframe != tfs.execution:
            raise ValueError("strategy.exit.evaluation_timeframe 은 data.timeframes.execution 과 같아야 합니다")
        if self.base.exchange.leverage > 1 and not self.risk.liquidation.enabled:
            raise ValueError("레버리지 > 1 이면 risk.liquidation.enabled 가 true 여야 합니다")
        if self.base.account.quote_asset != self.universe.trading.quote_asset:
            raise ValueError("계좌 통화와 거래 계약의 증거금 자산이 다릅니다")
        strategy = self.strategy.model_dump()
        for key in self.optimize.search_space:
            if not _has_path(strategy, key):
                raise ValueError(f"optimize.search_space 의 {key} 가 strategy.yaml 에 없습니다")
        return self

    def strategy_with(self, updates: Mapping[str, Any]) -> StrategyCfg:
        """'pivot.left' 같은 점 표기 경로의 값을 바꾼 전략 설정 (최적화 시도용)."""
        merged = self.strategy.model_dump()
        for key, value in updates.items():
            if not _has_path(merged, key):
                raise KeyError(key)
            node = merged
            *parents, leaf = key.split(".")
            for part in parents:
                node = node[part]
            node[leaf] = value
        return StrategyCfg.model_validate(merged)


def _has_path(tree: Mapping[str, Any], dotted: str) -> bool:
    node: Any = tree
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return True


M = TypeVar("M", bound=BaseModel)


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 최상위는 매핑이어야 합니다")
    return data


def resolve_config_dir(config_dir: str | Path | None = None) -> Path:
    return Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR


def load_settings(config_dir: str | Path | None = None, overrides: Sequence[str | Path] = ()) -> Settings:
    """설정 디렉터리의 YAML 9개를 읽어 검증한다. ``overrides`` 는 실험용 덮어쓰기 YAML
    (최상위 키 = 파일 이름, 예: ``strategy: {pivot: {left: 2}}``)."""
    directory = resolve_config_dir(config_dir)
    raw = {name: _read_yaml(directory / f"{name}.yaml") for name in CONFIG_FILES}
    for override in overrides:
        patch = _read_yaml(Path(override))
        unknown = set(patch) - set(CONFIG_FILES)
        if unknown:
            raise ValueError(f"{override}: 알 수 없는 최상위 키 {sorted(unknown)}")
        raw = deep_merge(raw, patch)
    return Settings.model_validate(raw)


def resolve_project_path(path: Path) -> Path:
    """설정의 상대 경로는 프로젝트 폴더 기준."""
    return path if path.is_absolute() else PROJECT_DIR / path
