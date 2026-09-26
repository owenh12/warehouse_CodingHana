# perpdiv — 바이낸스 USDⓈ-M 상위 10개 코인 RSI 다이버전스

백테스트 → 파라미터 검증 → 페이퍼 → 실거래를 한 프로젝트에서 진행한다. 설계는 [docs/DESIGN.md](docs/DESIGN.md),
단계별 결과는 `docs/STAGE*_*.md` 에 있다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 구조·설정 설계 (config/*.yaml 9개, 스키마 검증) | 완료 |
| 2 | 데이터 확보 점검, 데이터 계층, RSI·ATR·피벗, 단위 테스트 | 완료 ([보고서](docs/STAGE2_DATA_REPORT.md)) |
| 3 | 신호 생성 + 시각 검증 (BTCUSDT, 최근 12개월, 4개 TF) | 승인 대기 |
| 4~7 | 백테스트 · 검증 · 페이퍼 · 실거래 | — |

## 설치

```bash
cd futures_divergence
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[data,viz,dev]"                        # 실거래 단계에서는 ".[live]" 추가
cp .env.example .env                                    # 키는 .env 에만. 출금 권한 없는 키 + IP 제한 권장
```

## 명령

```bash
python -m perpdiv config                                # 설정 9개 파일 검증 + 요약
python -m perpdiv config --override exp.yaml            # 실험용 덮어쓰기 YAML (예: strategy: {pivot: {left: 2}})
python -m perpdiv secrets                               # .env 키 설정 여부 (이름만 출력)
python -m perpdiv data-check --sample-month 2024-03,2026-08   # 데이터 확보 점검 → storage/reports/data_check/
python -m perpdiv resample-check --months 2024-03,2025-06   # 5분봉 리샘플 vs 원본 15m/1h/4h/1d 대조
python -m perpdiv source-check --days 3                 # 아카이브 vs 거래소 REST 5분봉 대조 (국내 PC 에서)
```

## 테스트·정적 검사

```bash
python -m pytest                                        # 단위 테스트 (네트워크 불필요)
PERPDIV_NETWORK_TESTS=1 python -m pytest -k native      # 아카이브 원본 봉 대조 (네트워크)
ruff check src tests && python -m mypy src tests        # 린트 + 타입(strict)
```

## 폴더

```
config/      설정 (모든 파라미터는 여기에만. 코드에 하드코딩 금지)
docs/        설계·단계별 보고서
src/perpdiv/
  core/        설정 스키마(pydantic, 알 수 없는 키 거부), 시간 규칙(UTC 저장·KST 표시), .env 비밀정보
  data/        아카이브(data.binance.vision, SHA-256 검증)·ccxt REST, 월 파티션 Parquet 캐시, 리샘플,
               품질 규칙(결측·중복·거래 중단·상장폐지), 심볼 해석, 거래대금 순위, 확보 점검
  indicators/  Wilder RSI·ATR (일괄·증분 비트 단위 동일), 피벗(동률 규칙·확정 시점)
  signals/ backtest/ optimize/ risk/ broker/ live/ reports/   (3단계 이후)
tests/       단위 테스트
storage/     캐시·보고서·상태 DB (git 제외)
```

## 데이터 원천

- 과거 데이터: `data.binance.vision` 공개 아카이브 (월·일 zip + CHECKSUM). 이 개발 환경에서는 바이낸스 API(fapi)가
  지역 차단(HTTP 451)되어 아카이브를 기본으로 쓴다. 아카이브 값은 거래소 봉과 동일하다(리샘플 대조로 확인).
- 실시간·페이퍼·실거래: ccxt `binanceusdm` (국내 PC 에서 실행).
