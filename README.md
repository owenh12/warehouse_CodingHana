# RSI 강세 다이버전스 퀀트 시스템

KOSPI200 구성 종목과 BTC·ETH의 15분봉에서 **RSI 정규 강세 다이버전스**(가격 저점은 낮아지고 RSI 저점은 높아짐)를 찾아 매매한다. 백테스트 → 파라미터 검증 → 모의투자 → 실거래를 하나의 Python 프로젝트에서 같은 코드 경로로 진행한다.

- 설계 문서: [`docs/DESIGN.md`](docs/DESIGN.md)
- 모든 파라미터: [`config/`](config/) (코드에 하드코딩하지 않음)

## 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 프로젝트 구조, 설정 파일 설계 | ✅ 완료 |
| 2 | 데이터 확보 점검 → 데이터 계층, RSI, 피벗, 단위 테스트 | ✅ 바이낸스 완료 ([보고서](docs/STAGE2_DATA_REPORT.md)). KIS 보류 |
| 3 | 다이버전스 신호 + 신호 차트 육안 검증 (BTC/USDT 최근 3개월) | 🔶 완료, 승인 대기 ([보고서](docs/STAGE3_SIGNAL_REPORT.md)) |
| 4 | 백테스트 엔진 + 성과 리포트 | ⏳ |
| 5 | 파라미터 최적화, Walk-forward, 몬테카를로 | ⏳ |
| 6 | 모의투자 (PaperBroker) | ⏳ |
| 7 | 실거래 (KISBroker, CcxtBroker), 리스크 관리, 알림 | ⏳ |

## 설치

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,data,viz]"   # 단계가 진행되면 ".[all]"
cp .env.example .env               # API 키 입력 (git 에 커밋되지 않음)
```

## 명령

```bash
python -m rsidiv config                          # 설정 전체 검증 + 요약
python -m rsidiv config --override exp.yaml      # 실험용 덮어쓰기 적용 후 검증
python -m rsidiv secrets                         # .env 키 설정 여부 (이름만 표시, 값은 표시하지 않음)
python -m rsidiv data-check --markets spot usdm_futures   # 바이낸스 데이터 확보 점검 → storage/reports/data_check/
python -m rsidiv data-check --source rest        # REST API(ccxt)로 점검 (기본은 아카이브 data.binance.vision)
python -m rsidiv fetch                           # 백테스트 기간 15분봉을 Parquet 캐시에 저장
python -m rsidiv signals                         # BTC/USDT 최근 3개월 신호 + 육안 검증 차트 → storage/reports/signals/
python -m rsidiv signals --symbol ETH/USDT --months 6
python -m pytest                                 # 단위 테스트
RSIDIV_REAL_DATA=1 python -m pytest tests/test_real_data.py   # 실데이터 검증 (아카이브 다운로드)
```

## 디렉터리

```
config/            YAML 파라미터
docs/              설계·단계별 보고서
src/rsidiv/        core · data · indicators · signals · strategy · backtest ·
                   optimize · risk · broker · live · reports
tests/             단위 테스트
storage/           (gitignore) 캐시·상태 DB·로그·리포트
```
