"""데이터 계층 (2단계).

- base: DataProvider / UniverseProvider / FxProvider / FundingProvider 인터페이스
- kis: 한국투자증권 KIS Open API 분봉 → 15분봉
- ccxt_binance: 바이낸스 15분봉, 펀딩비 이력, 거래소 필터
- resample: 1분봉 → N분봉 (KRX 세션 경계 인식)
- universe: KOSPI200 시점별 구성종목 이력
- fx: USD/KRW 일별 환율
- adjust: 액면분할·병합 수정주가 보정
- cache: Parquet 로컬 캐시
"""
