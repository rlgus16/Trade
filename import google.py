import google.generativeai as genai
from pybit.unified_trading import HTTP
import time

# 1. API 설정
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
BYBIT_API_KEY = "YOUR_BYBIT_API_KEY"
BYBIT_API_SECRET = "YOUR_BYBIT_API_SECRET"

genai.configure(api_key=GEMINI_API_KEY)
session = HTTP(
    testnet=False, # 실거래 시 False, 테스트넷 시 True
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET
)

def get_market_analysis(symbol="BTCUSDT"):
    # Bybit에서 최근 캔들 데이터(1시간 봉 10개) 가져오기
    kline = session.get_kline(
        category="linear",
        symbol=symbol,
        interval=60,
        limit=10
    )
    
    # 데이터를 텍스트 형태로 가공
    data_str = str(kline['result']['list'])
    
    # Gemini 3.1 Pro 모델 호출 및 분석 요청
    model = genai.GenerativeModel('gemini-3.1-pro')
    prompt = f"""
    너는 전문 퀀트 트레이더야. 아래 제공되는 {symbol}의 최근 캔들 데이터를 분석해줘.
    데이터 포맷: [[시간, 시가, 고가, 저가, 종가, 거래량, ...]]
    
    분석 기준:
    1. 추세 분석 및 주요 지지/저항 확인
    2. 현재가 기준 매수(BUY), 매도(SELL), 또는 관망(HOLD) 결정
    
    응답은 반드시 아래 JSON 형식으로만 해줘:
    {{"decision": "BUY" 또는 "SELL" 또는 "HOLD", "reason": "이유 요약"}}
    
    데이터: {data_str}
    """
    
    response = model.generate_content(prompt)
    return response.text

def execute_trade(symbol, decision):
    if decision == "BUY":
        print(f"[{symbol}] 매수 주문 실행")
        return session.place_order(
            category="linear",
            symbol=symbol,
            side="Buy",
            orderType="Market",
            qty="0.001" # 주문 수량 설정 필요
        )
    elif decision == "SELL":
        print(f"[{symbol}] 매도 주문 실행")
        return session.place_order(
            category="linear",
            symbol=symbol,
            side="Sell",
            orderType="Market",
            qty="0.001"
        )
    else:
        print("관망 중...")

# 실행 루프 (예시: 1시간마다 반복)
while True:
    try:
        analysis_result = get_market_analysis("BTCUSDT")
        # JSON 응답 파싱 및 결정 추출 로직 추가 필요
        # execute_trade("BTCUSDT", extracted_decision)
    except Exception as e:
        print(f"오류 발생: {e}")
    time.sleep(3600)