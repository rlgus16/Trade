import google.generativeai as genai
from pybit.unified_trading import HTTP
import time
import json
import re

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

# 2. 트레이딩 규칙 설정 (ETH/USDT, 교차 마진, 5배 레버리지)
def setup_trading_rules(symbol="ETHUSDT"):
    try:
        # 교차 마진(Cross Margin) 설정 (tradeMode=0)
        session.switch_margin_mode(
            category="linear",
            symbol=symbol,
            tradeMode=0, 
            buyLeverage="5",
            sellLeverage="5"
        )
        print(f"[{symbol}] 교차 마진(Cross Mode) 및 5배 레버리지 설정 완료")
    except Exception as e:
        if "Not modified" in str(e) or "already" in str(e):
            print(f"[{symbol}] 이미 교차 마진 및 5배 레버리지로 설정되어 있습니다.")
        else:
            print(f"마진/레버리지 설정 안내: {e}")

def get_market_analysis(symbol="ETHUSDT"):
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
    # ETH의 경우 최소 주문 수량(qty)을 확인해야 합니다. (예: 0.01)
    trade_qty = "0.01" 

    if decision == "BUY":
        print(f"[{symbol}] 매수(Long) 주문 실행 (레버리지 5x)")
        return session.place_order(
            category="linear",
            symbol=symbol,
            side="Buy",
            orderType="Market",
            qty=trade_qty 
        )
    elif decision == "SELL":
        print(f"[{symbol}] 매도(Short) 주문 실행 (레버리지 5x)")
        return session.place_order(
            category="linear",
            symbol=symbol,
            side="Sell",
            orderType="Market",
            qty=trade_qty
        )
    else:
        print("관망 중 (포지션 유지 또는 대기)...")

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    TARGET_SYMBOL = "ETHUSDT"
    
    # 프로그램 시작 시 1회 레버리지 및 마진 모드 세팅
    setup_trading_rules(TARGET_SYMBOL)

    # 실행 루프 (예시: 1시간마다 반복)
    while True:
        try:
            print(f"\n--- {TARGET_SYMBOL} 시장 분석 시작 ---")
            analysis_result = get_market_analysis(TARGET_SYMBOL)
            
            # Gemini 응답에서 마크다운(```json) 제거 및 파싱
            clean_result = re.sub(r"```json\n|\n```|```", "", analysis_result).strip()
            data = json.loads(clean_result)
            
            decision = data.get("decision", "HOLD")
            reason = data.get("reason", "이유 없음")
            
            print(f"AI 판단: {decision}")
            print(f"분석 이유: {reason}")
            
            # 실제 주문 실행
            execute_trade(TARGET_SYMBOL, decision)
            
        except json.JSONDecodeError:
            print(f"JSON 파싱 실패. AI 응답 포맷이 맞지 않습니다.\n원본: {analysis_result}")
        except Exception as e:
            print(f"오류 발생: {e}")
            
        print("다음 분석까지 대기 중...")
        time.sleep(3600) # 1시간 대기