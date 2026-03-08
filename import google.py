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

# 2. 트레이딩 규칙 세팅 (Hedge 모드 추가)
def setup_trading_rules(symbol="ETHUSDT"):
    try:
        # 양방향 포지션(Hedge Mode) 설정 (mode=3)
        session.switch_position_mode(
            category="linear",
            symbol=symbol,
            mode=3
        )
        print(f"[{symbol}] 양방향(Hedge) 모드 설정 완료")
    except Exception as e:
        if "Not modified" in str(e) or "already" in str(e):
            print(f"[{symbol}] 이미 양방향(Hedge) 모드로 설정되어 있습니다.")
        else:
            print(f"포지션 모드 설정 안내: {e}")

    try:
        # 교차 마진(Cross) 및 5배 레버리지 설정
        session.switch_margin_mode(
            category="linear",
            symbol=symbol,
            tradeMode=0, 
            buyLeverage="5",
            sellLeverage="5"
        )
        print(f"[{symbol}] 교차 마진 및 5배 레버리지 설정 완료")
    except Exception as e:
        pass # 이미 설정된 경우 패스

# 3. 현재 포지션 정보 가져오기
def get_positions_info(symbol):
    positions = session.get_positions(category="linear", symbol=symbol)['result']['list']
    long_pos = {"size": 0.0, "avgPrice": 0.0}
    short_pos = {"size": 0.0, "avgPrice": 0.0}
    
    for p in positions:
        if p['positionIdx'] == 1: # Long
            long_pos['size'] = float(p['size'])
            long_pos['avgPrice'] = float(p['avgPrice'])
        elif p['positionIdx'] == 2: # Short
            short_pos['size'] = float(p['size'])
            short_pos['avgPrice'] = float(p['avgPrice'])
            
    return long_pos, short_pos

# 4. Gemini 시장 분석 및 결정
def get_market_analysis(symbol, long_pos, short_pos):
    kline = session.get_kline(
        category="linear",
        symbol=symbol,
        interval=60,
        limit=10
    )
    data_str = str(kline['result']['list'])
    
    model = genai.GenerativeModel('gemini-3.1-pro')
    prompt = f"""
    너는 전문 퀀트 트레이더야. 아래 데이터를 바탕으로 트레이딩 결정을 내려줘.
    
    [현재 포지션 상태]
    - Long 포지션: 수량 {long_pos['size']}, 평단가 {long_pos['avgPrice']}
    - Short 포지션: 수량 {short_pos['size']}, 평단가 {short_pos['avgPrice']}
    
    [트레이딩 절대 규칙]
    1. 절대 손절(Stop Loss)을 지시하지 마.
    2. 양방향(Hedge) 포지션을 운영 중이야.
    3. Short 포지션의 총 수량은 Long 포지션의 총 수량을 초과할 수 없어.
    4. 현재 물려있는 포지션이 있다면, 평단가를 낮추기 위한 물타기(Averaging down)를 적극적으로 고려해.
    
    [최근 {symbol} 캔들 데이터]
    {data_str}
    
    응답은 반드시 아래 JSON 형식으로만 해줘:
    {{"decision": "OPEN_LONG" 또는 "OPEN_SHORT" 또는 "HOLD", "reason": "이유 요약"}}
    """
    
    response = model.generate_content(prompt)
    return response.text

# 5. 주문 실행
def execute_trade(symbol, decision, long_pos, short_pos):
    trade_qty = 0.01 # 1회 거래 수량 (필요시 조정)

    if decision == "OPEN_LONG":
        print(f"[{symbol}] 롱(Long) 진입 또는 물타기 실행")
        return session.place_order(
            category="linear",
            symbol=symbol,
            side="Buy",
            orderType="Market",
            qty=str(trade_qty),
            positionIdx=1 # 1 = Long
        )
        
    elif decision == "OPEN_SHORT":
        # 파이썬 레벨에서 숏 수량 초과 방지 이중 체크
        if (short_pos['size'] + trade_qty) > long_pos['size']:
            print(f"[{symbol}] 숏 주문 거부: 숏 수량({short_pos['size'] + trade_qty})이 롱 수량({long_pos['size']})을 초과할 수 없습니다.")
            return None
            
        print(f"[{symbol}] 숏(Short) 진입 또는 물타기 실행")
        return session.place_order(
            category="linear",
            symbol=symbol,
            side="Sell",
            orderType="Market",
            qty=str(trade_qty),
            positionIdx=2 # 2 = Short
        )
    else:
        print("관망 중 (포지션 유지 또는 대기)...")

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    TARGET_SYMBOL = "ETHUSDT"
    
    setup_trading_rules(TARGET_SYMBOL)

    while True:
        try:
            print(f"\n--- {TARGET_SYMBOL} 시장 분석 시작 ---")
            
            # 1. 현재 포지션 파악
            long_info, short_info = get_positions_info(TARGET_SYMBOL)
            
            # 2. AI 분석 요청
            analysis_result = get_market_analysis(TARGET_SYMBOL, long_info, short_info)
            
            # 3. 결과 파싱
            clean_result = re.sub(r"```json\n|\n```|```", "", analysis_result).strip()
            data = json.loads(clean_result)
            
            decision = data.get("decision", "HOLD")
            reason = data.get("reason", "이유 없음")
            
            print(f"AI 판단: {decision}")
            print(f"분석 이유: {reason}")
            
            # 4. 주문 실행 (손절은 절대 하지 않음)
            execute_trade(TARGET_SYMBOL, decision, long_info, short_info)
            
        except json.JSONDecodeError:
            print(f"JSON 파싱 실패. AI 응답 포맷 오류.\n원본: {analysis_result}")
        except Exception as e:
            print(f"오류 발생: {e}")
            
        print("다음 분석까지 대기 중...")
        time.sleep(3600) # 1시간 대기