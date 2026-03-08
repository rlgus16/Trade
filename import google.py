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

# 2. 트레이딩 규칙 세팅 (Hedge 모드, 교차, 5배)
def setup_trading_rules(symbol="ETHUSDT"):
    try:
        session.switch_position_mode(category="linear", symbol=symbol, mode=3)
        print(f"[{symbol}] 양방향(Hedge) 모드 설정 완료")
    except Exception as e:
        if "Not modified" not in str(e) and "already" not in str(e):
            print(f"포지션 모드 설정 안내: {e}")

    try:
        session.switch_margin_mode(category="linear", symbol=symbol, tradeMode=0, buyLeverage="5", sellLeverage="5")
        print(f"[{symbol}] 교차 마진 및 5배 레버리지 설정 완료")
    except Exception:
        pass

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

# 4. Gemini 시장 분석 및 수량 결정
def get_market_analysis(symbol, long_pos, short_pos):
    kline = session.get_kline(category="linear", symbol=symbol, interval=60, limit=10)
    data_list = kline['result']['list']
    
    # 현재가(가장 최근 캔들의 종가) 추출
    current_price = float(data_list[0][4]) 
    
    # 현재 포지션의 USDT 가치 계산
    long_usdt = long_pos['size'] * current_price
    short_usdt = short_pos['size'] * current_price
    
    model = genai.GenerativeModel('gemini-3.1-pro')
    prompt = f"""
    너는 전문 퀀트 트레이더야. 아래 데이터를 바탕으로 트레이딩 결정을 내려줘.
    
    [현재 상태]
    - 현재 {symbol} 가격: {current_price} USDT
    - Long 포지션: 수량 {long_pos['size']} ETH (약 {long_usdt:.2f} USDT 가치), 평단가 {long_pos['avgPrice']}
    - Short 포지션: 수량 {short_pos['size']} ETH (약 {short_usdt:.2f} USDT 가치), 평단가 {short_pos['avgPrice']}
    
    [트레이딩 절대 규칙]
    1. 절대 손절(Stop Loss)을 지시하지 마.
    2. 양방향(Hedge) 포지션을 운영 중이야.
    3. Short 포지션의 총 수량은 Long 포지션의 총 수량을 초과할 수 없어.
    4. 현재 물려있는 포지션이 있다면, 평단가를 낮추기 위한 물타기(Averaging down)를 적극적으로 고려해.
    5. Long 포지션의 총 가치는 절대 2000 USDT를 초과할 수 없어. (현재 약 {long_usdt:.2f} USDT 보유 중)
    6. 진입 수량(qty)은 시장 상황과 리스크를 고려하여 네가 직접 결정해. (단위: ETH, 최소 0.01 이상)
    
    [최근 {symbol} 캔들 데이터]
    {str(data_list)}
    
    응답은 반드시 아래 JSON 형식으로만 해줘:
    {{"decision": "OPEN_LONG" 또는 "OPEN_SHORT" 또는 "HOLD", "qty": 0.01, "reason": "이유 요약"}}
    """
    
    response = model.generate_content(prompt)
    return response.text, current_price

# 5. 주문 실행 (안전 장치 포함)
def execute_trade(symbol, decision, ai_qty, long_pos, short_pos, current_price):
    if decision == "HOLD":
        print("관망 중 (포지션 유지 또는 대기)...")
        return

    # AI가 제안한 수량을 숫자로 변환 (기본값 0.01)
    try:
        trade_qty = float(ai_qty)
    except (ValueError, TypeError):
        trade_qty = 0.01
        
    # ETH 거래 최소 수량 및 소수점 둘째 자리 맞춤
    trade_qty = max(0.01, round(trade_qty, 2))

    if decision == "OPEN_LONG":
        current_long_usdt = long_pos['size'] * current_price
        trade_usdt = trade_qty * current_price
        
        # 1. 롱 포지션 2000 USDT 초과 방지 로직
        if current_long_usdt + trade_usdt > 2000:
            max_allowable_eth = (2000 - current_long_usdt) / current_price
            trade_qty = round(max_allowable_eth - 0.005, 2) # 내림 효과로 안전하게 계산
            
            if trade_qty < 0.01:
                print(f"[{symbol}] 롱 포지션 한도(2000 USDT) 도달. 더 이상 매수할 수 없습니다. (현재: {current_long_usdt:.2f} USDT)")
                return None
            print(f"[{symbol}] ⚠️ 롱 포지션 2000 USDT 한도 제한 발동! AI 제안 수량보다 적은 {trade_qty} ETH만 진입합니다.")
            
        print(f"[{symbol}] 롱(Long) 진입 또는 물타기 실행 (수량: {trade_qty} ETH)")
        return session.place_order(category="linear", symbol=symbol, side="Buy", orderType="Market", qty=str(trade_qty), positionIdx=1)
        
    elif decision == "OPEN_SHORT":
        # 2. 숏 수량 초과 방지 로직
        if (short_pos['size'] + trade_qty) > long_pos['size']:
            max_short_qty = long_pos['size'] - short_pos['size']
            trade_qty = round(max_short_qty - 0.005, 2)
            
            if trade_qty < 0.01:
                print(f"[{symbol}] 숏 주문 거부: 숏 수량이 롱 수량을 초과할 수 없습니다.")
                return None
            print(f"[{symbol}] ⚠️ 숏 수량 제한 발동! AI 제안 수량보다 적은 {trade_qty} ETH만 진입합니다.")
            
        print(f"[{symbol}] 숏(Short) 진입 또는 물타기 실행 (수량: {trade_qty} ETH)")
        return session.place_order(category="linear", symbol=symbol, side="Sell", orderType="Market", qty=str(trade_qty), positionIdx=2)

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    TARGET_SYMBOL = "ETHUSDT"
    
    setup_trading_rules(TARGET_SYMBOL)

    while True:
        try:
            print(f"\n--- {TARGET_SYMBOL} 시장 분석 시작 ---")
            
            long_info, short_info = get_positions_info(TARGET_SYMBOL)
            analysis_result, cur_price = get_market_analysis(TARGET_SYMBOL, long_info, short_info)
            
            clean_result = re.sub(r"```json\n|\n```|```", "", analysis_result).strip()
            data = json.loads(clean_result)
            
            decision = data.get("decision", "HOLD")
            ai_qty = data.get("qty", 0.01)
            reason = data.get("reason", "이유 없음")
            
            print(f"AI 판단: {decision} / 제안 수량: {ai_qty} ETH")
            print(f"분석 이유: {reason}")
            
            execute_trade(TARGET_SYMBOL, decision, ai_qty, long_info, short_info, cur_price)
            
        except json.JSONDecodeError:
            print(f"JSON 파싱 실패. AI 응답 포맷 오류.\n원본: {analysis_result}")
        except Exception as e:
            print(f"오류 발생: {e}")
            
        print("다음 분석까지 대기 중...")
        time.sleep(3600) # 1시간 대기