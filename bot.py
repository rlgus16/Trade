import ccxt
from google import genai
import time
import json

# ==========================================
# 1. API 키 설정 (본인의 키로 변경하세요)
# ==========================================
BINANCE_API_KEY = 'YOUR_BINANCE_API_KEY'
BINANCE_SECRET = 'YOUR_BINANCE_SECRET'
GEMINI_API_KEY = 'AIzaSyCnojY9u_hQp9fKqDQdd1YVL57N88FCc4M'

client = genai.Client(api_key=GEMINI_API_KEY)

# 바이낸스 선물 객체 생성 (헤지 모드 켜기)
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'positionMode': True # 양방향 거래를 위한 헤지 모드 활성화
    }
})

SYMBOL = 'LTC/USDT'
MAX_LONG_USDT = 2000
LEVERAGE = 5

# ==========================================
# 2. 초기 셋업 및 데이터 수집 함수
# ==========================================
def setup_exchange():
    """레버리지 및 마진 모드(교차) 설정"""
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
        exchange.set_margin_mode('cross', SYMBOL)
        print(f"✅ {SYMBOL} 셋업 완료: {LEVERAGE}x 레버리지, 교차(Cross) 마진")
    except Exception as e:
        print(f"⚠️ 셋업 경고 (이미 설정되어 있을 수 있음): {e}")

def get_account_state():
    """현재 포지션 및 잔고 조회"""
    balance = exchange.fetch_balance()
    free_usdt = balance['USDT']['free']
    
    positions = exchange.fetch_positions([SYMBOL])
    long_size = 0.0; long_price = 0.0
    short_size = 0.0; short_price = 0.0
    
    for pos in positions:
        if pos['side'] == 'long':
            long_size = float(pos['notional'])
            long_price = float(pos['entryPrice'])
        elif pos['side'] == 'short':
            short_size = abs(float(pos['notional']))
            short_price = float(pos['entryPrice'])
            
    return free_usdt, long_size, long_price, short_size, short_price

def get_market_data():
    """현재가 및 최근 캔들 데이터 조회"""
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    # 최근 15분봉 5개 가져오기
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='15m', limit=5)
    candles = [{"time": exchange.iso8601(c[0]), "open": c[1], "high": c[2], "low": c[3], "close": c[4]} for c in ohlcv]
    
    return current_price, candles

# ==========================================
# 3. Gemini 3.1 Pro 시그널 분석 함수
# ==========================================
def get_gemini_signal(free_usdt, long_size, long_price, short_size, short_price, current_price, candles):
    system_instruction = """
    너는 바이낸스 선물 시장에서 활동하는 퀀트 트레이더야. 
    다음 [거래 규칙]을 엄격히 지켜서 판단해.
    1. 어떠한 상황에서도 손실 확정(Stop-Loss)을 하지 마라. (물타기로 대응)
    2. 총 롱 포지션 규모는 2000 USDT를 초과할 수 없다.
    3. 숏 포지션 규모는 현재 보유 중인 롱 포지션 규모를 초과할 수 없다.
    
    반드시 아래 JSON 형식으로만 응답해. 마크다운이나 다른 텍스트는 금지.
    {"action": "LONG" | "SHORT" | "HOLD", "amount_usdt": 진입금액(USDT숫자), "reasoning": "이유 설명"}
    """
    
    prompt = f"""
    {system_instruction}
    
    [현재 계좌 상태]
    - 잔고(Free USDT): {free_usdt}
    - 현재 Long 포지션: {long_size} USDT (평단가: {long_price})
    - 현재 Short 포지션: {short_size} USDT (평단가: {short_price})
    
    [시장 데이터: {SYMBOL}]
    - 현재가: {current_price}
    - 최근 15분봉 데이터: {json.dumps(candles)}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro',
            contents=prompt
        )
        
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        signal_data = json.loads(clean_text)
        return signal_data
    except Exception as e:
        print(f"⚠️ Gemini 분석 오류: {e}")
        return {"action": "HOLD", "amount_usdt": 0, "reasoning": "Error parsing Gemini response"}

# ==========================================
# 4. 메인 거래 실행 로직
# ==========================================
def run_bot():
    setup_exchange()
    
    while True:
        try:
            print("\n" + "="*40)
            free_usdt, long_size, long_price, short_size, short_price = get_account_state()
            current_price, candles = get_market_data()
            
            print(f"💰 현재가: {current_price} | Long: {long_size} USDT | Short: {short_size} USDT")
            
            # Gemini에게 시그널 요청
            print("🧠 Gemini 3.1 Pro 분석 중...")
            signal = get_gemini_signal(free_usdt, long_size, long_price, short_size, short_price, current_price, candles)
            
            action = signal.get('action')
            amount_usdt = float(signal.get('amount_usdt', 0))
            reason = signal.get('reasoning')
            
            print(f"🔔 시그널: {action} | 요청 금액: {amount_usdt} USDT | 사유: {reason}")
            
            # 주문 수량(LTC) 계산
            order_qty = amount_usdt / current_price
            
            # 룰 검증 및 주문 실행
            if action == "LONG" and amount_usdt > 0:
                if long_size + amount_usdt <= MAX_LONG_USDT:
                    print(f"🚀 롱 포지션 진입/추가 (수량: {order_qty} LTC)")
                    # 실제 주문 코드 (안전을 위해 우선 주석 처리해 두었습니다. 테스트 후 주석 해제하세요)
                    # exchange.create_order(SYMBOL, 'market', 'buy', order_qty, params={'positionSide': 'LONG'})
                else:
                    print("⛔ 롱 포지션 한도(2000 USDT) 초과로 진입 불가.")
                    
            elif action == "SHORT" and amount_usdt > 0:
                if short_size + amount_usdt <= long_size:
                    print(f"📉 숏 포지션 진입/추가 (수량: {order_qty} LTC)")
                    # 실제 주문 코드
                    # exchange.create_order(SYMBOL, 'market', 'sell', order_qty, params={'positionSide': 'SHORT'})
                else:
                    print("⛔ 숏 포지션은 롱 포지션 규모를 초과할 수 없어 진입 불가.")
            
            else:
                print("⏸️ 관망(HOLD) 또는 조건 불충족 상태 유지.")

        except Exception as e:
            print(f"에러 발생: {e}")
            
        # 5분마다 반복 실행 (원하는 주기로 변경 가능)
        time.sleep(300) 

if __name__ == "__main__":
    # 봇 실행 시작
    run_bot()