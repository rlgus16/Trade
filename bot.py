import ccxt
from google import genai
import time
import json
import pandas as pd
import pandas_ta as ta
import logging # 💡 새로 추가된 로깅 모듈

# ==========================================
# 0. 로깅(Logging) 설정 (터미널 & 파일 동시 출력)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("trading_bot.log", encoding='utf-8'), # 텍스트 파일로 저장
        logging.StreamHandler() # 터미널에도 출력
    ]
)
logger = logging.getLogger()

# ==========================================
# 1. API 키 설정 (본인의 키로 변경하세요)
# ==========================================
BINANCE_API_KEY = 'YOUR_BINANCE_API_KEY'
BINANCE_SECRET = 'YOUR_BINANCE_SECRET'
GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY'

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
        exchange.load_markets() 
        exchange.set_leverage(LEVERAGE, SYMBOL)
        exchange.set_margin_mode('cross', SYMBOL)
        logger.info(f"✅ {SYMBOL} 셋업 완료: {LEVERAGE}x 레버리지, 교차(Cross) 마진")
    except Exception as e:
        logger.warning(f"⚠️ 셋업 경고 (이미 설정되어 있을 수 있음): {e}")

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
    """현재가 및 최근 캔들을 가져와 기술적 지표를 계산합니다."""
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='4h', limit=100)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    df.ta.rsi(length=14, append=True)           
    df.ta.macd(append=True)                     
    df.ta.bbands(length=20, std=2, append=True) 
    df.ta.sma(length=20, append=True)           
    
    df.fillna(0, inplace=True)
    
    recent_data = df.tail(3).to_dict(orient='records')
    for row in recent_data:
        row['timestamp'] = str(row['timestamp'])
        
    return current_price, recent_data

# ==========================================
# 3. Gemini 3.1 Pro 시그널 분석 함수
# ==========================================
def get_gemini_signal(free_usdt, long_size, long_price, short_size, short_price, current_price, recent_data):
    system_instruction = """
    너는 바이낸스 선물 시장에서 활동하는 최상위 퀀트 트레이더야. 
    다음 [거래 규칙]을 엄격히 지켜서 판단해.
    1. 어떠한 상황에서도 손실 확정(Stop-Loss)을 하지 마라. (물타기로 대응)
    2. 총 롱 포지션 규모는 2000 USDT를 초과할 수 없다.
    3. 숏 포지션 규모는 현재 보유 중인 롱 포지션 규모를 초과할 수 없다.
    4. 제공된 기술적 지표(RSI, MACD, 볼린저 밴드, 이동평균선)를 철저히 분석하여 추세와 과매수/과매도 구간을 파악해라.
    
    반드시 아래 JSON 형식으로만 응답해. 마크다운이나 다른 텍스트는 금지.
    {"action": "LONG" | "SHORT" | "HOLD", "amount_usdt": 진입금액(USDT숫자), "reasoning": "기술적 지표에 근거한 구체적인 진입/관망 이유"}
    """
    
    prompt = f"""
    {system_instruction}
    
    [현재 계좌 상태]
    - 잔고(Free USDT): {free_usdt}
    - 현재 Long 포지션: {long_size} USDT (평단가: {long_price})
    - 현재 Short 포지션: {short_size} USDT (평단가: {short_price})
    
    [시장 데이터: {SYMBOL}]
    - 현재가: {current_price}
    - 최근 4시간봉 데이터 및 기술적 지표: {json.dumps(recent_data, indent=2)}
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
        logger.error(f"⚠️ Gemini 분석 오류: {e}")
        return {"action": "HOLD", "amount_usdt": 0, "reasoning": f"Error: {e}"}

# ==========================================
# 4. 메인 거래 실행 로직
# ==========================================
def run_bot():
    setup_exchange()
    
    while True:
        try:
            logger.info("="*50)
            free_usdt, long_size, long_price, short_size, short_price = get_account_state()
            current_price, recent_data = get_market_data()
            
            logger.info(f"💰 현재가: {current_price} | Long: {long_size} USDT | Short: {short_size} USDT")
            
            logger.info("🧠 Gemini 3.1 Pro 분석 중...")
            signal = get_gemini_signal(free_usdt, long_size, long_price, short_size, short_price, current_price, recent_data)
            
            action = signal.get('action')
            amount_usdt = float(signal.get('amount_usdt', 0))
            reason = signal.get('reasoning')
            
            logger.info(f"🔔 시그널: {action} | 요청 금액: {amount_usdt} USDT | 사유: {reason}")
            
            raw_order_qty = amount_usdt / current_price if current_price > 0 else 0
            
            if amount_usdt > 0:
                order_qty_str = exchange.amount_to_precision(SYMBOL, raw_order_qty)
                order_qty = float(order_qty_str)
            else:
                order_qty = 0.0
            
            if action == "LONG" and amount_usdt > 0:
                if long_size + amount_usdt <= MAX_LONG_USDT:
                    logger.info(f"🚀 롱 포지션 진입/추가 (정제된 수량: {order_qty} LTC)")
                    # exchange.create_order(SYMBOL, 'market', 'buy', order_qty, params={'positionSide': 'LONG'})
                else:
                    logger.warning("⛔ 롱 포지션 한도(2000 USDT) 초과로 진입 불가.")
                    
            elif action == "SHORT" and amount_usdt > 0:
                if short_size + amount_usdt <= long_size:
                    logger.info(f"📉 숏 포지션 진입/추가 (정제된 수량: {order_qty} LTC)")
                    # exchange.create_order(SYMBOL, 'market', 'sell', order_qty, params={'positionSide': 'SHORT'})
                else:
                    logger.warning("⛔ 숏 포지션은 롱 포지션 규모를 초과할 수 없어 진입 불가.")
            
            else:
                logger.info("⏸️ 관망(HOLD) 또는 조건 불충족 상태 유지.")

        except Exception as e:
            logger.error(f"에러 발생: {e}")
            
        time.sleep(300) 

if __name__ == "__main__":
    run_bot()