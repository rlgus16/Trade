import ccxt
from google import genai
import time
import json
import pandas as pd
import pandas_ta as ta
import logging

# ==========================================
# 0. 로깅(Logging) 설정 (터미널 & 파일 동시 출력)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("trading_bot.log", encoding='utf-8'),
        logging.StreamHandler()
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
        'positionMode': True
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
        # 바이낸스 계정 자체를 명시적으로 양방향(헤지) 모드로 변경
        exchange.set_position_mode(True)
        exchange.set_leverage(LEVERAGE, SYMBOL)
        exchange.set_margin_mode('cross', SYMBOL)
        logger.info(f"✅ {SYMBOL} 셋업 완료: {LEVERAGE}x 레버리지, 교차(Cross) 마진, 양방향(Hedge) 모드")
    except Exception as e:
        logger.warning(f"⚠️ 셋업 경고 (이미 설정되어 있을 수 있음): {e}")

def get_account_state():
    balance = exchange.fetch_balance()
    free_usdt = balance['USDT']['free']
    
    positions = exchange.fetch_positions([SYMBOL])
    
    long_size = 0.0; long_price = 0.0; long_pnl = 0.0; long_contracts = 0.0
    short_size = 0.0; short_price = 0.0; short_pnl = 0.0; short_contracts = 0.0
    
    for pos in positions:
        contracts = float(pos.get('contracts', 0))
        entry_price = float(pos.get('entryPrice', 0))
        
        if pos['side'] == 'long':
            long_contracts = contracts 
            long_size = contracts * entry_price 
            long_price = entry_price
            long_pnl = float(pos.get('unrealizedPnl', 0.0))
        elif pos['side'] == 'short':
            short_contracts = contracts
            short_size = abs(contracts * entry_price)
            short_price = entry_price
            short_pnl = float(pos.get('unrealizedPnl', 0.0))
            
    return free_usdt, long_size, long_price, long_pnl, short_size, short_price, short_pnl, long_contracts, short_contracts

def get_market_data():
    """현재가, 기술적 지표, 그리고 현재 펀딩비를 계산/조회합니다."""
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    try:
        funding_info = exchange.fetch_funding_rate(SYMBOL)
        funding_rate = float(funding_info['fundingRate']) * 100 
    except Exception as e:
        logger.warning(f"⚠️ 펀딩비 조회 실패: {e}")
        funding_rate = 0.0
    
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='4h', limit=200)
    
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
        
    return current_price, recent_data, funding_rate

# ==========================================
# 3. Gemini 3.1 Pro 시그널 분석 함수
# ==========================================
def get_gemini_signal(free_usdt, long_size, long_price, long_pnl, short_size, short_price, short_pnl, current_price, recent_data, funding_rate):
    system_instruction = """
    너는 바이낸스 선물 시장에서 활동하는 최상위 퀀트 트레이더야. 
    다음 [거래 규칙]을 엄격히 지켜서 판단해.
    1. 어떠한 상황에서도 손실 확정(Stop-Loss)을 하지 마라. (물타기로 대응)
    2. 총 롱 포지션 규모는 2000 USDT를 초과할 수 없다.
    3. 숏 포지션 규모는 현재 보유 중인 롱 포지션 규모를 초과할 수 없다.
    4. 제공된 기술적 지표(RSI, MACD, 볼린저 밴드, 이동평균선)를 철저히 분석하여 추세와 과매수/과매도 구간 파악해라.
    5. 현재 미실현 손익(PnL) 상태와 펀딩비(Funding Rate)를 고려하여, 물타기 타점과 헤징(Hedging) 시점을 영리하게 계산해라.
    6. [매우 중요] 롱 포지션을 청산(CLOSE_LONG)할 때, 청산 후 남은 롱 포지션의 규모가 숏 포지션의 규모보다 작아지면 절대 안 된다. (숏 포지션의 무한 손실을 방어하기 위한 필수 헤징 유지)
    7. 수익 실현이 필요할 경우 CLOSE_LONG 또는 CLOSE_SHORT 액션을 사용하여 보유 중인 포지션을 청산해라.
    8. 특정 가격을 예측하여 지정가(Limit) 주문을 걸어둔다. 진입(LONG/SHORT) 시에는 지지/저항선을 분석하여 매복할 대기 가격을 설정하고, 익절 시에는 목표 수익 가격을 설정해라. 당장 체결되는 것이 유리하다고 판단되면 현재가와 비슷하게 설정해라.
    
    반드시 아래 JSON 형식으로만 응답해. 마크다운이나 다른 텍스트는 금지.
    {"action": "LONG" | "SHORT" | "CLOSE_LONG" | "CLOSE_SHORT" | "HOLD", "target_price": 예측한주문목표가격(숫자), "amount_usdt": 진입또는청산금액(USDT숫자), "reasoning": "기술적 지표 및 PnL/펀딩비를 근거로 한 상세한 이유"}
    """
    
    prompt = f"""
    {system_instruction}
    
    [현재 계좌 상태]
    - 잔고(Free USDT): {free_usdt}
    - 현재 Long 포지션: {long_size} USDT (평단가: {long_price}, 미실현손익: {long_pnl} USDT)
    - 현재 Short 포지션: {short_size} USDT (평단가: {short_price}, 미실현손익: {short_pnl} USDT)
    - 현재 펀딩비(Funding Rate): {funding_rate}%
    
    [시장 데이터: {SYMBOL}]
    - 현재가: {current_price}
    - 최근 4시간봉 데이터 및 기술적 지표: {json.dumps(recent_data, indent=2)}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
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
            
            try:
                exchange.cancel_all_orders(SYMBOL)
                logger.info(f"🧹 {SYMBOL} 미체결 주문 정리 완료")
            except Exception as e:
                logger.warning(f"⚠️ 미체결 주문 취소 중 오류 (무시 가능): {e}")

            free_usdt, long_size, long_price, long_pnl, short_size, short_price, short_pnl, long_contracts, short_contracts = get_account_state()
            current_price, recent_data, funding_rate = get_market_data()
            
            logger.info(f"💰 현재가: {current_price} | Funding: {funding_rate:.4f}%")
            logger.info(f"📊 LONG: {long_size} USDT (PnL: {long_pnl:.2f}) | SHORT: {short_size} USDT (PnL: {short_pnl:.2f}) | FREE: {free_usdt:.2f} USDT")
            
            logger.info("🧠 Gemini 3.1 Pro-preview 분석 중...")
            signal = get_gemini_signal(free_usdt, long_size, long_price, long_pnl, short_size, short_price, short_pnl, current_price, recent_data, funding_rate)
            
            action = signal.get('action')
            amount_usdt = float(signal.get('amount_usdt', 0))
            
            # AI가 예측한 목표가(target_price) 받아오기 (오류 시 현재가로 방어)
            target_price = float(signal.get('target_price', current_price))
            if target_price <= 0:
                target_price = current_price
                
            reason = signal.get('reasoning')
            
            logger.info(f"🔔 시그널: {action} | 목표 지정가: {target_price} | 요청 금액: {amount_usdt} USDT | 사유: {reason}")
            
            # 🛡️ 안전장치 1: 바이낸스 최소 주문 금액 (5 USDT) 방어 (진입/청산 모두 적용)
            if action in ["LONG", "SHORT", "CLOSE_LONG", "CLOSE_SHORT"] and amount_usdt > 0:
                if amount_usdt < 5.0:
                    logger.warning(f"🛡️ 방어 로직 작동: 금액({amount_usdt} USDT)이 바이낸스 최소 한도 미만입니다. 관망(HOLD)으로 전환합니다.")
                    action = "HOLD"
                    amount_usdt = 0.0

            # 🛡️ 안전장치 2: 가용 증거금 부족 방어 (신규 진입인 LONG, SHORT에만 적용)
            if action in ["LONG", "SHORT"] and amount_usdt > 0:
                required_margin = amount_usdt / LEVERAGE
                if required_margin > free_usdt:
                    logger.warning(f"🛡️ 방어 로직 작동: 가용 잔고 부족! (필요: {required_margin:.2f} > 잔고: {free_usdt:.2f}). 관망(HOLD)으로 전환합니다.")
                    action = "HOLD"
                    amount_usdt = 0.0
            
            raw_order_qty = amount_usdt / target_price if target_price > 0 else 0
            
            # 💡 수량 및 지정가(가격) 정밀도 동시 처리
            if amount_usdt > 0:
                order_qty_str = exchange.amount_to_precision(SYMBOL, raw_order_qty)
                order_qty = float(order_qty_str)
                
                # 💡 지정가 주문이 깔릴 가격을 AI가 예측한 target_price로 세팅
                order_price_str = exchange.price_to_precision(SYMBOL, target_price)
                order_price = float(order_price_str)
            else:
                order_qty = 0.0
                order_price = 0.0
            
            # ==========================================
            # 💡 실제 주문 실행부 (지정가 및 독립 청산 로직 완벽 적용)
            # ==========================================
            if action == "LONG" and amount_usdt > 0:
                if long_size + amount_usdt <= MAX_LONG_USDT:
                    logger.info(f"🚀 롱 포지션 진입/추가 (수량: {order_qty} LTC | 지정가: {order_price} USDT)")
                    exchange.create_order(SYMBOL, 'limit', 'buy', order_qty, order_price, params={'positionSide': 'LONG'})
                else:
                    logger.warning("⛔ 롱 포지션 한도 초과로 진입 불가.")
                    
            elif action == "SHORT" and amount_usdt > 0:
                if short_size + amount_usdt <= long_size:
                    logger.info(f"📉 숏 포지션 진입/추가 (수량: {order_qty} LTC | 지정가: {order_price} USDT)")
                    exchange.create_order(SYMBOL, 'limit', 'sell', order_qty, order_price, params={'positionSide': 'SHORT'})
                else:
                    logger.warning("⛔ 숏 포지션은 롱 포지션 규모 초과 불가.")
            
            # 💡 롱 포지션 수익 실현/청산 (지정가 매도) + 방패 붕괴 방지(안전장치 4)
            elif action == "CLOSE_LONG" and amount_usdt > 0:
                if long_size > 0:
                    if (long_size - amount_usdt) < short_size:
                        logger.warning(f"🛡️ 헤지 방어 작동: 롱 청산 후 남은 롱이 숏({short_size} USDT)보다 적어집니다! 숏 노출 위험으로 청산 금액을 축소합니다.")
                        amount_usdt = long_size - short_size 
                        
                    # 축소된 청산 금액이 바이낸스 최소 주문 한도(5 USDT) 미만이 되어버렸는지 2차 검사
                    if amount_usdt > 0 and amount_usdt < 5.0:
                        logger.warning("⛔ 방어 로직으로 축소된 청산 금액이 5 USDT 미만이 되어, 바이낸스 최소 한도 에러를 방지하기 위해 청산을 취소합니다.")
                        
                    elif amount_usdt <= 0:
                        logger.warning("⛔ 숏 포지션 방어를 위해 롱 포지션을 더 이상 청산할 수 없습니다. (숏 포지션을 먼저 청산해야 합니다)")
                    else:
                        raw_close_qty = amount_usdt / long_price
                        actual_close_qty = min(raw_close_qty, long_contracts)
                        
                        close_qty_str = exchange.amount_to_precision(SYMBOL, actual_close_qty)
                        final_close_qty = float(close_qty_str)
                        
                        logger.info(f"✅ 롱 포지션 청산 (수량: {final_close_qty} LTC | 지정가: {order_price} USDT)")
                        exchange.create_order(SYMBOL, 'limit', 'sell', final_close_qty, order_price, params={'positionSide': 'LONG'})
                else:
                    logger.warning("⛔ 보유 중인 롱 포지션이 없어 청산 불가.")

            # 💡 숏 포지션 수익 실현/청산 (지정가 매수)
            elif action == "CLOSE_SHORT" and amount_usdt > 0:
                if short_size > 0:
                    raw_close_qty = amount_usdt / short_price
                    actual_close_qty = min(raw_close_qty, short_contracts)
                    
                    close_qty_str = exchange.amount_to_precision(SYMBOL, actual_close_qty)
                    final_close_qty = float(close_qty_str)
                    
                    logger.info(f"✅ 숏 포지션 청산 (수량: {final_close_qty} LTC | 지정가: {order_price} USDT)")
                    exchange.create_order(SYMBOL, 'limit', 'buy', final_close_qty, order_price, params={'positionSide': 'SHORT'})
                else:
                    logger.warning("⛔ 보유 중인 숏 포지션이 없어 청산 불가.")

            else:
                logger.info("⏸️ 관망(HOLD) 또는 조건 불충족 상태 유지.")

        except Exception as e:
            logger.error(f"🚨 시스템/네트워크 에러 발생: {e}")
            logger.info("🛡️ 방어 로직 작동: 60초 대기 후 재시도합니다...")
            time.sleep(60)
            continue 
            
        time.sleep(1800) 

if __name__ == "__main__":
    run_bot()