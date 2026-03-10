import ccxt
from google import genai
import time
import json
import pandas as pd
import pandas_ta as ta
import logging

from logging.handlers import RotatingFileHandler

# ==========================================
# 0. 로깅(Logging) 설정 (터미널 & 파일 동시 출력)
# ==========================================
# 최대 5MB 크기로 제한하고, 최근 10개의 파일만 유지합니다.
file_handler = RotatingFileHandler(
    "trading_bot.log", 
    maxBytes=5 * 1024 * 1024,
    backupCount=10,
    encoding='utf-8'
)
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[file_handler, stream_handler]
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
    
    # 라이브러리 버전에 따른 이름 차이 에러를 막기 위해 동적으로 컬럼명 탐색
    rsi_col = next(c for c in df.columns if c.startswith('RSI'))
    macd_col = next(c for c in df.columns if c.startswith('MACD_'))
    macds_col = next(c for c in df.columns if c.startswith('MACDs_'))
    bbu_col = next(c for c in df.columns if c.startswith('BBU'))
    bbl_col = next(c for c in df.columns if c.startswith('BBL'))
    sma_col = next(c for c in df.columns if c.startswith('SMA'))
    
    recent_data = []
    for _, row in df.tail(6).iterrows():
        recent_data.append({
            "t": str(row['timestamp'])[-8:],
            "o": row['open'], "h": row['high'], "l": row['low'], "c": row['close'], "v": row['volume'],
            "rsi": row[rsi_col], "m": row[macd_col], "ms": row[macds_col],
            "bb_u": row[bbu_col], "bb_l": row[bbl_col], "sma": row[sma_col]
        })
        
    return current_price, recent_data, funding_rate

# ==========================================
# 3. Gemini 3.1 Pro 시그널 분석 함수
# ==========================================
def get_gemini_signal(free_usdt, long_size, long_price, long_pnl, short_size, short_price, short_pnl, current_price, recent_data, funding_rate):
    system_instruction = """
    1. NEVER execute a STOP_LOSS.
    2. You can average_down to maximize profit.
    3. The total long_position_size must not exceed 2000 USDT.
    4. Short_size must never exceed long_size at all times.
    5. Thoroughly analyze the provided technical indicators to identify trends.
    6. Do not hedge if free_balance is abundant.
    7. Focus on maximizing profit, rather than hedging.
    8. Open both LONG and SHORT position to maximize profit.
    9. Use CLOSE_LONG or CLOSE_SHORT actions to realize profit.
    10. Predict and place specific limit_order_prices for entries. 
    11. Predict and place take_profit_prices upon new entries.
    
    Respond ONLY in this JSON:
    {"act": "L"|"S"|"CL"|"CS"|"H", "ep": <entry_price>, "tp": <take_profit_price>, "amt": <usdt>, "rsn": "<reasoning>"}
    (L:LONG, S:SHORT, CL:CLOSE_LONG, CS:CLOSE_SHORT, H:HOLD. Use 'ep' for limit entry/close price, and 'tp' for take profit price when entering.)
    """
    
    prompt = f"""
    {system_instruction}
    
    [Current Account State]
    - Free Balance: {free_usdt} USDT
    - Current LONG Position: {long_size} USDT (Entry Price: {long_price}, Unrealized PnL: {long_pnl} USDT)
    - Current SHORT Position: {short_size} USDT (Entry Price: {short_price}, Unrealized PnL: {short_pnl} USDT)
    - Current Funding Rate: {funding_rate}%
    
    [Market Data: {SYMBOL}]
    - Current Price: {current_price}
    - Recent 4H timeframe data & Technical Indicators: {json.dumps(recent_data, separators=(',', ':'))}
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
        return {"act": "H", "amt": 0, "rsn": f"Error: {e}"}

# ==========================================
# 4. 메인 거래 실행 로직
# ==========================================
def run_bot():
    setup_exchange()
    
    while True:
        try:
            logger.info("="*50)
            
            try:
                # 일반 진입 지정가(limit) 주문만 취소하고 Take Profit 주문은 유지합니다.
                open_orders = exchange.fetch_open_orders(SYMBOL)
                cancel_count = 0
                for order in open_orders:
                    if order['type'] == 'limit':
                        exchange.cancel_order(order['id'], SYMBOL)
                        cancel_count += 1
                if cancel_count > 0:
                    logger.info(f"🧹 {SYMBOL} 미체결 진입 주문 {cancel_count}건 정리 완료 (TP 주문 유지)")
            except Exception as e:
                logger.warning(f"⚠️ 미체결 주문 취소 중 오류 (무시 가능): {e}")

            free_usdt, long_size, long_price, long_pnl, short_size, short_price, short_pnl, long_contracts, short_contracts = get_account_state()
            current_price, recent_data, funding_rate = get_market_data()
            
            logger.info(f"💰 현재가: {current_price} | Funding: {funding_rate:.4f}%")
            logger.info(f"📊 LONG: {long_size} USDT (PnL: {long_pnl:.2f}) | SHORT: {short_size} USDT (PnL: {short_pnl:.2f}) | FREE: {free_usdt:.2f} USDT")
            
            logger.info("🧠 Gemini 3.1 Pro-preview 분석 중...")
            signal = get_gemini_signal(free_usdt, long_size, long_price, long_pnl, short_size, short_price, short_pnl, current_price, recent_data, funding_rate)
            
            action_map = {"L": "LONG", "S": "SHORT", "CL": "CLOSE_LONG", "CS": "CLOSE_SHORT", "H": "HOLD"}
            action = action_map.get(signal.get('act'), "HOLD")
            amount_usdt = float(signal.get('amt', 0))
            
            # 진입가(ep) 및 익절가(tp) 받아오기
            order_price_raw = float(signal.get('ep', current_price))
            if order_price_raw <= 0:
                order_price_raw = current_price
                
            tp_price_raw = float(signal.get('tp', 0.0))
            reason = signal.get('rsn')
            
            logger.info(f"🔔 시그널: {action} | 진입 지정가: {order_price_raw} | 목표가(TP): {tp_price_raw} | 요청 금액: {amount_usdt} USDT | 사유: {reason}")
            
            # ==========================================
            # 💡 1차 방어: 신규 진입 시 '전면 거절' 대신 '부분 진입(금액 축소)'으로 유연성 확보
            # ==========================================
            if action == "LONG" and amount_usdt > 0:
                if long_size + amount_usdt > MAX_LONG_USDT:
                    available_space = MAX_LONG_USDT - long_size
                    if available_space >= 5.0:
                        logger.info(f"🔄 롱 한도 초과 감지: 요청 금액을 남은 한도({available_space:.2f} USDT)로 축소하여 부분 진입합니다.")
                        amount_usdt = available_space
                    else:
                        logger.warning("⛔ 롱 포지션이 최대 한도에 도달하여 추가 진입 불가.")
                        action = "HOLD"; amount_usdt = 0.0
                        
            elif action == "SHORT" and amount_usdt > 0:
                if short_size + amount_usdt > long_size:
                    available_space = long_size - short_size
                    if available_space >= 5.0:
                        logger.info(f"🔄 숏 한도 초과 감지: 숏은 롱 규모를 넘을 수 없어 남은 한도({available_space:.2f} USDT)로 축소 진입합니다.")
                        amount_usdt = available_space
                    else:
                        logger.warning("⛔ 숏 포지션이 롱 규모와 동일하여 추가 진입 불가.")
                        action = "HOLD"; amount_usdt = 0.0

            # 🛡️ 2차 방어: 가용 증거금 부족 시에도 전면 거절 대신 '가용 한도 내 최대 진입'
            if action in ["LONG", "SHORT"] and amount_usdt > 0:
                required_margin = amount_usdt / LEVERAGE
                if required_margin > free_usdt:
                    max_possible_usdt = free_usdt * LEVERAGE * 0.95 # 수수료 및 슬리피지 여유분 5% 차감
                    if max_possible_usdt >= 5.0:
                        logger.warning(f"🔄 잔고 부족 감지: 요청 금액을 최대 가용 금액({max_possible_usdt:.2f} USDT)으로 축소 진입합니다.")
                        amount_usdt = max_possible_usdt
                    else:
                        logger.warning("⛔ 가용 잔고가 너무 부족하여 진입할 수 없습니다. 관망(HOLD) 전환.")
                        action = "HOLD"; amount_usdt = 0.0

            # 🛡️ 3차 방어: 바이낸스 최소 주문 금액 (5 USDT) 1차 공통 확인
            if action in ["LONG", "SHORT", "CLOSE_LONG", "CLOSE_SHORT"] and amount_usdt > 0:
                if amount_usdt < 5.0:
                    logger.warning(f"🛡️ 최종 요청 금액({amount_usdt:.2f} USDT)이 바이낸스 최소 한도 미만입니다. 관망(HOLD) 전환.")
                    action = "HOLD"
                    amount_usdt = 0.0
            
            # ==========================================
            # 💡 조절이 완료된 최종 금액으로 수량(qty) 및 가격(price) 정밀도 계산
            # ==========================================
            raw_order_qty = amount_usdt / order_price_raw if order_price_raw > 0 else 0
            
            if amount_usdt > 0:
                order_qty_str = exchange.amount_to_precision(SYMBOL, raw_order_qty)
                order_qty = float(order_qty_str)
                order_price_str = exchange.price_to_precision(SYMBOL, order_price_raw)
                order_price = float(order_price_str)
                tp_price = float(exchange.price_to_precision(SYMBOL, tp_price_raw)) if tp_price_raw > 0 else 0.0
            else:
                order_qty = 0.0
                order_price = 0.0
                tp_price = 0.0
            
            # ==========================================
            # 💡 실제 주문 실행부
            # ==========================================
            if action == "LONG" and amount_usdt > 0:
                logger.info(f"🚀 롱 포지션 진입/추가 (수량: {order_qty} LTC | 지정가: {order_price} USDT)")
                exchange.create_order(SYMBOL, 'limit', 'buy', order_qty, order_price, params={'positionSide': 'LONG'})
                
                # Take Profit 주문 추가 (롱: 목표가가 진입가보다 높을 때만)
                if tp_price > order_price:
                    logger.info(f"🎯 롱 포지션 익절(TP) 예약 (목표가: {tp_price} USDT)")
                    exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell', order_qty, params={'positionSide': 'LONG', 'stopPrice': tp_price})
                    
            elif action == "SHORT" and amount_usdt > 0:
                logger.info(f"📉 숏 포지션 진입/추가 (수량: {order_qty} LTC | 지정가: {order_price} USDT)")
                exchange.create_order(SYMBOL, 'limit', 'sell', order_qty, order_price, params={'positionSide': 'SHORT'})
                
                # Take Profit 주문 추가 (숏: 목표가가 진입가보다 낮을 때만)
                if tp_price > 0 and tp_price < order_price:
                    logger.info(f"🎯 숏 포지션 익절(TP) 예약 (목표가: {tp_price} USDT)")
                    exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'buy', order_qty, params={'positionSide': 'SHORT', 'stopPrice': tp_price})
            
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
                        # 전량 청산 의도시 먼지(Dust) 방지
                        if amount_usdt >= long_size * 0.99:
                            logger.info("🧹 전량 청산: 잔류 방지를 위해 보유 롱 수량 전체를 매도합니다.")
                            actual_close_qty = long_contracts
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
                    # 전량 청산 의도시 먼지(Dust) 방지
                    if amount_usdt >= short_size * 0.99:
                        logger.info("🧹 전량 청산: 잔류 방지를 위해 보유 숏 수량 전체를 매수합니다.")
                        actual_close_qty = short_contracts
                    else:
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
            logger.info("🛡️ 방어 로직 작동: 10분 대기 후 재시도합니다...")
            time.sleep(600)
            continue 
            
        time.sleep(1800) 

if __name__ == "__main__":
    try:
        logger.info("🚀 트레이딩 봇 가동을 시작합니다...")
        run_bot()
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 봇이 안전하게 종료되었습니다 (Ctrl+C).")
    except Exception as e:
        logger.critical(f"💥 치명적인 오류로 봇이 종료되었습니다: {e}")