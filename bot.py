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
    3. The total long_size must not exceed 2000 USDT.
    4. Short_size must never exceed long_size at all times.
    5. Thoroughly analyze the provided technical indicators to identify trends.
    6. Do not hedge if free_balance is abundant. Focus on maximizing profit.
    7. Open LONG and SHORT positions to maximize profit.
    8. Exits should rely on take_profit_orders hitting their targets.
    9. Predict and place limit_order for entries.
    10. Always predict and place take_profit_order for open positions, even if your act is HOLD.
    
    Respond ONLY in this JSON:
    {"act": "L"|"S"|"H", "ep": <entry_price>, "tp": <take_profit_price>, "amt": <usdt>, "rsn": "<reasoning>"}
    (L:LONG, S:SHORT, H:HOLD. Use 'ep' for limit entry price, and 'tp' for take profit price.)
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
        # ⏰ 메인 사이클(루프) 시작 시간을 기록합니다.
        loop_start_time = time.time()
        
        try:
            logger.info("="*50)
            
            try:
                # 이전 루프의 미체결 지정가 및 익절(TP) 주문을 모두 취소하여 초기화합니다.
                exchange.cancel_all_orders(SYMBOL)
                logger.info(f"🧹 {SYMBOL} 미체결 주문 및 기존 TP 주문 모두 정리 완료")
            except Exception as e:
                logger.warning(f"⚠️ 미체결 주문 취소 중 오류 (무시 가능): {e}")

            free_usdt, long_size, long_price, long_pnl, short_size, short_price, short_pnl, long_contracts, short_contracts = get_account_state()
            current_price, recent_data, funding_rate = get_market_data()
            
            logger.info(f"💰 현재가: {current_price} | Funding: {funding_rate:.4f}%")
            logger.info(f"📊 LONG: {long_size} USDT (PnL: {long_pnl:.2f}) | SHORT: {short_size} USDT (PnL: {short_pnl:.2f}) | FREE: {free_usdt:.2f} USDT")
            
            logger.info("🧠 Gemini 3.1 Pro-preview 분석 중...")
            signal = get_gemini_signal(free_usdt, long_size, long_price, long_pnl, short_size, short_price, short_pnl, current_price, recent_data, funding_rate)
            
            # 사용하지 않는 CL(CLOSE_LONG), CS(CLOSE_SHORT) 삭제
            action_map = {"L": "LONG", "S": "SHORT", "H": "HOLD"}
            action = action_map.get(signal.get('act'), "HOLD")
            amount_usdt = float(signal.get('amt') or 0.0)
            
            # 진입가(ep) 및 익절가(tp) 받아오기
            order_price_raw = float(signal.get('ep') or current_price)
            if order_price_raw <= 0:
                order_price_raw = current_price
                
            tp_price_raw = float(signal.get('tp') or 0.0)
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

            # 🛡️ 3차 방어: 바이낸스 최소 주문 금액 (5 USDT) 1차 공통 확인 (CLOSE 액션 삭제됨)
            if action in ["LONG", "SHORT"] and amount_usdt > 0:
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
            # 💡 실제 주문 실행부 (TP 전용 전략으로 수정)
            # ==========================================
            if action == "LONG" and amount_usdt > 0:
                logger.info(f"🚀 롱 포지션 진입/추가 (수량: {order_qty} LTC | 지정가: {order_price} USDT)")
                
                # 1. 주문을 넣고, 해당 주문의 고유 ID를 저장합니다.
                order = exchange.create_order(SYMBOL, 'limit', 'buy', order_qty, order_price, params={'positionSide': 'LONG'})
                order_id = order['id']
                
                if tp_price > current_price:
                    logger.info("👀 롱 지정가 주문 체결 감시 시작... (5초 간격, 최대 10분 대기)")
                    is_filled = False
                    
                    # 2. 5초마다 거래소에 물어보는 미니 루프 (최대 120회 = 10분)
                    for _ in range(120):
                        time.sleep(5)
                        try:
                            check_order = exchange.fetch_order(order_id, SYMBOL)
                            if check_order['status'] == 'closed':
                                is_filled = True
                                break # 체결 확인! 즉시 루프 탈출
                            elif check_order['status'] in ['canceled', 'rejected', 'expired']:
                                logger.warning("⚠️ 주문이 취소되거나 거절되어 감시를 종료합니다.")
                                break
                        except Exception as e:
                            pass # 네트워크 지연 에러 등은 무시하고 5초 뒤 다시 확인
                            
                    # 3. 체결이 확인되었다면, 실시간 계좌 상태를 다시 읽어와서 완벽한 안전 수량으로 TP 설정
                    if is_filled:
                        logger.info("✅ 롱 주문 체결 확인! 즉시 계좌를 스캔하여 안전 방패(TP)를 세웁니다.")
                        _, _, _, _, _, _, _, new_long_contracts, new_short_contracts = get_account_state()
                        
                        safe_tp_qty_raw = new_long_contracts - new_short_contracts
                        
                        if safe_tp_qty_raw > 0:
                            safe_tp_qty = float(exchange.amount_to_precision(SYMBOL, safe_tp_qty_raw))
                            
                            if safe_tp_qty * tp_price >= 5.0:
                                logger.info(f"🎯 롱 포지션 부분 익절(TP) 실시간 설정 완료 (목표가: {tp_price} USDT | 수량: {safe_tp_qty} LTC, 숏 보호)")
                                exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell', safe_tp_qty, params={
                                    'positionSide': 'LONG', 
                                    'stopPrice': tp_price
                                })
                            else:
                                logger.warning("🛡️ 롱 TP 보류: 익절 가능 물량이 최소 주문 금액(5 USDT) 미만입니다.")
                        else:
                            logger.warning("🛡️ 롱 TP 보류: 숏 물량 보호를 위해 당장 롱 포지션을 익절할 수 없습니다.")
                    else:
                        logger.info("⏳ 10분 내에 체결되지 않았습니다. 남은 시간은 대기하며 미체결 상태는 다음 메인 루프에서 감시합니다.")
                    
            elif action == "SHORT" and amount_usdt > 0:
                logger.info(f"📉 숏 포지션 진입/추가 (수량: {order_qty} LTC | 지정가: {order_price} USDT)")
                
                # 1. 숏 주문 ID 저장
                order = exchange.create_order(SYMBOL, 'limit', 'sell', order_qty, order_price, params={'positionSide': 'SHORT'})
                order_id = order['id']
                
                if tp_price > 0 and tp_price < current_price:
                    logger.info("👀 숏 지정가 주문 체결 감시 시작... (5초 간격, 최대 10분 대기)")
                    is_filled = False
                    
                    # 2. 숏 미니 루프 감시
                    for _ in range(120):
                        time.sleep(5)
                        try:
                            check_order = exchange.fetch_order(order_id, SYMBOL)
                            if check_order['status'] == 'closed':
                                is_filled = True
                                break
                            elif check_order['status'] in ['canceled', 'rejected', 'expired']:
                                logger.warning("⚠️ 주문이 취소되거나 거절되어 감시를 종료합니다.")
                                break
                        except Exception as e:
                            pass
                            
                    # 3. 체결 확인 시 즉시 숏 전용 TP(closePosition) 전송
                    if is_filled:
                        logger.info("✅ 숏 주문 체결 확인! 즉시 안전 방패(TP)를 세웁니다.")
                        logger.info(f"🎯 숏 포지션 전체 익절(TP) 실시간 설정 완료 (목표가: {tp_price} USDT)")
                        exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'buy', None, params={
                            'positionSide': 'SHORT', 
                            'stopPrice': tp_price,
                            'closePosition': True
                        })
                    else:
                        logger.info("⏳ 10분 내에 체결되지 않았습니다. 남은 시간은 대기하며 미체결 상태는 다음 메인 루프에서 감시합니다.")
            
            else:
                logger.info("⏸️ 관망(HOLD) 또는 조건 불충족 상태 유지.")
                
                if tp_price > 0:
                    if long_contracts > 0 and tp_price > current_price:
                        # 기존 롱 물량에서도 숏 물량을 뺀 만큼만 갱신
                        safe_tp_qty_raw = long_contracts - short_contracts
                        
                        if safe_tp_qty_raw > 0:
                            safe_tp_qty = float(exchange.amount_to_precision(SYMBOL, safe_tp_qty_raw))
                            
                            if safe_tp_qty * tp_price >= 5.0:
                                logger.info(f"🎯 기존 롱 포지션 부분 익절(TP) 갱신 (목표가: {tp_price} USDT | 수량: {safe_tp_qty} LTC, 숏 보호)")
                                exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell', safe_tp_qty, params={
                                    'positionSide': 'LONG', 
                                    'stopPrice': tp_price
                                })
                            else:
                                logger.warning("🛡️ 기존 롱 TP 갱신 보류: 숏을 제외한 익절 가능 물량이 최소 주문 금액(5 USDT) 미만입니다.")
                        else:
                            logger.warning("🛡️ 기존 롱 TP 갱신 보류: 숏 물량 보호를 위해 현재 롱 포지션을 익절할 수 없습니다.")

                    if short_contracts > 0 and tp_price < current_price:
                        logger.info(f"🎯 기존 숏 포지션 익절(TP) 갱신 (목표가: {tp_price} USDT)")
                        exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'buy', None, params={
                            'positionSide': 'SHORT', 
                            'stopPrice': tp_price,
                            'closePosition': True
                        })

        except Exception as e:
            logger.error(f"🚨 시스템/네트워크 에러 발생: {e}")
            logger.info("🛡️ 방어 로직 작동: 10분 대기 후 재시도합니다...")
            time.sleep(600)
            continue 
            
        # ==========================================
        # ⏰ 남은 시간 계산 및 스마트 대기 로직
        # ==========================================
        elapsed_time = time.time() - loop_start_time 
        remaining_sleep_time = max(0, 1800 - elapsed_time) 
        
        logger.info(f"⏳ 현재 사이클 완료. 분석, 주문 및 감시에 {elapsed_time:.1f}초를 소모했습니다.")
        logger.info(f"⏳ 정확한 30분 주기 유지를 위해 남은 {remaining_sleep_time:.1f}초 동안 대기합니다...")
        
        time.sleep(remaining_sleep_time)

if __name__ == "__main__":
    try:
        logger.info("🚀 트레이딩 봇 가동을 시작합니다...")
        run_bot()
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 봇이 안전하게 종료되었습니다 (Ctrl+C).")
    except Exception as e:
        logger.critical(f"💥 치명적인 오류로 봇이 종료되었습니다: {e}")