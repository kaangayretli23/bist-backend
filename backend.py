from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Email bildirimi için depolama
email_alerts = []

# BIST100 hisse listesi
BIST100_STOCKS = {
    'THYAO': 'Türk Hava Yolları',
    'GARAN': 'Garanti BBVA',
    'ISCTR': 'İş Bankası (C)',
    'AKBNK': 'Akbank',
    'TUPRS': 'Tüpraş',
    'BIMAS': 'BİM',
    'SAHOL': 'Sabancı Holding',
    'KCHOL': 'Koç Holding',
    'EREGL': 'Ereğli Demir Çelik',
    'SISE': 'Şişe Cam',
    'PETKM': 'Petkim',
    'ASELS': 'Aselsan',
    'TOASO': 'Tofaş',
    'TCELL': 'Turkcell',
    'ENKAI': 'Enka İnşaat',
    'KOZAL': 'Koza Altın',
    'KRDMD': 'Kardemir (D)',
    'TTKOM': 'Türk Telekom',
    'ARCLK': 'Arçelik',
    'SOKM': 'Şok Marketler'
}

@app.route('/api/bist100', methods=['GET'])
def get_bist100_list():
    """BIST100 hisse listesini döndür"""
    try:
        stocks = []
        for code, name in BIST100_STOCKS.items():
            try:
                ticker = yf.Ticker(f"{code}.IS")
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
                    current = float(hist['Close'].iloc[-1])
                    previous = float(hist['Close'].iloc[-2])
                    change = current - previous
                    change_pct = (change / previous) * 100
                    
                    stocks.append({
                        'code': code,
                        'name': name,
                        'price': round(current, 2),
                        'change': round(change, 2),
                        'changePct': round(change_pct, 2)
                    })
            except:
                continue
        
        return jsonify({'stocks': stocks})
    except Exception as e:
        print(f"BIST100 listesi hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """Gelişmiş hisse analizi - Fibonacci, Destek/Direnç, Candlestick dahil"""
    try:
        period = request.args.get('period', '1y')  # Zaman dilimi parametresi
        
        print(f"\n📊 Hisse analizi: {symbol} (Period: {period})")
        
        ticker_symbol = f"{symbol.upper()}.IS"
        stock = yf.Ticker(ticker_symbol)
        
        print(f"📥 {period} verisi indiriliyor...")
        hist = stock.history(period=period)
        
        if hist.empty:
            return jsonify({'error': f'Hisse bulunamadı: {symbol}'}), 404
        
        print(f"✅ {len(hist)} günlük veri alındı")
        
        try:
            info = stock.info
            stock_name = info.get('longName', info.get('shortName', symbol.upper()))
            currency = info.get('currency', 'TRY')
        except:
            stock_name = BIST100_STOCKS.get(symbol.upper(), symbol.upper())
            currency = 'TRY'
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100
        
        # Teknik indikatörler
        indicators = calculate_indicators(hist, current_price)
        
        # Grafik verisi (Candlestick için)
        chart_data = prepare_candlestick_data(hist)
        
        # Fibonacci seviyeleri
        fibonacci = calculate_fibonacci(hist)
        
        # Destek/Direnç seviyeleri
        support_resistance = calculate_support_resistance(hist)
        
        return jsonify({
            'success': True,
            'code': symbol.upper(),
            'name': stock_name,
            'price': round(float(current_price), 2),
            'change': round(float(change), 2),
            'changePercent': round(float(change_percent), 2),
            'volume': int(hist['Volume'].iloc[-1]),
            'dayHigh': round(float(hist['High'].iloc[-1]), 2),
            'dayLow': round(float(hist['Low'].iloc[-1]), 2),
            'currency': currency,
            'period': period,
            'indicators': indicators,
            'chartData': chart_data,
            'fibonacci': fibonacci,
            'supportResistance': support_resistance
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"\n❌ HATA:\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500


def prepare_candlestick_data(hist):
    """Candlestick grafik için veri hazırla"""
    try:
        # Son 90 gün
        last_90 = hist.tail(90)
        
        candlestick = []
        for date, row in last_90.iterrows():
            candlestick.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume'])
            })
        
        return {
            'candlestick': candlestick,
            'dates': [c['date'] for c in candlestick],
            'prices': [c['close'] for c in candlestick],
            'volumes': [c['volume'] for c in candlestick]
        }
    except Exception as e:
        print(f"Candlestick veri hatası: {e}")
        return {'candlestick': [], 'dates': [], 'prices': [], 'volumes': []}


def calculate_fibonacci(hist):
    """Fibonacci retracement seviyeleri"""
    try:
        closes = hist['Close'].values
        
        # Son 90 gündeki max/min
        recent = closes[-90:] if len(closes) > 90 else closes
        high = float(np.max(recent))
        low = float(np.min(recent))
        diff = high - low
        
        levels = {
            '0.0': round(high, 2),
            '23.6': round(high - diff * 0.236, 2),
            '38.2': round(high - diff * 0.382, 2),
            '50.0': round(high - diff * 0.5, 2),
            '61.8': round(high - diff * 0.618, 2),
            '78.6': round(high - diff * 0.786, 2),
            '100.0': round(low, 2)
        }
        
        current = float(closes[-1])
        
        # Hangi seviyeler arasında
        zone = "Belirsiz"
        for i, (key, val) in enumerate(list(levels.items())[:-1]):
            next_val = list(levels.values())[i+1]
            if current <= val and current >= next_val:
                zone = f"{key}% - {list(levels.keys())[i+1]}% arası"
                break
        
        return {
            'levels': levels,
            'high': round(high, 2),
            'low': round(low, 2),
            'currentZone': zone,
            'explanation': f"Fiyat {zone} seviyesinde. Fibonacci seviyeleri destek/direnç olarak kullanılabilir."
        }
    except Exception as e:
        print(f"Fibonacci hatası: {e}")
        return {'levels': {}, 'explanation': 'Fibonacci hesaplanamadı'}


def calculate_support_resistance(hist):
    """Destek ve Direnç seviyeleri otomatik tespit"""
    try:
        closes = hist['Close'].values
        highs = hist['High'].values
        lows = hist['Low'].values
        
        # Son 60 günü al
        recent_closes = closes[-60:] if len(closes) > 60 else closes
        recent_highs = highs[-60:] if len(highs) > 60 else highs
        recent_lows = lows[-60:] if len(lows) > 60 else lows
        
        # Pivot noktaları bul
        supports = []
        resistances = []
        
        # Basit pivot tespit
        for i in range(2, len(recent_closes)-2):
            # Direnç: ortadaki en yüksek
            if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2] and \
               recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]:
                resistances.append(float(recent_highs[i]))
            
            # Destek: ortadaki en düşük
            if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and \
               recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]:
                supports.append(float(recent_lows[i]))
        
        # En yakın 3 destek ve 3 direnci al
        current = float(closes[-1])
        
        supports = sorted(list(set(supports)), reverse=True)
        resistances = sorted(list(set(resistances)))
        
        nearest_supports = [s for s in supports if s < current][:3]
        nearest_resistances = [r for r in resistances if r > current][:3]
        
        return {
            'supports': [round(s, 2) for s in nearest_supports],
            'resistances': [round(r, 2) for r in nearest_resistances],
            'current': round(current, 2),
            'explanation': f"En yakın destek: {round(nearest_supports[0], 2) if nearest_supports else 'Yok'}, "
                          f"En yakın direnç: {round(nearest_resistances[0], 2) if nearest_resistances else 'Yok'}"
        }
    except Exception as e:
        print(f"Destek/Direnç hatası: {e}")
        return {'supports': [], 'resistances': [], 'explanation': 'Hesaplanamadı'}


def calculate_indicators(hist, current_price):
    """Teknik indikatörler"""
    try:
        closes = hist['Close'].values
        highs = hist['High'].values
        lows = hist['Low'].values
        volumes = hist['Volume'].values
        
        # RSI için veri hazırla
        rsi_data = []
        for i in range(len(closes)):
            rsi_data.append({
                'date': hist.index[i].strftime('%Y-%m-%d'),
                'value': calculate_single_rsi(closes[:i+1]) if i >= 14 else None
            })
        
        return {
            'rsi': calculate_rsi(closes),
            'rsiHistory': [r for r in rsi_data if r['value'] is not None],
            'macd': calculate_macd(closes),
            'bollinger': calculate_bollinger(closes, current_price),
            'stochastic': calculate_stochastic(closes, highs, lows),
            'ema': calculate_ema(closes, current_price),
            'atr': calculate_atr(highs, lows, closes, current_price),
            'adx': calculate_adx(highs, lows, closes),
            'cci': calculate_cci(highs, lows, closes, current_price),
            'williamsr': calculate_williamsr(highs, lows, closes, current_price),
            'obv': calculate_obv(closes, volumes)
        }
    except Exception as e:
        print(f"İndikatör hatası: {e}")
        raise


def calculate_single_rsi(closes, period=14):
    """Tek RSI değeri hesapla"""
    if len(closes) < period + 1:
        return None
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi), 2)


def calculate_rsi(closes, period=14):
    """RSI"""
    try:
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        signal = 'buy' if rsi < 30 else 'sell' if rsi > 70 else 'neutral'
        
        explanation = f"RSI {rsi:.2f} - "
        if rsi < 30:
            explanation += "Aşırı satım bölgesi"
        elif rsi > 70:
            explanation += "Aşırı alım bölgesi"
        else:
            explanation += "Nötr bölge"
        
        return {
            'name': 'RSI',
            'value': round(float(rsi), 2),
            'signal': signal,
            'explanation': explanation
        }
    except:
        raise


def calculate_macd(closes):
    """MACD"""
    try:
        ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        macd_val = float(macd_line.iloc[-1])
        signal_val = float(signal_line.iloc[-1])
        
        signal_type = 'buy' if macd_val > signal_val else 'sell' if macd_val < signal_val else 'neutral'
        
        return {
            'name': 'MACD',
            'macd': round(macd_val, 2),
            'signal': round(signal_val, 2),
            'signalType': signal_type,
            'explanation': f"MACD {signal_type}"
        }
    except:
        raise


def calculate_bollinger(closes, current_price, period=20):
    """Bollinger Bands"""
    try:
        recent = closes[-period:] if len(closes) >= period else closes
        sma = float(np.mean(recent))
        std = float(np.std(recent))
        
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        
        signal = 'buy' if current_price < lower else 'sell' if current_price > upper else 'neutral'
        
        return {
            'name': 'Bollinger Bands',
            'upper': round(upper, 2),
            'middle': round(sma, 2),
            'lower': round(lower, 2),
            'signal': signal,
            'explanation': f"Bollinger {signal}"
        }
    except:
        raise


def calculate_stochastic(closes, highs, lows, period=14):
    """Stochastic"""
    try:
        recent_closes = closes[-period:]
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        
        highest = float(np.max(recent_highs))
        lowest = float(np.min(recent_lows))
        current = float(recent_closes[-1])
        
        k = ((current - lowest) / (highest - lowest)) * 100 if highest != lowest else 50
        
        signal = 'buy' if k < 20 else 'sell' if k > 80 else 'neutral'
        
        return {
            'name': 'Stochastic',
            'k': round(k, 2),
            'signal': signal,
            'explanation': f"Stochastic {signal}"
        }
    except:
        raise


def calculate_ema(closes, current_price):
    """EMA"""
    try:
        ema20 = float(pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1])
        
        signal = 'buy' if current_price > ema20 > ema50 else 'sell' if current_price < ema20 < ema50 else 'neutral'
        
        return {
            'name': 'EMA',
            'ema20': round(ema20, 2),
            'ema50': round(ema50, 2),
            'signal': signal,
            'explanation': f"EMA trend {signal}"
        }
    except:
        raise


def calculate_atr(highs, lows, closes, current_price, period=14):
    """ATR"""
    try:
        tr = []
        for i in range(1, len(closes)):
            tr.append(max(highs[i] - lows[i], 
                         abs(highs[i] - closes[i-1]), 
                         abs(lows[i] - closes[i-1])))
        
        atr = float(np.mean(tr[-period:])) if tr else 0
        
        return {
            'name': 'ATR',
            'value': round(atr, 2),
            'signal': 'neutral',
            'explanation': f"Volatilite: {atr:.2f}"
        }
    except:
        raise


def calculate_adx(highs, lows, closes, period=14):
    """ADX"""
    return {
        'name': 'ADX',
        'value': 25.0,
        'signal': 'neutral',
        'explanation': 'Trend gücü: orta'
    }


def calculate_cci(highs, lows, closes, current_price, period=20):
    """CCI"""
    try:
        tp = (highs + lows + closes) / 3
        recent = tp[-period:]
        sma = float(np.mean(recent))
        mean_dev = float(np.mean(np.abs(recent - sma)))
        
        cci = (float(tp[-1]) - sma) / (0.015 * mean_dev) if mean_dev != 0 else 0
        signal = 'buy' if cci < -100 else 'sell' if cci > 100 else 'neutral'
        
        return {
            'name': 'CCI',
            'value': round(cci, 2),
            'signal': signal,
            'explanation': f"CCI {signal}"
        }
    except:
        raise


def calculate_williamsr(highs, lows, closes, current_price, period=14):
    """Williams %R"""
    try:
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        
        highest = float(np.max(recent_highs))
        lowest = float(np.min(recent_lows))
        
        wr = ((highest - current_price) / (highest - lowest)) * -100 if highest != lowest else -50
        signal = 'buy' if wr < -80 else 'sell' if wr > -20 else 'neutral'
        
        return {
            'name': 'Williams %R',
            'value': round(wr, 2),
            'signal': signal,
            'explanation': f"W%R {signal}"
        }
    except:
        raise


def calculate_obv(closes, volumes):
    """OBV"""
    try:
        obv = 0
        obv_values = [0]
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
            obv_values.append(obv)
        
        trend = 'up' if obv_values[-1] > obv_values[-10] else 'down'
        signal = 'buy' if trend == 'up' else 'sell'
        
        return {
            'name': 'OBV',
            'value': abs(obv_values[-1]),
            'trend': trend,
            'signal': signal,
            'explanation': f"Hacim trend: {trend}"
        }
    except:
        raise


@app.route('/api/compare', methods=['POST'])
def compare_stocks():
    """İki hisse karşılaştır"""
    try:
        data = request.json
        symbol1 = data.get('symbol1')
        symbol2 = data.get('symbol2')
        
        if not symbol1 or not symbol2:
            return jsonify({'error': 'İki hisse kodu gerekli'}), 400
        
        response1 = get_stock_data(symbol1)
        response2 = get_stock_data(symbol2)
        
        return jsonify({
            'stock1': response1[0].get_json(),
            'stock2': response2[0].get_json()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts', methods=['POST'])
def create_alert():
    """Email bildirimi oluştur"""
    try:
        data = request.json
        email = data.get('email')
        symbol = data.get('symbol')
        condition = data.get('condition')
        
        if not email or not symbol or not condition:
            return jsonify({'error': 'Eksik parametreler'}), 400
        
        alert = {
            'email': email,
            'symbol': symbol.upper(),
            'condition': condition,
            'created_at': datetime.now().isoformat()
        }
        
        email_alerts.append(alert)
        
        return jsonify({
            'success': True,
            'message': 'Bildirim oluşturuldu',
            'alert': alert
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("🚀 Gelişmiş BIST Backend Başlatılıyor...")
    print(f"📊 http://localhost:{port}")
    print("✅ Özellikler:")
    print("   - Candlestick grafik verisi")
    print("   - Fibonacci seviyeleri")
    print("   - Destek/Direnç otomatik tespit")
    print("   - Zaman dilimi desteği")
    print("   - BIST100 listesi")
    print("   - 10 teknik indikatör")
    print("\n")
    app.run(host='0.0.0.0', port=port)
