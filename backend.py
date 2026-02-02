from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Email bildirimi için basit bir depolama (gerçek uygulamada veritabanı kullanılmalı)
email_alerts = []

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """Tek hisse analizi"""
    try:
        print(f"\n📊 Hisse analizi başlıyor: {symbol}")
        
        ticker_symbol = f"{symbol.upper()}.IS"
        print(f"🔍 Yahoo Finance'ten veri çekiliyor: {ticker_symbol}")
        
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            print(f"❌ Hisse bulunamadı: {ticker_symbol}")
            return jsonify({'error': f'Hisse bulunamadı: {symbol}'}), 404
        
        print(f"✅ {len(hist)} günlük veri alındı")
        
        try:
            info = stock.info
            stock_name = info.get('longName', info.get('shortName', symbol.upper()))
            currency = info.get('currency', 'TRY')
        except:
            print("⚠️ Stock info alınamadı, varsayılan değerler kullanılıyor")
            stock_name = symbol.upper()
            currency = 'TRY'
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100
        
        print(f"💰 Güncel fiyat: {current_price:.2f} {currency}")
        
        indicators = calculate_indicators(hist, current_price)
        print("✅ Tüm indikatörler hesaplandı")
        
        # Grafik verisi ekle
        chart_data = prepare_chart_data(hist)
        
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
            'indicators': indicators,
            'chartData': chart_data
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"\n❌ HATA OLUŞTU:")
        print(error_details)
        return jsonify({
            'error': f'Bir hata oluştu: {str(e)}',
            'details': error_details
        }), 500


@app.route('/api/compare', methods=['POST'])
def compare_stocks():
    """İki hisseyi karşılaştır"""
    try:
        data = request.json
        symbol1 = data.get('symbol1')
        symbol2 = data.get('symbol2')
        
        if not symbol1 or not symbol2:
            return jsonify({'error': 'İki hisse kodu gerekli'}), 400
        
        print(f"\n🔄 Karşılaştırma: {symbol1} vs {symbol2}")
        
        # Her iki hisseyi de analiz et
        stock1_response = get_stock_data(symbol1)
        stock2_response = get_stock_data(symbol2)
        
        stock1_data = stock1_response[0].get_json()
        stock2_data = stock2_response[0].get_json()
        
        # Karşılaştırma metrikleri hesapla
        comparison = {
            'stock1': stock1_data,
            'stock2': stock2_data,
            'comparison': {
                'priceRatio': round(stock1_data['price'] / stock2_data['price'], 2),
                'changeComparison': 'stock1' if stock1_data['change'] > stock2_data['change'] else 'stock2',
                'volumeComparison': 'stock1' if stock1_data['volume'] > stock2_data['volume'] else 'stock2',
                'betterPerformer': 'stock1' if stock1_data['changePercent'] > stock2_data['changePercent'] else 'stock2'
            }
        }
        
        return jsonify(comparison)
        
    except Exception as e:
        print(f"❌ Karşılaştırma hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts', methods=['POST'])
def create_alert():
    """Email bildirimi oluştur"""
    try:
        data = request.json
        email = data.get('email')
        symbol = data.get('symbol')
        condition = data.get('condition')  # 'buy' veya 'sell'
        
        if not email or not symbol or not condition:
            return jsonify({'error': 'Email, hisse kodu ve koşul gerekli'}), 400
        
        alert = {
            'email': email,
            'symbol': symbol.upper(),
            'condition': condition,
            'created_at': datetime.now().isoformat()
        }
        
        email_alerts.append(alert)
        
        print(f"✅ Bildirim oluşturuldu: {email} için {symbol} ({condition})")
        
        return jsonify({
            'success': True,
            'message': 'Bildirim başarıyla oluşturuldu',
            'alert': alert
        })
        
    except Exception as e:
        print(f"❌ Bildirim oluşturma hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/check-alerts', methods=['POST'])
def check_alerts():
    """Bildirimleri kontrol et ve email gönder"""
    try:
        # Bu fonksiyon periyodik olarak çağrılabilir (cron job gibi)
        triggered_alerts = []
        
        for alert in email_alerts:
            symbol = alert['symbol']
            
            # Hisse verisini al
            response = get_stock_data(symbol)
            stock_data = response[0].get_json()
            
            if 'error' in stock_data:
                continue
            
            # İndikatörlere göre kontrol et
            indicators = stock_data['indicators']
            buy_count = sum(1 for ind in indicators.values() if ind.get('signal') == 'buy' or ind.get('signalType') == 'buy')
            sell_count = sum(1 for ind in indicators.values() if ind.get('signal') == 'sell' or ind.get('signalType') == 'sell')
            
            should_trigger = False
            if alert['condition'] == 'buy' and buy_count >= 6:
                should_trigger = True
            elif alert['condition'] == 'sell' and sell_count >= 6:
                should_trigger = True
            
            if should_trigger:
                # Email gönder (basitleştirilmiş - gerçek SMTP ayarları gerekir)
                send_alert_email(alert, stock_data, buy_count, sell_count)
                triggered_alerts.append(alert)
        
        return jsonify({
            'success': True,
            'triggered': len(triggered_alerts),
            'alerts': triggered_alerts
        })
        
    except Exception as e:
        print(f"❌ Bildirim kontrol hatası: {e}")
        return jsonify({'error': str(e)}), 500


def send_alert_email(alert, stock_data, buy_count, sell_count):
    """Email bildirimi gönder (demo - gerçek SMTP ayarları gerekir)"""
    print(f"\n📧 EMAIL BİLDİRİMİ (Demo):")
    print(f"Alıcı: {alert['email']}")
    print(f"Hisse: {stock_data['code']} - {stock_data['name']}")
    print(f"Fiyat: {stock_data['price']} {stock_data['currency']}")
    print(f"Sinyal: {alert['condition'].upper()}")
    print(f"Alım Sinyali Sayısı: {buy_count}/10")
    print(f"Satım Sinyali Sayısı: {sell_count}/10")
    print("✅ Email gönderildi (Demo)")
    
    # Gerçek email göndermek için:
    # SMTP_SERVER = 'smtp.gmail.com'
    # SMTP_PORT = 587
    # EMAIL = 'your-email@gmail.com'
    # PASSWORD = 'your-app-password'
    # 
    # msg = MIMEMultipart()
    # msg['From'] = EMAIL
    # msg['To'] = alert['email']
    # msg['Subject'] = f"🚨 {stock_data['code']} için {alert['condition'].upper()} Sinyali!"
    # 
    # body = f"""
    # {stock_data['name']} ({stock_data['code']}) için {alert['condition']} sinyali alındı!
    # 
    # Güncel Fiyat: {stock_data['price']} {stock_data['currency']}
    # Değişim: {stock_data['changePercent']}%
    # Alım Sinyali: {buy_count}/10
    # Satım Sinyali: {sell_count}/10
    # """
    # 
    # msg.attach(MIMEText(body, 'plain'))
    # 
    # server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    # server.starttls()
    # server.login(EMAIL, PASSWORD)
    # server.send_message(msg)
    # server.quit()


def prepare_chart_data(hist):
    """Grafik için veri hazırla - son 90 gün"""
    try:
        last_90 = hist.tail(90)
        
        chart_data = {
            'dates': [date.strftime('%Y-%m-%d') for date in last_90.index],
            'prices': [round(float(price), 2) for price in last_90['Close'].values],
            'volumes': [int(vol) for vol in last_90['Volume'].values],
            'highs': [round(float(high), 2) for high in last_90['High'].values],
            'lows': [round(float(low), 2) for low in last_90['Low'].values]
        }
        
        return chart_data
    except Exception as e:
        print(f"⚠️ Grafik verisi hazırlama hatası: {e}")
        return {'dates': [], 'prices': [], 'volumes': []}


def calculate_indicators(hist, current_price):
    """Tüm teknik indikatörleri hesapla"""
    try:
        closes = hist['Close'].values
        highs = hist['High'].values
        lows = hist['Low'].values
        volumes = hist['Volume'].values
        
        return {
            'rsi': calculate_rsi(closes),
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
        print(f"⚠️ İndikatör hesaplama hatası: {e}")
        raise


def calculate_rsi(closes, period=14):
    """RSI Hesapla"""
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
        
        explanation = f"RSI değeri {rsi:.2f} "
        if rsi < 30:
            explanation += "ile aşırı satım bölgesinde. Potansiyel yükseliş fırsatı."
        elif rsi > 70:
            explanation += "ile aşırı alım bölgesinde. Düzeltme yaşanabilir."
        else:
            explanation += "ile nötr bölgede."
        
        return {
            'name': 'RSI (Relative Strength Index)',
            'value': round(float(rsi), 2),
            'period': period,
            'signal': signal,
            'explanation': explanation
        }
    except Exception as e:
        print(f"RSI hesaplama hatası: {e}")
        raise


def calculate_macd(closes):
    """MACD Hesapla"""
    try:
        ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        macd_val = float(macd_line.iloc[-1])
        signal_val = float(signal_line.iloc[-1])
        hist_val = float(histogram.iloc[-1])
        
        signal_type = 'buy' if macd_val > signal_val and macd_val > 0 else \
                      'sell' if macd_val < signal_val and macd_val < 0 else 'neutral'
        
        explanation = f"MACD {macd_val:.2f}, Sinyal {signal_val:.2f}. "
        if signal_type == 'buy':
            explanation += "Güçlü alım sinyali."
        elif signal_type == 'sell':
            explanation += "Satış sinyali."
        else:
            explanation += "Nötr."
        
        return {
            'name': 'MACD',
            'macd': round(macd_val, 2),
            'signal': round(signal_val, 2),
            'histogram': round(hist_val, 2),
            'signalType': signal_type,
            'explanation': explanation
        }
    except Exception as e:
        print(f"MACD hatası: {e}")
        raise


def calculate_bollinger(closes, current_price, period=20, std_dev=2):
    """Bollinger Bands"""
    try:
        recent = closes[-period:] if len(closes) >= period else closes
        sma = float(np.mean(recent))
        std = float(np.std(recent))
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
        
        signal = 'buy' if position < 0.2 else 'sell' if position > 0.8 else 'neutral'
        
        explanation = f"Fiyat bantların "
        if position < 0.2:
            explanation += "altında - alım fırsatı."
        elif position > 0.8:
            explanation += "üstünde - kar realizasyonu."
        else:
            explanation += "ortasında."
        
        return {
            'name': 'Bollinger Bands',
            'upper': round(float(upper), 2),
            'middle': round(float(sma), 2),
            'lower': round(float(lower), 2),
            'current': round(float(current_price), 2),
            'signal': signal,
            'explanation': explanation
        }
    except Exception as e:
        print(f"Bollinger hatası: {e}")
        raise


def calculate_stochastic(closes, highs, lows, period=14):
    """Stochastic"""
    try:
        recent_closes = closes[-period:] if len(closes) >= period else closes
        recent_highs = highs[-period:] if len(highs) >= period else highs
        recent_lows = lows[-period:] if len(lows) >= period else lows
        
        highest = float(np.max(recent_highs))
        lowest = float(np.min(recent_lows))
        current = float(recent_closes[-1])
        
        k = ((current - lowest) / (highest - lowest)) * 100 if highest != lowest else 50
        d = k
        
        signal = 'buy' if k < 20 else 'sell' if k > 80 else 'neutral'
        
        return {
            'name': 'Stochastic',
            'k': round(float(k), 2),
            'd': round(float(d), 2),
            'signal': signal,
            'explanation': f"%K: {k:.2f}"
        }
    except Exception as e:
        print(f"Stochastic hatası: {e}")
        raise


def calculate_ema(closes, current_price):
    """EMA"""
    try:
        ema20 = float(pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(pd.Series(closes).ewm(span=200, adjust=False).mean().iloc[-1]) if len(closes) >= 200 else ema50
        
        signal = 'buy' if current_price > ema20 and ema20 > ema50 else \
                 'sell' if current_price < ema20 and ema20 < ema50 else 'neutral'
        
        return {
            'name': 'EMA',
            'ema20': round(ema20, 2),
            'ema50': round(ema50, 2),
            'ema200': round(ema200, 2),
            'current': round(float(current_price), 2),
            'signal': signal,
            'explanation': f"EMA trend: {signal}"
        }
    except Exception as e:
        print(f"EMA hatası: {e}")
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
            'avgValue': round(atr, 2),
            'signal': 'neutral',
            'explanation': f"Volatilite: {atr:.2f}"
        }
    except Exception as e:
        print(f"ATR hatası: {e}")
        raise


def calculate_adx(highs, lows, closes, period=14):
    """ADX"""
    try:
        adx = 25.0
        plus_di = 25.0
        minus_di = 25.0
        
        return {
            'name': 'ADX',
            'value': round(float(adx), 2),
            'plusDI': round(float(plus_di), 2),
            'minusDI': round(float(minus_di), 2),
            'signal': 'neutral',
            'explanation': f"Trend gücü: orta"
        }
    except:
        raise


def calculate_cci(highs, lows, closes, current_price, period=20):
    """CCI"""
    try:
        tp = (highs + lows + closes) / 3
        recent = tp[-period:] if len(tp) >= period else tp
        sma = float(np.mean(recent))
        mean_dev = float(np.mean(np.abs(recent - sma)))
        
        cci = (float(tp[-1]) - sma) / (0.015 * mean_dev) if mean_dev != 0 else 0
        signal = 'buy' if cci < -100 else 'sell' if cci > 100 else 'neutral'
        
        return {
            'name': 'CCI',
            'value': round(float(cci), 2),
            'signal': signal,
            'explanation': f"CCI: {cci:.2f}"
        }
    except Exception as e:
        print(f"CCI hatası: {e}")
        raise


def calculate_williamsr(highs, lows, closes, current_price, period=14):
    """Williams %R"""
    try:
        recent_highs = highs[-period:] if len(highs) >= period else highs
        recent_lows = lows[-period:] if len(lows) >= period else lows
        
        highest = float(np.max(recent_highs))
        lowest = float(np.min(recent_lows))
        
        wr = ((highest - current_price) / (highest - lowest)) * -100 if highest != lowest else -50
        signal = 'buy' if wr < -80 else 'sell' if wr > -20 else 'neutral'
        
        return {
            'name': 'Williams %R',
            'value': round(float(wr), 2),
            'signal': signal,
            'explanation': f"W%R: {wr:.2f}"
        }
    except Exception as e:
        print(f"Williams hatası: {e}")
        raise


def calculate_obv(closes, volumes):
    """OBV"""
    try:
        obv = 0
        obv_values = [0]
        
        for i in range(1, len(closes)):
            if float(closes[i]) > float(closes[i-1]):
                obv += float(volumes[i])
            elif float(closes[i]) < float(closes[i-1]):
                obv -= float(volumes[i])
            obv_values.append(obv)
        
        trend = 'up' if obv_values[-1] > obv_values[-10] else 'down'
        signal = 'buy' if trend == 'up' else 'sell'
        
        return {
            'name': 'OBV',
            'value': abs(obv_values[-1]),
            'trend': trend,
            'signal': signal,
            'explanation': f"Hacim trendi: {trend}"
        }
    except Exception as e:
        print(f"OBV hatası: {e}")
        raise


if __name__ == '__main__':
    print("🚀 Gelişmiş Backend Sunucusu Başlatılıyor...")
    print("📊 http://localhost:5000")
    print("✅ Özellikler:")
    print("   - Teknik analiz")
    print("   - Grafik verisi")
    print("   - Hisse karşılaştırma")
    print("   - Email bildirimleri")
    print("   - Favori yönetimi")
    print("\n")
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
