"""
BIST Pro - Basic Technical Indicators
Temel teknik analiz hesaplama fonksiyonlari:
RSI, MACD, Bollinger, EMA, Stochastic, ATR, ADX, OBV,
Williams %R, CCI, MFI, VWAP, Ichimoku, PSAR, Pivot, ROC, Aroon, TRIX, DMI
+ Support/Resistance, Fibonacci, History fonksiyonlari
"""
import numpy as np
import pandas as pd
from config import sf, si

# =====================================================================
# UTILITY
# =====================================================================
def _resample_to_tf(hist_daily, tf):
    """Gunluk OHLCV verisini haftalik/aylik bara donustur."""
    try:
        df = hist_daily.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        period_code = 'W' if tf == 'weekly' else 'M'
        df['_p'] = df.index.to_period(period_code)
        agg = {'Close': 'last', 'High': 'max', 'Low': 'min', 'Volume': 'sum'}
        if 'Open' in df.columns:
            agg['Open'] = 'first'
        resampled = df.groupby('_p').agg(agg)
        resampled.index = resampled.index.to_timestamp()
        resampled = resampled.dropna(subset=['Close'])
        if len(resampled) > 2:
            resampled = resampled.iloc[:-1]
        if 'High' not in resampled.columns:
            resampled['High'] = resampled['Close']
        if 'Low' not in resampled.columns:
            resampled['Low'] = resampled['Close']
        return resampled if len(resampled) >= 5 else None
    except Exception as e:
        print(f"  [MTF-RESAMPLE] {tf}: {e}")
        return None


# =====================================================================
# TEMEL MOMENTUM / TREND / VOLATILITE
# =====================================================================
EMA_PERIODS = [5, 10, 20, 50, 100, 200]

def calc_rsi(closes, period=14):
    if len(closes)<period+1: return {'name':'RSI','value':50.0,'signal':'neutral'}
    d=np.diff(closes)
    ag=float(np.mean(np.where(d>0,d,0)[-period:]))
    al=float(np.mean(np.where(d<0,-d,0)[-period:]))
    rsi=100.0 if al==0 else sf(100-100/(1+ag/al))
    return {'name':'RSI','value':rsi,'signal':'buy' if rsi<30 else ('sell' if rsi>70 else 'neutral')}

def calc_rsi_single(closes, period=14):
    if len(closes)<period+1: return None
    d=np.diff(closes)
    ag=float(np.mean(np.where(d>0,d,0)[-period:]))
    al=float(np.mean(np.where(d<0,-d,0)[-period:]))
    return 100.0 if al==0 else sf(100-100/(1+ag/al))

def calc_macd(closes):
    if len(closes)<26: return {'name':'MACD','macd':0,'signal':0,'histogram':0,'signalType':'neutral'}
    s=pd.Series(list(closes),dtype=float); ml=s.ewm(span=12).mean()-s.ewm(span=26).mean(); sl=ml.ewm(span=9).mean()
    mv,sv=sf(ml.iloc[-1]),sf(sl.iloc[-1])
    return {'name':'MACD','macd':mv,'signal':sv,'histogram':sf(mv-sv),'signalType':'buy' if mv>sv else ('sell' if mv<sv else 'neutral')}

def calc_macd_history(closes):
    if len(closes)<26: return []
    s=pd.Series(list(closes),dtype=float); ml=s.ewm(span=12).mean()-s.ewm(span=26).mean(); sl=ml.ewm(span=9).mean()
    return [{'macd':sf(ml.iloc[i]),'signal':sf(sl.iloc[i]),'histogram':sf((ml-sl).iloc[i])} for i in range(26,len(closes))]

def calc_bollinger(closes, cp, period=20):
    if len(closes)<period: return {'name':'Bollinger','upper':0,'middle':0,'lower':0,'signal':'neutral','bandwidth':0}
    r=closes[-period:]; sma,std=float(np.mean(r)),float(np.std(r))
    u,m,lo=sf(sma+2*std),sf(sma),sf(sma-2*std)
    return {'name':'Bollinger','upper':u,'middle':m,'lower':lo,'bandwidth':sf((u-lo)/m*100 if m else 0),'signal':'buy' if float(cp)<lo else ('sell' if float(cp)>u else 'neutral')}

def calc_bollinger_history(closes, period=20):
    r=[]
    for i in range(period,len(closes)):
        w=closes[i-period:i]; sma,std=float(np.mean(w)),float(np.std(w))
        r.append({'upper':sf(sma+2*std),'middle':sf(sma),'lower':sf(sma-2*std)})
    return r

def calc_ema(closes, cp):
    result={'name':'EMA','signal':'neutral'}
    s=pd.Series(list(closes),dtype=float)
    for p in EMA_PERIODS:
        if len(closes)>=p:
            result[f'ema{p}']=sf(s.ewm(span=p).mean().iloc[-1])
    e20,e50=result.get('ema20',cp),result.get('ema50',cp)
    if float(cp)>e20>e50: result['signal']='buy'
    elif float(cp)<e20<e50: result['signal']='sell'
    return result

def calc_ema_history(closes):
    s=pd.Series(list(closes),dtype=float)
    emas={}
    for p in EMA_PERIODS:
        if len(closes)>=p:
            emas[f'ema{p}']=s.ewm(span=p).mean()
    r=[]
    for i in range(len(closes)):
        pt={}
        for k,v in emas.items():
            if i<len(v): pt[k]=sf(v.iloc[i])
        r.append(pt)
    return r

def calc_stochastic(closes, highs, lows, period=14):
    if len(closes)<period: return {'name':'Stochastic','k':50,'d':50,'signal':'neutral'}
    hi,lo,cur=float(np.max(highs[-period:])),float(np.min(lows[-period:])),float(closes[-1])
    k=sf(((cur-lo)/(hi-lo))*100 if hi!=lo else 50)
    return {'name':'Stochastic','k':k,'d':k,'signal':'buy' if k<20 else ('sell' if k>80 else 'neutral')}

def calc_stochastic_history(closes, highs, lows, period=14):
    r=[]
    for i in range(period, len(closes)):
        hi=float(np.max(highs[i-period:i+1]))
        lo=float(np.min(lows[i-period:i+1]))
        cur=float(closes[i])
        k=sf(((cur-lo)/(hi-lo))*100 if hi!=lo else 50)
        r.append({'k':k})
    for i in range(len(r)):
        if i >= 2:
            r[i]['d'] = sf((r[i]['k'] + r[i-1]['k'] + r[i-2]['k']) / 3)
        else:
            r[i]['d'] = r[i]['k']
    return r

def calc_atr(highs, lows, closes, period=14):
    if len(closes)<period+1: return {'name':'ATR','value':0,'pct':0,'signal':'neutral'}
    tr=[max(float(highs[i])-float(lows[i]),abs(float(highs[i])-float(closes[i-1])),abs(float(lows[i])-float(closes[i-1]))) for i in range(1,len(closes))]
    atr=sf(np.mean(tr[-period:]))
    return {'name':'ATR','value':atr,'pct':sf(atr/float(closes[-1])*100 if closes[-1] else 0),'signal':'neutral'}

def calc_adx(highs, lows, closes, period=14):
    n=len(closes)
    if n<period+1: return {'name':'ADX','value':25,'plusDI':0,'minusDI':0,'signal':'neutral'}
    tr,pdm,mdm=[],[],[]
    for i in range(1,n):
        hv,lv,phv,plv,pcv=float(highs[i]),float(lows[i]),float(highs[i-1]),float(lows[i-1]),float(closes[i-1])
        tr.append(max(hv-lv,abs(hv-pcv),abs(lv-pcv)))
        um,dm=hv-phv,plv-lv
        pdm.append(um if um>dm and um>0 else 0); mdm.append(dm if dm>um and dm>0 else 0)
    if len(tr)<period: return {'name':'ADX','value':25,'plusDI':0,'minusDI':0,'signal':'neutral'}
    atr_s,pdm_s,mdm_s=float(np.mean(tr[:period])),float(np.mean(pdm[:period])),float(np.mean(mdm[:period]))
    for i in range(period,len(tr)):
        atr_s=(atr_s*(period-1)+tr[i])/period; pdm_s=(pdm_s*(period-1)+pdm[i])/period; mdm_s=(mdm_s*(period-1)+mdm[i])/period
    pdi=sf((pdm_s/atr_s)*100 if atr_s else 0); mdi=sf((mdm_s/atr_s)*100 if atr_s else 0)
    ds=pdi+mdi; adx=sf(abs(pdi-mdi)/ds*100 if ds else 0)
    return {'name':'ADX','value':adx,'plusDI':pdi,'minusDI':mdi,'signal':'buy' if pdi>mdi and adx>25 else ('sell' if mdi>pdi and adx>25 else 'neutral')}

def calc_obv(closes, volumes):
    if len(closes)<10: return {'name':'OBV','value':0,'trend':'neutral','signal':'neutral'}
    obv,vals=0,[0]
    for i in range(1,len(closes)):
        if float(closes[i])>float(closes[i-1]): obv+=int(volumes[i])
        elif float(closes[i])<float(closes[i-1]): obv-=int(volumes[i])
        vals.append(obv)
    trend='up' if vals[-1]>vals[-min(10,len(vals)-1)] else 'down'
    return {'name':'OBV','value':si(abs(vals[-1])),'trend':trend,'signal':'buy' if trend=='up' else 'sell'}

def calc_williams_r(closes, highs, lows, period=14):
    if len(closes)<period: return {'name':'Williams %R','value':-50,'signal':'neutral'}
    hh=float(np.max(highs[-period:])); ll=float(np.min(lows[-period:])); cur=float(closes[-1])
    wr=sf(((hh-cur)/(hh-ll))*-100 if hh!=ll else -50)
    sig='buy' if wr<-80 else ('sell' if wr>-20 else 'neutral')
    return {'name':'Williams %R','value':wr,'signal':sig}

def calc_cci(closes, highs, lows, period=20):
    if len(closes)<period: return {'name':'CCI','value':0,'signal':'neutral'}
    tp=[(float(highs[i])+float(lows[i])+float(closes[i]))/3 for i in range(len(closes))]
    tp_r=tp[-period:]
    sma=np.mean(tp_r); md=np.mean(np.abs(np.array(tp_r)-sma))
    cci=sf((tp[-1]-sma)/(0.015*md) if md>0 else 0)
    sig='buy' if cci<-100 else ('sell' if cci>100 else 'neutral')
    return {'name':'CCI','value':cci,'signal':sig}

def calc_mfi(closes, highs, lows, volumes, period=14):
    if len(closes)<period+1: return {'name':'MFI','value':50,'signal':'neutral'}
    tp=[(float(highs[i])+float(lows[i])+float(closes[i]))/3 for i in range(len(closes))]
    pmf=nmf=0
    for i in range(-period,0):
        mf=tp[i]*float(volumes[i])
        if tp[i]>tp[i-1]: pmf+=mf
        else: nmf+=mf
    mfi=sf(100-(100/(1+pmf/nmf)) if nmf>0 else 100)
    sig='buy' if mfi<20 else ('sell' if mfi>80 else 'neutral')
    return {'name':'MFI','value':mfi,'signal':sig}

def calc_vwap(closes, highs, lows, volumes, period=20):
    if len(closes)<period: return {'name':'VWAP','value':0,'signal':'neutral'}
    tp=np.array([(float(highs[i])+float(lows[i])+float(closes[i]))/3 for i in range(-period,0)])
    vol=np.array([float(volumes[i]) for i in range(-period,0)])
    vwap=sf(np.sum(tp*vol)/np.sum(vol) if np.sum(vol)>0 else float(closes[-1]))
    cp=float(closes[-1])
    sig='buy' if cp>vwap else ('sell' if cp<vwap else 'neutral')
    return {'name':'VWAP','value':vwap,'signal':sig}

def calc_ichimoku(closes, highs, lows):
    n=len(closes)
    if n<52: return {'name':'Ichimoku','tenkan':0,'kijun':0,'signal':'neutral'}
    hh9=float(np.max(highs[-9:])); ll9=float(np.min(lows[-9:]))
    hh26=float(np.max(highs[-26:])); ll26=float(np.min(lows[-26:]))
    hh52=float(np.max(highs[-52:])); ll52=float(np.min(lows[-52:]))
    tenkan=sf((hh9+ll9)/2); kijun=sf((hh26+ll26)/2)
    ssa=sf((tenkan+kijun)/2); ssb=sf((hh52+ll52)/2)
    cp=float(closes[-1])
    if cp>ssa and cp>ssb and tenkan>kijun: sig='buy'
    elif cp<ssa and cp<ssb and tenkan<kijun: sig='sell'
    else: sig='neutral'
    return {'name':'Ichimoku','tenkan':tenkan,'kijun':kijun,'senkouA':ssa,'senkouB':ssb,'signal':sig}

def calc_psar(closes, highs, lows, af_start=0.02, af_step=0.02, af_max=0.2):
    n=len(closes)
    if n<5: return {'name':'Parabolic SAR','value':0,'trend':'neutral','signal':'neutral'}
    bull=True; af=af_start; ep=float(highs[0]); sar=float(lows[0])
    for i in range(1,n):
        hi,lo,cl=float(highs[i]),float(lows[i]),float(closes[i])
        prev_sar=sar; sar=prev_sar+af*(ep-prev_sar)
        if bull:
            if lo<sar: bull=False; sar=ep; ep=lo; af=af_start
            else:
                if hi>ep: ep=hi; af=min(af+af_step,af_max)
        else:
            if hi>sar: bull=True; sar=ep; ep=hi; af=af_start
            else:
                if lo<ep: ep=lo; af=min(af+af_step,af_max)
    trend='up' if bull else 'down'
    return {'name':'Parabolic SAR','value':sf(sar),'trend':trend,'signal':'buy' if bull else 'sell'}

def calc_roc(closes, period=12):
    if len(closes)<period+1: return {'name':'ROC','value':0,'signal':'neutral'}
    cur,prev=float(closes[-1]),float(closes[-period-1])
    roc=sf(((cur-prev)/prev)*100 if prev!=0 else 0)
    sig='buy' if roc>5 else ('sell' if roc<-5 else 'neutral')
    return {'name':'ROC','value':roc,'signal':sig,'period':period}

def calc_aroon(highs, lows, period=25):
    if len(highs)<period+1: return {'name':'Aroon','up':50,'down':50,'signal':'neutral'}
    h_slice=list(highs[-period-1:]); l_slice=list(lows[-period-1:])
    up=sf((h_slice.index(max(h_slice))/period)*100)
    down=sf((l_slice.index(min(l_slice))/period)*100)
    if up>70 and down<30: sig='buy'
    elif down>70 and up<30: sig='sell'
    else: sig='neutral'
    return {'name':'Aroon','up':up,'down':down,'signal':sig,'oscillator':sf(up-down)}

def calc_trix(closes, period=15):
    if len(closes)<period*3: return {'name':'TRIX','value':0,'signal':'neutral'}
    def ema_arr(data, p):
        result=[float(data[0])]; k=2/(p+1)
        for i in range(1,len(data)):
            result.append(float(data[i])*k + result[-1]*(1-k))
        return result
    e1=ema_arr(closes,period); e2=ema_arr(e1,period); e3=ema_arr(e2,period)
    if len(e3)<2 or abs(e3[-2])<1e-9: return {'name':'TRIX','value':0,'signal':'neutral'}
    trix=sf(((e3[-1]-e3[-2])/e3[-2])*10000)
    sig='buy' if trix>0 else ('sell' if trix<0 else 'neutral')
    return {'name':'TRIX','value':trix,'signal':sig}

def calc_dmi(highs, lows, closes, period=14):
    n=len(closes)
    if n<period+1: return {'name':'DMI','diPlus':0,'diMinus':0,'adx':0,'signal':'neutral'}
    pDM,nDM,tr_list=[],[],[]
    for i in range(1,n):
        hi,lo,cl=float(highs[i]),float(lows[i]),float(closes[i])
        phi,plo,pcl=float(highs[i-1]),float(lows[i-1]),float(closes[i-1])
        up_move=hi-phi; down_move=plo-lo
        pDM.append(up_move if up_move>down_move and up_move>0 else 0)
        nDM.append(down_move if down_move>up_move and down_move>0 else 0)
        tr_list.append(max(hi-lo,abs(hi-pcl),abs(lo-pcl)))
    if len(pDM)<period: return {'name':'DMI','diPlus':0,'diMinus':0,'adx':0,'signal':'neutral'}
    atr=np.mean(tr_list[-period:]); s_pDM=np.mean(pDM[-period:]); s_nDM=np.mean(nDM[-period:])
    diP=sf((s_pDM/atr)*100 if atr>0 else 0); diM=sf((s_nDM/atr)*100 if atr>0 else 0)
    dx=abs(diP-diM)/(diP+diM)*100 if (diP+diM)>0 else 0
    sig='buy' if diP>diM and dx>20 else ('sell' if diM>diP and dx>20 else 'neutral')
    return {'name':'DMI','diPlus':diP,'diMinus':diM,'adx':sf(dx),'signal':sig}


# =====================================================================
# DESTEK / DIRENC / FIBONACCI / PIVOT
# =====================================================================
def calc_support_resistance(hist):
    try:
        c,h,l=hist['Close'].values.astype(float),hist['High'].values.astype(float),hist['Low'].values.astype(float)
        h_clean = np.where(np.isnan(h), c, h)
        l_clean = np.where(np.isnan(l), c, l)
        n=min(90,len(c)); rh,rl=h_clean[-n:],l_clean[-n:]; sups,ress=[],[]
        for i in range(2,n-2):
            if rh[i]>rh[i-1] and rh[i]>rh[i-2] and rh[i]>rh[i+1] and rh[i]>rh[i+2]: ress.append(float(rh[i]))
            if rl[i]<rl[i-1] and rl[i]<rl[i-2] and rl[i]<rl[i+1] and rl[i]<rl[i+2]: sups.append(float(rl[i]))
        cur=float(c[-1])
        return {'supports':[sf(s) for s in sorted([s for s in sups if s<cur],reverse=True)[:3]],'resistances':[sf(r) for r in sorted([r for r in ress if r>cur])[:3]],'current':sf(cur)}
    except Exception: return {'supports':[],'resistances':[],'current':0}

def calc_fibonacci(hist):
    try:
        c=hist['Close'].values.astype(float)
        h=hist['High'].values.astype(float)
        l=hist['Low'].values.astype(float)
        for lookback in [90, 60, 30, len(c)]:
            n=min(lookback, len(c))
            if n < 5: continue
            rc, rh, rl = c[-n:], h[-n:], l[-n:]
            hi = float(np.nanmax(rh))
            lo = float(np.nanmin(rl))
            if hi != hi: hi = float(np.nanmax(rc))
            if lo != lo: lo = float(np.nanmin(rc))
            d = hi - lo
            if d > 0: break
        else:
            return {'levels':{},'currentZone':'-','trend':'-','description':'Yeterli veri yok'}

        mid_idx = n // 2
        first_half_avg = float(np.mean(rc[:mid_idx]))
        second_half_avg = float(np.mean(rc[mid_idx:]))
        trend = 'yukari' if second_half_avg > first_half_avg else 'asagi'

        if trend == 'yukari':
            levels = {
                '0.0 (Tepe)': sf(hi), '23.6': sf(hi - d * 0.236),
                '38.2': sf(hi - d * 0.382), '50.0': sf(hi - d * 0.5),
                '61.8': sf(hi - d * 0.618), '78.6': sf(hi - d * 0.786),
                '100.0 (Dip)': sf(lo)
            }
        else:
            levels = {
                '0.0 (Dip)': sf(lo), '23.6': sf(lo + d * 0.236),
                '38.2': sf(lo + d * 0.382), '50.0': sf(lo + d * 0.5),
                '61.8': sf(lo + d * 0.618), '78.6': sf(lo + d * 0.786),
                '100.0 (Tepe)': sf(hi)
            }

        cur = float(c[-1])
        zone = "Belirsiz"
        lk = list(levels.keys())
        lv = list(levels.values())
        sorted_vals = sorted([(lk[i], lv[i]) for i in range(len(lv))], key=lambda x: -x[1])
        for i in range(len(sorted_vals) - 1):
            if cur <= sorted_vals[i][1] and cur >= sorted_vals[i+1][1]:
                zone = f"{sorted_vals[i][0]} - {sorted_vals[i+1][0]}"
                break

        nearest_support = None
        nearest_resistance = None
        for k, v in sorted(levels.items(), key=lambda x: x[1]):
            if v < cur:
                nearest_support = {'level': k, 'price': v}
            elif v > cur and nearest_resistance is None:
                nearest_resistance = {'level': k, 'price': v}

        desc_parts = [f"Son {n} barda analiz"]
        desc_parts.append(f"Trend: {'Yukari' if trend == 'yukari' else 'Asagi'}")
        desc_parts.append(f"Aralik: {sf(lo)} - {sf(hi)} ({sf(d)} TL fark)")
        if nearest_support:
            desc_parts.append(f"En yakin destek: {nearest_support['price']} TL ({nearest_support['level']})")
        if nearest_resistance:
            desc_parts.append(f"En yakin direnc: {nearest_resistance['price']} TL ({nearest_resistance['level']})")
        pos_pct = sf((cur - lo) / d * 100) if d > 0 else 50
        desc_parts.append(f"Fiyat araliktaki konum: %{pos_pct}")

        return {
            'levels': levels, 'high': sf(hi), 'low': sf(lo),
            'currentZone': zone, 'trend': trend,
            'nearestSupport': nearest_support, 'nearestResistance': nearest_resistance,
            'positionPct': pos_pct, 'lookbackBars': n,
            'description': ' | '.join(desc_parts)
        }
    except Exception as e:
        print(f"  [FIB] Hata: {e}")
        return {'levels':{},'currentZone':'-','trend':'-','description':'Hesaplama hatasi'}

def calc_pivot_points(hist):
    """Klasik, Camarilla, Woodie pivot noktalari"""
    try:
        c=float(hist['Close'].iloc[-1])
        h=float(hist['High'].iloc[-1])
        l=float(hist['Low'].iloc[-1])
        o=float(hist['Open'].iloc[-1])
        if h != h: h = c
        if l != l: l = c
        if o != o: o = c
        pp=(h+l+c)/3
        classic={
            'pp':sf(pp),'r1':sf(2*pp-l),'r2':sf(pp+(h-l)),'r3':sf(h+2*(pp-l)),
            's1':sf(2*pp-h),'s2':sf(pp-(h-l)),'s3':sf(l-2*(h-pp))
        }
        d=h-l
        camarilla={
            'pp':sf(pp),'r1':sf(c+d*1.1/12),'r2':sf(c+d*1.1/6),'r3':sf(c+d*1.1/4),'r4':sf(c+d*1.1/2),
            's1':sf(c-d*1.1/12),'s2':sf(c-d*1.1/6),'s3':sf(c-d*1.1/4),'s4':sf(c-d*1.1/2)
        }
        wpp=(h+l+2*c)/4
        woodie={
            'pp':sf(wpp),'r1':sf(2*wpp-l),'r2':sf(wpp+(h-l)),
            's1':sf(2*wpp-h),'s2':sf(wpp-(h-l))
        }
        return {'classic':classic,'camarilla':camarilla,'woodie':woodie,'current':sf(c)}
    except Exception: return {'classic':{},'camarilla':{},'woodie':{},'current':0}
