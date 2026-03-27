"""
BIST Pro - Technical Indicators Module
All technical analysis calculation functions.
"""
import numpy as np
import pandas as pd
from config import sf, si


# =====================================================================
# ORTAK HELPER FONKSIYONLAR
# =====================================================================

def _extract_ohlcv(hist):
    """DataFrame'den OHLCV array'lerini cikar ve NaN temizle"""
    c = hist['Close'].values.astype(float)
    h = hist['High'].values.astype(float) if 'High' in hist.columns else c.copy()
    l = hist['Low'].values.astype(float) if 'Low' in hist.columns else c.copy()
    v = hist['Volume'].values.astype(float) if 'Volume' in hist.columns else np.zeros(len(c))
    o = hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
    h = np.where(np.isnan(h), c, h)
    l = np.where(np.isnan(l), c, l)
    v = np.where(np.isnan(v), 0, v)
    o = np.where(np.isnan(o), c, o)
    return o, h, l, c, v


def _resample_to_tf(hist_daily, tf):
    """
    Gunluk OHLCV verisini haftalik ('weekly') veya aylik ('monthly') bara donustur.
    Portabl yaklasim: pandas groupby + Period kullanır (resample versiyonuna bagli degil).
    """
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

        # Son bar tamamlanmamis olabilir (suanki hafta/ay) — cikar
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

EMA_PERIODS = [5, 10, 20, 50, 100, 200]

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
    # Smooth %D (3-period SMA of %K)
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

def calc_support_resistance(hist):
    try:
        c,h,l=hist['Close'].values.astype(float),hist['High'].values.astype(float),hist['Low'].values.astype(float)
        # NaN temizligi: NaN degerleri Close ile degistir
        h_clean = np.where(np.isnan(h), c, h)
        l_clean = np.where(np.isnan(l), c, l)
        n=min(90,len(c)); rh,rl=h_clean[-n:],l_clean[-n:]; sups,ress=[],[]
        for i in range(2,n-2):
            if rh[i]>rh[i-1] and rh[i]>rh[i-2] and rh[i]>rh[i+1] and rh[i]>rh[i+2]: ress.append(float(rh[i]))
            if rl[i]<rl[i-1] and rl[i]<rl[i-2] and rl[i]<rl[i+1] and rl[i]<rl[i+2]: sups.append(float(rl[i]))
        cur=float(c[-1])
        return {'supports':[sf(s) for s in sorted([s for s in sups if s<cur],reverse=True)[:3]],'resistances':[sf(r) for r in sorted([r for r in ress if r>cur])[:3]],'current':sf(cur)}
    except Exception: return {'supports':[],'resistances':[],'current':0}

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
    """Rate of Change - momentum osilatoru"""
    if len(closes)<period+1: return {'name':'ROC','value':0,'signal':'neutral'}
    cur,prev=float(closes[-1]),float(closes[-period-1])
    roc=sf(((cur-prev)/prev)*100 if prev!=0 else 0)
    sig='buy' if roc>5 else ('sell' if roc<-5 else 'neutral')
    return {'name':'ROC','value':roc,'signal':sig,'period':period}

def calc_aroon(highs, lows, period=25):
    """Aroon Up/Down - trend yonu gostergesi"""
    if len(highs)<period+1: return {'name':'Aroon','up':50,'down':50,'signal':'neutral'}
    h_slice=list(highs[-period-1:]); l_slice=list(lows[-period-1:])
    # h_slice[0]=oldest, h_slice[period]=newest; Aroon Up = (index_of_max / period) * 100
    up=sf((h_slice.index(max(h_slice))/period)*100)
    down=sf((l_slice.index(min(l_slice))/period)*100)
    if up>70 and down<30: sig='buy'
    elif down>70 and up<30: sig='sell'
    else: sig='neutral'
    return {'name':'Aroon','up':up,'down':down,'signal':sig,'oscillator':sf(up-down)}

def calc_trix(closes, period=15):
    """TRIX - triple smoothed EMA oscillator"""
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
    """Directional Movement - trend gucu (ADX'in detayli hali)"""
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


def calc_all_indicators(hist, cp):
    c,h,l,v=hist['Close'].values.astype(float),hist['High'].values.astype(float),hist['Low'].values.astype(float),hist['Volume'].values.astype(float)
    o=hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
    # NaN temizligi
    h=np.where(np.isnan(h), c, h); l=np.where(np.isnan(l), c, l)
    v=np.where(np.isnan(v), 0, v); o=np.where(np.isnan(o), c, o)
    cp=float(cp)
    rsi_h=[{'date':hist.index[i].strftime('%Y-%m-%d'),'value':rv} for i in range(14,len(c)) if (rv:=calc_rsi_single(c[:i+1])) is not None]

    # Dinamik esikler hesapla
    dyn_thresholds = calc_dynamic_thresholds(c, h, l, v)

    # RSI'yi dinamik esiklerle hesapla
    rsi_data = calc_rsi(c)
    rsi_val = rsi_data.get('value', 50)
    dyn_oversold = float(dyn_thresholds.get('rsi_oversold', 30))
    dyn_overbought = float(dyn_thresholds.get('rsi_overbought', 70))
    if rsi_val < dyn_oversold:
        rsi_data['signal'] = 'buy'
        rsi_data['dynamicNote'] = f'Dinamik esik: <{dyn_oversold}'
    elif rsi_val > dyn_overbought:
        rsi_data['signal'] = 'sell'
        rsi_data['dynamicNote'] = f'Dinamik esik: >{dyn_overbought}'
    rsi_data['dynamicOversold'] = dyn_oversold
    rsi_data['dynamicOverbought'] = dyn_overbought

    ind={
        'rsi':rsi_data,'rsiHistory':rsi_h,
        'macd':calc_macd(c),'macdHistory':calc_macd_history(c),
        'bollinger':calc_bollinger(c,cp),'bollingerHistory':calc_bollinger_history(c),
        'stochastic':calc_stochastic(c,h,l),'stochasticHistory':calc_stochastic_history(c,h,l),
        'ema':calc_ema(c,cp),'emaHistory':calc_ema_history(c),
        'atr':calc_atr(h,l,c),
        'adx':calc_adx(h,l,c),
        'obv':calc_obv(c,v),
        'williamsR':calc_williams_r(c,h,l),
        'cci':calc_cci(c,h,l),
        'mfi':calc_mfi(c,h,l,v),
        'vwap':calc_vwap(c,h,l,v),
        'ichimoku':calc_ichimoku(c,h,l),
        'psar':calc_psar(c,h,l),
        'roc':calc_roc(c),
        'aroon':calc_aroon(h,l),
        'trix':calc_trix(c),
        'dmi':calc_dmi(h,l,c),
        'candlestick':calc_candlestick_patterns(o,h,l,c),
        'dynamicThresholds':dyn_thresholds,
    }
    sigs=[x.get('signal','neutral') for x in ind.values() if isinstance(x,dict) and 'signal' in x]
    bc,sc=sigs.count('buy'),sigs.count('sell'); t=len(sigs)
    ind['summary']={'overall':'buy' if bc>sc and bc>=t*0.4 else ('sell' if sc>bc and sc>=t*0.4 else 'neutral'),'buySignals':bc,'sellSignals':sc,'neutralSignals':t-bc-sc,'totalIndicators':t}
    return ind



def calc_mtf_signal(hist_daily):
    """
    Gercek coklu zaman dilimi sinyali:
      - daily  : mevcut gunluk bar verisi
      - weekly : gunluk veriyi haftalik bara resample et
      - monthly: gunluk veriyi aylik bara resample et
    Her zaman dilimi icin RSI / MACD / EMA / Bollinger → al/sat/notr karar.
    Kac tanesinin ayni yonde oldugunu say → mtfScore (0-3).
    """
    def _tf_signal(hist):
        """Bir OHLCV DataFrame'i icin basit al/sat/notr uret"""
        if hist is None or len(hist) < 10:
            return {'signal': 'neutral', 'score': 0, 'rsi': 50,
                    'macd': 'neutral', 'ema': 'neutral', 'bars': 0}
        try:
            c = hist['Close'].values.astype(float)
            h = hist['High'].values.astype(float) if 'High' in hist.columns else c.copy()
            l = hist['Low'].values.astype(float) if 'Low' in hist.columns else c.copy()
            h = np.where(np.isnan(h), c, h)
            l = np.where(np.isnan(l), c, l)
            n = len(c)
            score = 0

            # RSI
            rsi_d   = calc_rsi(c)
            rsi_val = float(rsi_d.get('value', 50))
            if   rsi_val < 35: score += 2
            elif rsi_val < 45: score += 1
            elif rsi_val > 65: score -= 2
            elif rsi_val > 55: score -= 1

            # MACD (histogram yonu)
            macd_sig = 'neutral'
            if n >= 26:
                md = calc_macd(c)
                hist_val = float(md.get('histogram', 0))
                if   hist_val > 0: score += 1; macd_sig = 'buy'
                elif hist_val < 0: score -= 1; macd_sig = 'sell'

            # EMA 20 / 50 hizalaması
            ema_sig = 'neutral'
            if n >= 50:
                s     = pd.Series(c)
                e20   = float(s.ewm(span=20, adjust=False).mean().iloc[-1])
                e50   = float(s.ewm(span=50, adjust=False).mean().iloc[-1])
                cur   = float(c[-1])
                if   cur > e20 and e20 > e50: score += 1; ema_sig = 'buy'
                elif cur < e20 and e20 < e50: score -= 1; ema_sig = 'sell'

            # Bollinger bantları
            if n >= 20:
                bb  = calc_bollinger(c, float(c[-1]))
                bbl = float(bb.get('lower', 0))
                bbu = float(bb.get('upper', 0))
                cp  = float(c[-1])
                if bbl > 0 and cp < bbl: score += 1
                elif bbu > 0 and cp > bbu: score -= 1

            signal = 'buy' if score >= 2 else ('sell' if score <= -2 else 'neutral')
            return {
                'signal': signal, 'score': sf(score),
                'rsi': sf(rsi_val), 'macd': macd_sig, 'ema': ema_sig,
                'bars': n, 'currentPrice': sf(float(c[-1])),
            }
        except Exception as e:
            return {'signal': 'neutral', 'score': 0, 'rsi': 50,
                    'macd': 'neutral', 'ema': 'neutral', 'bars': 0, 'error': str(e)}

    try:
        daily_sig   = _tf_signal(hist_daily)
        weekly_sig  = _tf_signal(_resample_to_tf(hist_daily, 'weekly'))
        monthly_sig = _tf_signal(_resample_to_tf(hist_daily, 'monthly'))

        sigs = [daily_sig['signal'], weekly_sig['signal'], monthly_sig['signal']]
        buy_c  = sigs.count('buy')
        sell_c = sigs.count('sell')

        if   buy_c >= 2:  dominant = 'buy';  mtf_score = buy_c
        elif sell_c >= 2: dominant = 'sell'; mtf_score = sell_c
        else:             dominant = 'neutral'; mtf_score = 0

        alignment = f'{max(buy_c, sell_c)}/3'
        strength  = ('Guclu' if max(buy_c, sell_c) == 3
                     else ('Orta' if max(buy_c, sell_c) == 2 else 'Uyumsuz'))

        return {
            'daily':   daily_sig,
            'weekly':  weekly_sig,
            'monthly': monthly_sig,
            'mtfScore':     mtf_score,
            'mtfAlignment': alignment,
            'mtfDirection': dominant,
            'mtfStrength':  strength,
            'description': (
                f'Gunluk: {daily_sig["signal"]} | '
                f'Haftalik: {weekly_sig["signal"]} | '
                f'Aylik: {monthly_sig["signal"]} '
                f'→ {alignment} uyum ({strength})'
            ),
        }
    except Exception as e:
        print(f"  [MTF] Hata: {e}")
        return {
            'daily': {'signal': 'neutral'}, 'weekly': {'signal': 'neutral'},
            'monthly': {'signal': 'neutral'}, 'mtfScore': 0,
            'mtfAlignment': '0/3', 'mtfDirection': 'neutral',
            'mtfStrength': 'Uyumsuz', 'error': str(e),
        }


# =====================================================================
# FAZ 3: UYUMSUZLUK (DIVERGENCE) TESPİTİ
# RSI ve MACD tabanlı klasik/gizli divergence tespiti.
# =====================================================================
def _rsi_series(closes, period=14):
    """Wilder yumuşatma ile tam RSI serisi (vektörel)"""
    c = np.array(closes, dtype=float)
    if len(c) < period + 1:
        return np.full(len(c), 50.0)
    delta = np.diff(c)
    gains  = np.where(delta > 0,  delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_g  = np.zeros(len(delta))
    avg_l  = np.zeros(len(delta))
    avg_g[period - 1] = np.mean(gains[:period])
    avg_l[period - 1] = np.mean(losses[:period])
    for i in range(period, len(delta)):
        avg_g[i] = (avg_g[i-1] * (period-1) + gains[i])  / period
        avg_l[i] = (avg_l[i-1] * (period-1) + losses[i]) / period
    rs  = np.where(avg_l == 0, np.inf, avg_g / avg_l)
    rsi = np.where(avg_l == 0, 100.0, 100 - 100 / (1 + rs))
    result = np.full(len(c), np.nan)
    result[period:] = rsi[period - 1:]
    return result

def _find_peaks(arr, window=5):
    """Lokal zirve indekslerini döndür"""
    peaks = []
    for i in range(window, len(arr) - window):
        if arr[i] == max(arr[i-window:i+window+1]):
            peaks.append(i)
    return peaks

def _find_troughs(arr, window=5):
    """Lokal dip indekslerini döndür"""
    troughs = []
    for i in range(window, len(arr) - window):
        if arr[i] == min(arr[i-window:i+window+1]):
            troughs.append(i)
    return troughs

def calc_divergence(hist, lookback=90):
    """
    RSI + MACD uyumsuzluk (divergence) tespiti.
      Regular Bullish : Fiyat LL, RSI HL  → Al
      Regular Bearish : Fiyat HH, RSI LH  → Sat
      Hidden Bullish  : Fiyat HL, RSI LL  → Uptrend devam (Al)
      Hidden Bearish  : Fiyat LH, RSI HH  → Downtrend devam (Sat)
    MACD histogram uyumsuzlukları da eklenir.
    """
    try:
        c  = hist['Close'].values.astype(float)
        n  = len(c)
        if n < 50:
            return {'divergences': [], 'recentDivergences': [],
                    'summary': {'bullish': 0, 'bearish': 0, 'signal': 'neutral', 'count': 0, 'hasRecent': False}}

        lb   = min(lookback, n)
        c_lb = c[-lb:]

        # RSI serisi
        rsi_arr  = _rsi_series(c_lb)
        rsi_vals = np.where(np.isnan(rsi_arr), 50.0, rsi_arr)

        # MACD histogram serisi
        s          = pd.Series(c_lb, dtype=float)
        macd_line  = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
        sig_line   = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist_arr = (macd_line - sig_line).values

        divergences = []
        window = 5

        price_peaks   = _find_peaks(c_lb, window)
        price_troughs = _find_troughs(c_lb, window)

        # --- Regular Bearish: Fiyat HH, RSI LH ---
        if len(price_peaks) >= 2:
            p1, p2 = price_peaks[-2], price_peaks[-1]
            if c_lb[p2] > c_lb[p1] and rsi_vals[p2] < rsi_vals[p1]:
                divergences.append({
                    'type': 'regular_bearish', 'label': 'Klasik Ayı Uyumsuzluğu', 'signal': 'sell',
                    'description': f'Fiyat yeni zirve ({sf(c_lb[p2])}) ama RSI düşüyor '
                                   f'({sf(rsi_vals[p2])} < {sf(rsi_vals[p1])})',
                    'strength': sf(abs(rsi_vals[p1] - rsi_vals[p2])),
                    'recency': int(lb - p2),
                    'priceBar1': sf(c_lb[p1]), 'priceBar2': sf(c_lb[p2]),
                    'rsiBar1':   sf(rsi_vals[p1]), 'rsiBar2': sf(rsi_vals[p2]),
                })

        # --- Regular Bullish: Fiyat LL, RSI HL ---
        if len(price_troughs) >= 2:
            t1, t2 = price_troughs[-2], price_troughs[-1]
            if c_lb[t2] < c_lb[t1] and rsi_vals[t2] > rsi_vals[t1]:
                divergences.append({
                    'type': 'regular_bullish', 'label': 'Klasik Boğa Uyumsuzluğu', 'signal': 'buy',
                    'description': f'Fiyat yeni dip ({sf(c_lb[t2])}) ama RSI yükseliyor '
                                   f'({sf(rsi_vals[t2])} > {sf(rsi_vals[t1])})',
                    'strength': sf(abs(rsi_vals[t2] - rsi_vals[t1])),
                    'recency': int(lb - t2),
                    'priceBar1': sf(c_lb[t1]), 'priceBar2': sf(c_lb[t2]),
                    'rsiBar1':   sf(rsi_vals[t1]), 'rsiBar2': sf(rsi_vals[t2]),
                })

        # --- Hidden Bullish: Fiyat HL, RSI LL (uptrend devam) ---
        if len(price_troughs) >= 2:
            t1, t2 = price_troughs[-2], price_troughs[-1]
            if c_lb[t2] > c_lb[t1] and rsi_vals[t2] < rsi_vals[t1]:
                divergences.append({
                    'type': 'hidden_bullish', 'label': 'Gizli Boğa Uyumsuzluğu', 'signal': 'buy',
                    'description': f'Fiyat yüksek dip ({sf(c_lb[t2])}) ama RSI düşük → Uptrend devam',
                    'strength': sf(abs(rsi_vals[t2] - rsi_vals[t1])),
                    'recency': int(lb - t2),
                    'priceBar1': sf(c_lb[t1]), 'priceBar2': sf(c_lb[t2]),
                    'rsiBar1':   sf(rsi_vals[t1]), 'rsiBar2': sf(rsi_vals[t2]),
                })

        # --- Hidden Bearish: Fiyat LH, RSI HH (downtrend devam) ---
        if len(price_peaks) >= 2:
            p1, p2 = price_peaks[-2], price_peaks[-1]
            if c_lb[p2] < c_lb[p1] and rsi_vals[p2] > rsi_vals[p1]:
                divergences.append({
                    'type': 'hidden_bearish', 'label': 'Gizli Ayı Uyumsuzluğu', 'signal': 'sell',
                    'description': f'Fiyat düşük zirve ({sf(c_lb[p2])}) ama RSI yükseliyor → Downtrend devam',
                    'strength': sf(abs(rsi_vals[p2] - rsi_vals[p1])),
                    'recency': int(lb - p2),
                    'priceBar1': sf(c_lb[p1]), 'priceBar2': sf(c_lb[p2]),
                    'rsiBar1':   sf(rsi_vals[p1]), 'rsiBar2': sf(rsi_vals[p2]),
                })

        # --- MACD Bearish Divergence ---
        if len(price_peaks) >= 2:
            p1, p2 = price_peaks[-2], price_peaks[-1]
            if (p1 < len(macd_hist_arr) and p2 < len(macd_hist_arr) and
                    c_lb[p2] > c_lb[p1] and macd_hist_arr[p2] < macd_hist_arr[p1]):
                divergences.append({
                    'type': 'macd_bearish', 'label': 'MACD Ayı Uyumsuzluğu', 'signal': 'sell',
                    'description': f'Fiyat HH ama MACD histogram düşük '
                                   f'({sf(float(macd_hist_arr[p2]))} < {sf(float(macd_hist_arr[p1]))})',
                    'strength': sf(abs(float(macd_hist_arr[p1]) - float(macd_hist_arr[p2]))),
                    'recency': int(lb - p2),
                })

        # --- MACD Bullish Divergence ---
        if len(price_troughs) >= 2:
            t1, t2 = price_troughs[-2], price_troughs[-1]
            if (t1 < len(macd_hist_arr) and t2 < len(macd_hist_arr) and
                    c_lb[t2] < c_lb[t1] and macd_hist_arr[t2] > macd_hist_arr[t1]):
                divergences.append({
                    'type': 'macd_bullish', 'label': 'MACD Boğa Uyumsuzluğu', 'signal': 'buy',
                    'description': f'Fiyat LL ama MACD histogram yükseliyor '
                                   f'({sf(float(macd_hist_arr[t2]))} > {sf(float(macd_hist_arr[t1]))})',
                    'strength': sf(abs(float(macd_hist_arr[t2]) - float(macd_hist_arr[t1]))),
                    'recency': int(lb - t2),
                })

        recent    = [d for d in divergences if d.get('recency', 999) <= 20]
        bull_cnt  = sum(1 for d in divergences if d['signal'] == 'buy')
        bear_cnt  = sum(1 for d in divergences if d['signal'] == 'sell')
        overall   = 'buy' if bull_cnt > bear_cnt else ('sell' if bear_cnt > bull_cnt else 'neutral')

        return {
            'divergences':       divergences,
            'recentDivergences': recent,
            'summary': {
                'bullish':   bull_cnt,
                'bearish':   bear_cnt,
                'signal':    overall,
                'count':     len(divergences),
                'hasRecent': len(recent) > 0,
            },
            'currentRsi':      sf(float(rsi_vals[-1])),
            'currentMacdHist': sf(float(macd_hist_arr[-1])),
        }
    except Exception as e:
        print(f"  [DIVERGENCE] Hata: {e}")
        return {'divergences': [], 'recentDivergences': [],
                'summary': {'bullish': 0, 'bearish': 0, 'signal': 'neutral', 'count': 0, 'hasRecent': False},
                'error': str(e)}


# =====================================================================
# FAZ 4: HACİM PROFİLİ & VWAP
# POC / VAH / VAL (Hacim Profili), VWAP, Hacim Anomali tespiti
# =====================================================================
def calc_volume_profile(hist, bins=20):
    """
    Hacim Profili ve VWAP analizi:
      VWAP : Hacimle ağırlıklı ortalama fiyat
      POC  : Point of Control — en yüksek hacim bin'i
      VAH  : Value Area High  — hacmin %70'i üst sınırı
      VAL  : Value Area Low   — hacmin %70'i alt sınırı
      Anomaly: Son mum hacmi 20 günlük ortalamanın 2x üzerindeyse uyar
    """
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)  if 'High'   in hist.columns else c.copy()
        l = hist['Low'].values.astype(float)   if 'Low'    in hist.columns else c.copy()
        v = hist['Volume'].values.astype(float) if 'Volume' in hist.columns else np.ones(len(c))
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        v = np.where(np.isnan(v) | (v <= 0), 0, v)
        n = len(c)

        # VWAP
        typical   = (h + l + c) / 3
        cum_vol   = np.cumsum(v)
        cum_tpv   = np.cumsum(typical * v)
        vwap_ser  = np.where(cum_vol > 0, cum_tpv / cum_vol, typical)
        vwap      = float(vwap_ser[-1])
        cur_price = float(c[-1])
        vwap_pct  = sf((cur_price - vwap) / vwap * 100) if vwap > 0 else 0
        vwap_sig  = ('buy' if cur_price < vwap * 0.99
                     else ('sell' if cur_price > vwap * 1.01 else 'neutral'))

        # Fiyat aralığı → bins
        price_min = float(np.min(l))
        price_max = float(np.max(h))
        if price_max <= price_min:
            price_max = price_min * 1.01
        bin_edges   = np.linspace(price_min, price_max, bins + 1)
        bin_volumes = np.zeros(bins)

        for i in range(n):
            bar_range = h[i] - l[i]
            if bar_range <= 0:
                idx = min(max(int(np.searchsorted(bin_edges, c[i], side='right') - 1), 0), bins - 1)
                bin_volumes[idx] += v[i]
            else:
                for b in range(bins):
                    ov_lo = max(l[i], bin_edges[b])
                    ov_hi = min(h[i], bin_edges[b + 1])
                    if ov_hi > ov_lo:
                        bin_volumes[b] += v[i] * (ov_hi - ov_lo) / bar_range

        # POC
        poc_idx    = int(np.argmax(bin_volumes))
        poc_price  = sf((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)
        poc_volume = sf(bin_volumes[poc_idx])

        # Value Area (toplam hacmin %70'i)
        total_vol  = float(np.sum(bin_volumes))
        va_target  = total_vol * 0.70
        va_vol     = bin_volumes[poc_idx]
        lo, hi     = poc_idx, poc_idx

        while va_vol < va_target and (lo > 0 or hi < bins - 1):
            add_lo = bin_volumes[lo - 1] if lo > 0        else 0.0
            add_hi = bin_volumes[hi + 1] if hi < bins - 1 else 0.0
            if add_hi >= add_lo and hi < bins - 1:
                hi += 1; va_vol += bin_volumes[hi]
            elif lo > 0:
                lo -= 1; va_vol += bin_volumes[lo]
            else:
                hi += 1; va_vol += bin_volumes[hi]

        vah = sf((bin_edges[hi] + bin_edges[hi + 1]) / 2)
        val = sf((bin_edges[lo] + bin_edges[lo + 1]) / 2)

        # Hacim anomalisi
        avg_vol_20  = float(np.mean(v[-20:])) if n >= 20 else float(np.mean(v))
        last_vol    = float(v[-1])
        vol_ratio   = sf(last_vol / avg_vol_20) if avg_vol_20 > 0 else 0
        vol_anomaly = last_vol > avg_vol_20 * 2

        # Hacim trendi (son 3 vs önceki 3)
        vol_trend = ('artiyor' if n >= 6 and float(np.mean(v[-3:])) > float(np.mean(v[-6:-3]))
                     else 'azaliyor')

        # Frontend için profil listesi
        profile = [
            {
                'priceLevel': sf((bin_edges[i] + bin_edges[i + 1]) / 2),
                'volume':     sf(bin_volumes[i]),
                'isPOC':      i == poc_idx,
                'isVAH':      i == hi,
                'isVAL':      i == lo,
                'inValueArea': lo <= i <= hi,
            }
            for i in range(bins)
        ]

        return {
            'vwap':         sf(vwap),
            'vwapSignal':   vwap_sig,
            'vwapPct':      vwap_pct,
            'poc':          poc_price,
            'pocVolume':    poc_volume,
            'vah':          vah,
            'val':          val,
            'profile':      profile,
            'volumeAnomaly': vol_anomaly,
            'volumeRatio':  vol_ratio,
            'volumeTrend':  vol_trend,
            'avgVolume20':  sf(avg_vol_20),
            'lastVolume':   sf(last_vol),
            'currentPrice': sf(cur_price),
            'priceVsVwap':  vwap_pct,
            'priceVsVAH':   sf((cur_price - float(vah)) / float(vah) * 100) if float(vah) > 0 else 0,
            'priceVsVAL':   sf((cur_price - float(val)) / float(val) * 100) if float(val) > 0 else 0,
            'priceVsPOC':   sf((cur_price - float(poc_price)) / float(poc_price) * 100) if float(poc_price) > 0 else 0,
        }
    except Exception as e:
        print(f"  [VOL-PROFILE] Hata: {e}")
        return {'error': str(e), 'vwap': 0, 'poc': 0, 'vah': 0, 'val': 0,
                'volumeAnomaly': False, 'volumeRatio': 0}


# =====================================================================
# FAZ 5: SMART MONEY CONCEPTS (SMC)
# FVG (Fair Value Gap), Order Block, BOS, CHoCH tespiti
# =====================================================================
def calc_smc(hist, lookback=120):
    """
    Smart Money Concepts (SMC) analizi:
      FVG   : 3-mum imbalance bölgeleri (dolmamış boşluklar)
      OB    : Order Block — kurumsal momentum öncesi zıt mum
      BOS   : Break of Structure — swing high/low kırılması
      CHoCH : Change of Character — karşı yönlü BOS (trend değişimi)
    """
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)  if 'High'  in hist.columns else c.copy()
        l = hist['Low'].values.astype(float)   if 'Low'   in hist.columns else c.copy()
        o = hist['Open'].values.astype(float)  if 'Open'  in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        o = np.where(np.isnan(o), c, o)
        n = len(c)
        lb = min(lookback, n)
        c_lb, h_lb, l_lb, o_lb = c[-lb:], h[-lb:], l[-lb:], o[-lb:]

        # ---- Fair Value Gaps (FVG) ----
        fvgs = []
        for i in range(2, lb):
            # Bullish FVG: mum[i-2].high < mum[i].low
            if h_lb[i - 2] < l_lb[i]:
                gap_pct = (l_lb[i] - h_lb[i - 2]) / h_lb[i - 2] * 100
                filled  = float(np.min(l_lb[i:])) < h_lb[i - 2]
                fvgs.append({
                    'type': 'bullish_fvg', 'label': 'Boğa FVG',
                    'top':      sf(l_lb[i]),
                    'bottom':   sf(h_lb[i - 2]),
                    'midpoint': sf((l_lb[i] + h_lb[i - 2]) / 2),
                    'sizePct':  sf(gap_pct),
                    'filled':   filled,
                    'barsAgo':  int(lb - i),
                })
            # Bearish FVG: mum[i-2].low > mum[i].high
            elif l_lb[i - 2] > h_lb[i]:
                gap_pct = (l_lb[i - 2] - h_lb[i]) / h_lb[i] * 100
                filled  = float(np.max(h_lb[i:])) > l_lb[i - 2]
                fvgs.append({
                    'type': 'bearish_fvg', 'label': 'Ayı FVG',
                    'top':      sf(l_lb[i - 2]),
                    'bottom':   sf(h_lb[i]),
                    'midpoint': sf((l_lb[i - 2] + h_lb[i]) / 2),
                    'sizePct':  sf(gap_pct),
                    'filled':   filled,
                    'barsAgo':  int(lb - i),
                })

        # Dolmamış, son 30 bardaki FVG'ler (en yakın 5'i)
        active_fvgs = sorted(
            [f for f in fvgs if not f['filled'] and f['barsAgo'] <= 30],
            key=lambda x: x['barsAgo']
        )[:5]

        # ---- Order Blocks (OB) ----
        obs = []
        imp_thr = 0.015  # %1.5 impulse eşiği
        for i in range(1, lb - 3):
            # Bullish OB: Ayı mumu + ardından güçlü yukarı hareket
            if c_lb[i] < o_lb[i]:
                nxt_hi = float(np.max(h_lb[i + 1:min(i + 4, lb)]))
                if (nxt_hi - c_lb[i]) / c_lb[i] > imp_thr:
                    obs.append({
                        'type': 'bullish_ob', 'label': 'Boğa Order Block',
                        'top':    sf(max(o_lb[i], c_lb[i])),
                        'bottom': sf(min(o_lb[i], c_lb[i])),
                        'midpoint': sf((o_lb[i] + c_lb[i]) / 2),
                        'impulseStrength': sf((nxt_hi - c_lb[i]) / c_lb[i] * 100),
                        'barsAgo': int(lb - i),
                    })
            # Bearish OB: Boğa mumu + ardından güçlü aşağı hareket
            elif c_lb[i] > o_lb[i]:
                nxt_lo = float(np.min(l_lb[i + 1:min(i + 4, lb)]))
                if (c_lb[i] - nxt_lo) / c_lb[i] > imp_thr:
                    obs.append({
                        'type': 'bearish_ob', 'label': 'Ayı Order Block',
                        'top':    sf(max(o_lb[i], c_lb[i])),
                        'bottom': sf(min(o_lb[i], c_lb[i])),
                        'midpoint': sf((o_lb[i] + c_lb[i]) / 2),
                        'impulseStrength': sf((c_lb[i] - nxt_lo) / c_lb[i] * 100),
                        'barsAgo': int(lb - i),
                    })

        # Son 40 bardaki OB'lar (en yakın 5'i)
        recent_obs = sorted(
            [ob for ob in obs if ob['barsAgo'] <= 40],
            key=lambda x: x['barsAgo']
        )[:5]

        # ---- Break of Structure (BOS) ----
        swing_highs = _find_peaks(h_lb, 5)
        swing_lows  = _find_troughs(l_lb, 5)
        bos_events  = []
        structure_trend = 'neutral'

        if len(swing_highs) >= 1:
            last_sh = swing_highs[-1]
            if c_lb[-1] > h_lb[last_sh]:
                bos_events.append({
                    'type': 'bullish_bos', 'label': 'Yukarı BOS',
                    'description': f'Kapanış ({sf(c_lb[-1])}) swing high kırdı ({sf(h_lb[last_sh])})',
                    'level': sf(h_lb[last_sh]),
                    'barsAgo': int(lb - 1 - last_sh),
                })
                structure_trend = 'bullish'

        if len(swing_lows) >= 1:
            last_sl = swing_lows[-1]
            if c_lb[-1] < l_lb[last_sl]:
                bos_events.append({
                    'type': 'bearish_bos', 'label': 'Aşağı BOS',
                    'description': f'Kapanış ({sf(c_lb[-1])}) swing low kırdı ({sf(l_lb[last_sl])})',
                    'level': sf(l_lb[last_sl]),
                    'barsAgo': int(lb - 1 - last_sl),
                })
                structure_trend = 'bearish'

        # ---- Change of Character (CHoCH) ----
        choch_events = []
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            sh1, sh2 = swing_highs[-2], swing_highs[-1]
            sl1, sl2 = swing_lows[-2],  swing_lows[-1]
            # Bullish CHoCH: Downtrend (LH + LL) iken swing high kırılması
            if h_lb[sh2] < h_lb[sh1] and l_lb[sl2] < l_lb[sl1] and c_lb[-1] > h_lb[sh2]:
                choch_events.append({
                    'type': 'bullish_choch', 'label': 'Boğa CHoCH',
                    'description': "Downtrend'te swing high kirildi - Olasi trend degisimi",
                    'level': sf(h_lb[sh2]),
                })
            # Bearish CHoCH: Uptrend (HH + HL) iken swing low kırılması
            elif h_lb[sh2] > h_lb[sh1] and l_lb[sl2] > l_lb[sl1] and c_lb[-1] < l_lb[sl2]:
                choch_events.append({
                    'type': 'bearish_choch', 'label': 'Ayı CHoCH',
                    'description': 'Uptrend\'de swing low kırıldı → Olası trend değişimi',
                    'level': sf(l_lb[sl2]),
                })

        # ---- Giriş Bölgeleri ----
        cur = float(c_lb[-1])
        entry_zones = []
        for ob in recent_obs:
            if ob['type'] == 'bullish_ob' and float(ob['bottom']) < cur:
                entry_zones.append({'source': 'bullish_ob', 'level': ob['midpoint'],
                                    'top': ob['top'], 'bottom': ob['bottom']})
            elif ob['type'] == 'bearish_ob' and float(ob['top']) > cur:
                entry_zones.append({'source': 'bearish_ob', 'level': ob['midpoint'],
                                    'top': ob['top'], 'bottom': ob['bottom']})
        for fg in active_fvgs:
            if fg['type'] == 'bullish_fvg' and float(fg['bottom']) < cur:
                entry_zones.append({'source': 'fvg_support', 'level': fg['midpoint'],
                                    'top': fg['top'], 'bottom': fg['bottom']})
            elif fg['type'] == 'bearish_fvg' and float(fg['top']) > cur:
                entry_zones.append({'source': 'fvg_resistance', 'level': fg['midpoint'],
                                    'top': fg['top'], 'bottom': fg['bottom']})

        bull_score = (sum(1 for f in active_fvgs if f['type'] == 'bullish_fvg') +
                      sum(1 for ob in recent_obs if ob['type'] == 'bullish_ob') +
                      sum(1 for b in bos_events if b['type'] == 'bullish_bos') +
                      sum(1 for cc in choch_events if cc['type'] == 'bullish_choch'))
        bear_score = (sum(1 for f in active_fvgs if f['type'] == 'bearish_fvg') +
                      sum(1 for ob in recent_obs if ob['type'] == 'bearish_ob') +
                      sum(1 for b in bos_events if b['type'] == 'bearish_bos') +
                      sum(1 for cc in choch_events if cc['type'] == 'bearish_choch'))
        smc_signal = ('buy' if bull_score > bear_score
                      else ('sell' if bear_score > bull_score else 'neutral'))

        return {
            'signal':         smc_signal,
            'structureTrend': structure_trend,
            'bullScore':      bull_score,
            'bearScore':      bear_score,
            'fvgs':           active_fvgs,
            'orderBlocks':    recent_obs,
            'bosEvents':      bos_events,
            'chochEvents':    choch_events,
            'entryZones':     entry_zones[:4],
            'summary': {
                'activeFvgCount': len(active_fvgs),
                'activeObCount':  len(recent_obs),
                'hasBOS':         len(bos_events) > 0,
                'hasCHoCH':       len(choch_events) > 0,
            },
        }
    except Exception as e:
        print(f"  [SMC] Hata: {e}")
        return {'signal': 'neutral', 'error': str(e),
                'fvgs': [], 'orderBlocks': [], 'bosEvents': [], 'chochEvents': [],
                'summary': {'activeFvgCount': 0, 'activeObCount': 0,
                            'hasBOS': False, 'hasCHoCH': False}}


# =====================================================================
# FAZ 6: KLASİK GRAFİK FORMASYON TESPİTİ
# Double Top/Bottom, H&S, Triangle, Flag/Pennant
# =====================================================================
def calc_chart_patterns(hist, lookback=120):
    """
    Klasik grafik formasyonları:
      - Çift Tepe / Çift Dip
      - Omuz-Baş-Omuz / Ters OBO
      - Yükselen / Alçalan / Simetrik Üçgen
      - Boğa / Ayı Bayrağı
    """
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)  if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float)   if 'Low'  in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        n = len(c)
        lb = min(lookback, n)
        c_lb, h_lb, l_lb = c[-lb:], h[-lb:], l[-lb:]

        patterns = []
        tol = 0.03  # %3 tolerans
        peaks   = _find_peaks(h_lb, 5)
        troughs = _find_troughs(l_lb, 5)

        # ---- Çift Tepe ----
        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if abs(h_lb[p2] - h_lb[p1]) / h_lb[p1] <= tol and p2 - p1 >= 10:
                neckline = float(np.min(l_lb[p1:p2 + 1]))
                completed = bool(c_lb[-1] < neckline)
                height = float(h_lb[p2]) - neckline
                patterns.append({
                    'type': 'double_top', 'label': 'Çift Tepe', 'signal': 'sell',
                    'reliability': 'high', 'completed': completed,
                    'description': (f'İki benzer zirve ({sf(h_lb[p1])}, {sf(h_lb[p2])}) '
                                    f'Neckline: {sf(neckline)}' +
                                    (' → TAMAMLANDI' if completed else ' → Neckline kırılması bekleniyor')),
                    'peak1': sf(h_lb[p1]), 'peak2': sf(h_lb[p2]),
                    'neckline': sf(neckline),
                    'target': sf(neckline - height),
                    'barsAgo': int(lb - p2),
                })

        # ---- Çift Dip ----
        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            if abs(l_lb[t2] - l_lb[t1]) / l_lb[t1] <= tol and t2 - t1 >= 10:
                neckline = float(np.max(h_lb[t1:t2 + 1]))
                completed = bool(c_lb[-1] > neckline)
                height = neckline - float(l_lb[t2])
                patterns.append({
                    'type': 'double_bottom', 'label': 'Çift Dip', 'signal': 'buy',
                    'reliability': 'high', 'completed': completed,
                    'description': (f'İki benzer dip ({sf(l_lb[t1])}, {sf(l_lb[t2])}) '
                                    f'Neckline: {sf(neckline)}' +
                                    (' → TAMAMLANDI' if completed else ' → Neckline kırılması bekleniyor')),
                    'trough1': sf(l_lb[t1]), 'trough2': sf(l_lb[t2]),
                    'neckline': sf(neckline),
                    'target': sf(neckline + height),
                    'barsAgo': int(lb - t2),
                })

        # ---- Omuz-Baş-Omuz ----
        if len(peaks) >= 3:
            ls, hd, rs = peaks[-3], peaks[-2], peaks[-1]
            lsh, hp, rsh = float(h_lb[ls]), float(h_lb[hd]), float(h_lb[rs])
            if (hp > lsh and hp > rsh and
                    abs(rsh - lsh) / lsh <= 0.05 and rs - ls >= 20):
                neckline = (float(np.min(l_lb[ls:hd + 1])) + float(np.min(l_lb[hd:rs + 1]))) / 2
                completed = bool(c_lb[-1] < neckline)
                patterns.append({
                    'type': 'head_shoulders', 'label': 'Omuz-Baş-Omuz', 'signal': 'sell',
                    'reliability': 'very_high', 'completed': completed,
                    'description': (f'Sol omuz ({sf(lsh)}), Baş ({sf(hp)}), '
                                    f'Sağ omuz ({sf(rsh)}) Neckline: {sf(neckline)}' +
                                    (' → TAMAMLANDI' if completed else '')),
                    'leftShoulder': sf(lsh), 'head': sf(hp), 'rightShoulder': sf(rsh),
                    'neckline': sf(neckline),
                    'target': sf(neckline - (hp - neckline)),
                    'barsAgo': int(lb - rs),
                })

        # ---- Ters Omuz-Baş-Omuz ----
        if len(troughs) >= 3:
            ls, hd, rs = troughs[-3], troughs[-2], troughs[-1]
            lsh, hp, rsh = float(l_lb[ls]), float(l_lb[hd]), float(l_lb[rs])
            if (hp < lsh and hp < rsh and
                    abs(rsh - lsh) / lsh <= 0.05 and rs - ls >= 20):
                neckline = (float(np.max(h_lb[ls:hd + 1])) + float(np.max(h_lb[hd:rs + 1]))) / 2
                completed = bool(c_lb[-1] > neckline)
                patterns.append({
                    'type': 'inv_head_shoulders', 'label': 'Ters Omuz-Baş-Omuz', 'signal': 'buy',
                    'reliability': 'very_high', 'completed': completed,
                    'description': (f'Sol omuz ({sf(lsh)}), Baş ({sf(hp)}), '
                                    f'Sağ omuz ({sf(rsh)}) Neckline: {sf(neckline)}' +
                                    (' → TAMAMLANDI' if completed else '')),
                    'leftShoulder': sf(lsh), 'head': sf(hp), 'rightShoulder': sf(rsh),
                    'neckline': sf(neckline),
                    'target': sf(neckline + (neckline - hp)),
                    'barsAgo': int(lb - rs),
                })

        # ---- Üçgen Formasyonları (son 30 bar) ----
        if lb >= 30:
            x      = np.arange(30, dtype=float)
            h_seg  = h_lb[-30:].astype(float)
            l_seg  = l_lb[-30:].astype(float)
            h_slope = float(np.polyfit(x, h_seg, 1)[0])
            l_slope = float(np.polyfit(x, l_seg, 1)[0])
            h_pct   = h_slope / float(np.mean(h_seg)) * 100
            l_pct   = l_slope / float(np.mean(l_seg)) * 100

            if abs(h_pct) < 0.08 and l_pct > 0.08:         # Yükselen Üçgen
                res = sf(float(np.max(h_seg[-10:])))
                rng = float(np.max(h_seg)) - float(np.min(l_seg))
                patterns.append({
                    'type': 'ascending_triangle', 'label': 'Yükselen Üçgen', 'signal': 'buy',
                    'reliability': 'medium', 'completed': bool(c_lb[-1] > float(np.max(h_seg))),
                    'description': f'Düz direnç ({res}) + yükselen dip → Yukarı kırılım beklenir',
                    'resistance': res, 'target': sf(float(np.max(h_seg)) + rng), 'barsAgo': 0,
                })
            elif abs(l_pct) < 0.08 and h_pct < -0.08:      # Alçalan Üçgen
                sup = sf(float(np.min(l_seg[-10:])))
                rng = float(np.max(h_seg)) - float(np.min(l_seg))
                patterns.append({
                    'type': 'descending_triangle', 'label': 'Alçalan Üçgen', 'signal': 'sell',
                    'reliability': 'medium', 'completed': bool(c_lb[-1] < float(np.min(l_seg))),
                    'description': f'Düz destek ({sup}) + düşen zirve → Aşağı kırılım beklenir',
                    'support': sup, 'target': sf(float(np.min(l_seg)) - rng), 'barsAgo': 0,
                })
            elif h_pct < -0.05 and l_pct > 0.05:           # Simetrik Üçgen
                apex = sf((float(np.max(h_seg[-5:])) + float(np.min(l_seg[-5:]))) / 2)
                patterns.append({
                    'type': 'symmetrical_triangle', 'label': 'Simetrik Üçgen', 'signal': 'neutral',
                    'reliability': 'medium', 'completed': False,
                    'description': f'Daralan fiyat aralığı → Güçlü kırılım bekleniyor (apex: {apex})',
                    'apex': apex, 'barsAgo': 0,
                })

        # ---- Bayrak / Flama ----
        if lb >= 26:
            pre_move_pct = (float(c_lb[-16]) - float(c_lb[-26])) / max(float(c_lb[-26]), 1) * 100
            consol_range = ((float(np.max(h_lb[-15:])) - float(np.min(l_lb[-15:]))) /
                            max(float(c_lb[-15]), 1) * 100)
            if abs(pre_move_pct) > 5 and consol_range < 4:
                is_bull = pre_move_pct > 0
                patterns.append({
                    'type': 'bull_flag' if is_bull else 'bear_flag',
                    'label': 'Boğa Bayrağı' if is_bull else 'Ayı Bayrağı',
                    'signal': 'buy' if is_bull else 'sell',
                    'reliability': 'medium', 'completed': False,
                    'description': (f'{sf(abs(pre_move_pct))}% ön hareket + '
                                    f'{sf(consol_range)}% konsolidasyon → Trend devam bekleniyor'),
                    'priorMovePct': sf(pre_move_pct),
                    'consolidationRangePct': sf(consol_range),
                    'barsAgo': 0,
                })

        completed = [p for p in patterns if p.get('completed', False)]
        bull_patt = [p for p in patterns if p['signal'] == 'buy']
        bear_patt = [p for p in patterns if p['signal'] == 'sell']

        if completed:
            overall = completed[0]['signal']
        elif len(bull_patt) > len(bear_patt):
            overall = 'buy'
        elif len(bear_patt) > len(bull_patt):
            overall = 'sell'
        else:
            overall = 'neutral'

        return {
            'signal':            overall,
            'patterns':          patterns,
            'completedPatterns': completed,
            'pendingPatterns':   [p for p in patterns if not p.get('completed', False)],
            'summary': {
                'total':     len(patterns),
                'bullish':   len(bull_patt),
                'bearish':   len(bear_patt),
                'completed': len(completed),
            },
        }
    except Exception as e:
        print(f"  [PATTERNS] Hata: {e}")
        return {'signal': 'neutral', 'patterns': [], 'completedPatterns': [], 'pendingPatterns': [],
                'summary': {'total': 0, 'bullish': 0, 'bearish': 0, 'completed': 0},
                'error': str(e)}


# =====================================================================
# FAZ 7: FİBONACCİ SEVİYELERİ & PİVOT NOKTALARI
# Fibonacci retracement/extension + Classic/Camarilla/Woodie Pivot Points
# =====================================================================
def calc_fibonacci_adv(hist, lookback=60):
    """
    Fibonacci retracement ve extension seviyeleri.
    Son lookback bardaki en yüksek/düşük noktadan hesaplar.
    Retracement : 0.236, 0.382, 0.5, 0.618, 0.786
    Extension   : 1.272, 1.618, 2.618
    """
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)  if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float)   if 'Low'  in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        n = len(c)
        lb = min(lookback, n)

        seg_h = h[-lb:]
        seg_l = l[-lb:]
        hi_idx = int(np.argmax(seg_h))
        lo_idx = int(np.argmin(seg_l))
        swing_high = float(seg_h[hi_idx])
        swing_low  = float(seg_l[lo_idx])
        diff = swing_high - swing_low
        cur  = float(c[-1])

        # Trend yönü: yüksek mi önce, düşük mü?
        if hi_idx > lo_idx:
            # Önce dip, sonra zirve → uptrend retracement (yukarıdan aşağı seviyeler)
            trend = 'uptrend'
            base, top = swing_low, swing_high
        else:
            # Önce zirve, sonra dip → downtrend retracement (aşağıdan yukarı seviyeler)
            trend = 'downtrend'
            base, top = swing_low, swing_high

        ret_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        ext_ratios = [1.272, 1.618, 2.618]

        def label_level(lvl):
            """Mevcut fiyata göre destek/direnç etiketi"""
            if lvl < cur - diff * 0.01:
                return 'support'
            elif lvl > cur + diff * 0.01:
                return 'resistance'
            return 'current'

        retracements = []
        for r in ret_ratios:
            lvl = sf(top - diff * r)
            retracements.append({
                'ratio': r, 'label': f'Fib {r:.3f}',
                'level': lvl,
                'role': label_level(float(lvl)),
                'distPct': sf((float(lvl) - cur) / cur * 100),
            })

        extensions = []
        for r in ext_ratios:
            lvl = sf(top + diff * (r - 1)) if trend == 'uptrend' else sf(base - diff * (r - 1))
            extensions.append({
                'ratio': r, 'label': f'Fib {r:.3f}',
                'level': lvl,
                'role': 'extension_target',
                'distPct': sf((float(lvl) - cur) / cur * 100),
            })

        # En yakın destek ve dirençler
        supports    = sorted([lv for lv in retracements if lv['role'] == 'support'],
                             key=lambda x: -float(x['level']))[:3]
        resistances = sorted([lv for lv in retracements if lv['role'] == 'resistance'],
                             key=lambda x: float(x['level']))[:3]

        # Golden Pocket (0.618-0.65 bölgesi)
        golden_top = sf(top - diff * 0.618)
        golden_bot = sf(top - diff * 0.65)
        in_golden  = float(golden_bot) <= cur <= float(golden_top)

        return {
            'trend':          trend,
            'swingHigh':      sf(swing_high),
            'swingLow':       sf(swing_low),
            'currentPrice':   sf(cur),
            'retracements':   retracements,
            'extensions':     extensions,
            'nearestSupports':    supports,
            'nearestResistances': resistances,
            'goldenPocket': {
                'top': golden_top, 'bottom': golden_bot, 'inZone': in_golden,
            },
        }
    except Exception as e:
        print(f"  [FIB] Hata: {e}")
        return {'error': str(e), 'retracements': [], 'extensions': []}


def calc_pivot_points_adv(hist):
    """
    Klasik, Camarilla ve Woodie Pivot Noktaları.
    Son kapanan günün OHLC verisinden hesaplar.
    Classic   : PP = (H+L+C)/3
    Camarilla : PP = (H+L+C)/3, farklı katsayılar
    Woodie    : PP = (H+L+2C)/4
    """
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)  if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float)   if 'Low'  in hist.columns else c.copy()
        o = hist['Open'].values.astype(float)  if 'Open' in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        o = np.where(np.isnan(o), c, o)

        # Son kapanan günün OHLC değerleri
        H, L, C, O = float(h[-2]), float(l[-2]), float(c[-2]), float(o[-2])
        rng = H - L
        cur = float(c[-1])

        def _role(lvl):
            return 'support' if lvl < cur else ('resistance' if lvl > cur else 'pivot')

        # ---- Classic ----
        pp_c = (H + L + C) / 3
        classic = {
            'pp':  sf(pp_c),
            'r1':  sf(2 * pp_c - L),   'r2': sf(pp_c + rng),
            'r3':  sf(H + 2 * (pp_c - L)),
            's1':  sf(2 * pp_c - H),   's2': sf(pp_c - rng),
            's3':  sf(L - 2 * (H - pp_c)),
        }

        # ---- Camarilla ----
        cam = {
            'pp':  sf(pp_c),
            'r1':  sf(C + rng * 1.1 / 12), 'r2': sf(C + rng * 1.1 / 6),
            'r3':  sf(C + rng * 1.1 / 4),  'r4': sf(C + rng * 1.1 / 2),
            's1':  sf(C - rng * 1.1 / 12), 's2': sf(C - rng * 1.1 / 6),
            's3':  sf(C - rng * 1.1 / 4),  's4': sf(C - rng * 1.1 / 2),
        }

        # ---- Woodie ----
        pp_w = (H + L + 2 * C) / 4
        woodie = {
            'pp':  sf(pp_w),
            'r1':  sf(2 * pp_w - L),        'r2': sf(pp_w + rng),
            's1':  sf(2 * pp_w - H),        's2': sf(pp_w - rng),
        }

        # En yakın pivot seviyeleri (tüm modeller birlikte)
        all_levels = []
        for name, val in classic.items():
            all_levels.append({'model': 'classic', 'name': name.upper(),
                                'level': val, 'role': _role(float(val))})

        supports    = sorted([lv for lv in all_levels if lv['role'] == 'support'],
                             key=lambda x: -float(x['level']))[:3]
        resistances = sorted([lv for lv in all_levels if lv['role'] == 'resistance'],
                             key=lambda x: float(x['level']))[:3]

        # Fiyat pivot'un üstünde mi altında mı?
        bias = 'bullish' if cur > float(classic['pp']) else 'bearish'

        return {
            'currentPrice':       sf(cur),
            'bias':               bias,
            'classic':            classic,
            'camarilla':          cam,
            'woodie':             woodie,
            'nearestSupports':    supports,
            'nearestResistances': resistances,
        }
    except Exception as e:
        print(f"  [PIVOT] Hata: {e}")
        return {'error': str(e), 'classic': {}, 'camarilla': {}, 'woodie': {}}


# =====================================================================
# FAZ 9: İLERİ TEKNİK İNDİKATÖRLER
# Ichimoku Cloud, Stochastic Oscillator, Williams %R
# =====================================================================
def calc_advanced_indicators(hist):
    """
    İleri teknik indikatörler:
      Ichimoku : Tenkan, Kijun, Senkou A/B, Chikou — bulut içi mi?
      Stochastic: %K ve %D (14,3,3) → aşırı alım/satım
      Williams %R: -80 altı aşırı satım, -20 üstü aşırı alım
    """
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)  if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float)   if 'Low'  in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        n = len(c)

        result = {}

        # ---- Ichimoku Cloud ----
        if n >= 52:
            def mid(arr, period):
                return (pd.Series(arr).rolling(period).max() +
                        pd.Series(arr).rolling(period).min()) / 2

            tenkan  = mid(h, 9).values
            kijun   = mid(h, 26).values
            senkou_a = ((pd.Series(tenkan) + pd.Series(kijun)) / 2).shift(26).values
            senkou_b = mid(h, 52).shift(26).values
            chikou  = np.roll(c, -26)

            cur_price = float(c[-1])
            sa = float(senkou_a[-27]) if not np.isnan(senkou_a[-27]) else 0
            sb = float(senkou_b[-27]) if not np.isnan(senkou_b[-27]) else 0
            cloud_top = max(sa, sb)
            cloud_bot = min(sa, sb)

            tk = float(tenkan[-1]) if not np.isnan(tenkan[-1]) else cur_price
            kj = float(kijun[-1])  if not np.isnan(kijun[-1])  else cur_price

            above_cloud = cur_price > cloud_top
            below_cloud = cur_price < cloud_bot
            in_cloud    = cloud_bot <= cur_price <= cloud_top
            tk_kj_cross = ('bullish' if tk > kj else ('bearish' if tk < kj else 'neutral'))

            ich_signal = ('buy'  if above_cloud and tk > kj
                          else ('sell' if below_cloud and tk < kj
                                else 'neutral'))

            result['ichimoku'] = {
                'tenkan':      sf(tk),
                'kijun':       sf(kj),
                'senkouA':     sf(sa),
                'senkouB':     sf(sb),
                'cloudTop':    sf(cloud_top),
                'cloudBottom': sf(cloud_bot),
                'aboveCloud':  above_cloud,
                'belowCloud':  below_cloud,
                'inCloud':     in_cloud,
                'tkKjCross':   tk_kj_cross,
                'signal':      ich_signal,
            }
        else:
            result['ichimoku'] = {'signal': 'neutral', 'error': 'Yetersiz veri (min 52 bar)'}

        # ---- Stochastic (14, 3, 3) ----
        if n >= 17:
            period = 14
            h_ser = pd.Series(h)
            l_ser = pd.Series(l)
            c_ser = pd.Series(c)

            highest_h = h_ser.rolling(period).max()
            lowest_l  = l_ser.rolling(period).min()
            raw_k     = 100 * (c_ser - lowest_l) / (highest_h - lowest_l + 1e-10)
            k_line    = raw_k.rolling(3).mean()
            d_line    = k_line.rolling(3).mean()

            k_val = sf(float(k_line.iloc[-1]))
            d_val = sf(float(d_line.iloc[-1]))

            if float(k_val) < 20 and float(d_val) < 20:
                sto_signal = 'buy'
            elif float(k_val) > 80 and float(d_val) > 80:
                sto_signal = 'sell'
            elif float(k_val) > float(d_val) and float(k_val) < 50:
                sto_signal = 'buy'   # Yukarı kesişim düşük bölgede
            elif float(k_val) < float(d_val) and float(k_val) > 50:
                sto_signal = 'sell'  # Aşağı kesişim yüksek bölgede
            else:
                sto_signal = 'neutral'

            result['stochastic'] = {
                'k': k_val, 'd': d_val,
                'overbought': float(k_val) > 80,
                'oversold':   float(k_val) < 20,
                'signal':     sto_signal,
            }
        else:
            result['stochastic'] = {'signal': 'neutral', 'k': 50, 'd': 50}

        # ---- Williams %R (14) ----
        if n >= 14:
            period = 14
            highest_h = float(np.max(h[-period:]))
            lowest_l  = float(np.min(l[-period:]))
            wr = ((highest_h - float(c[-1])) / (highest_h - lowest_l + 1e-10)) * -100
            wr = sf(wr)

            if float(wr) < -80:
                wr_signal = 'buy'    # Aşırı satım
            elif float(wr) > -20:
                wr_signal = 'sell'   # Aşırı alım
            else:
                wr_signal = 'neutral'

            result['williamsR'] = {
                'value':      wr,
                'overbought': float(wr) > -20,
                'oversold':   float(wr) < -80,
                'signal':     wr_signal,
            }
        else:
            result['williamsR'] = {'signal': 'neutral', 'value': -50}

        # ---- Genel Özet ----
        signals = [result.get('ichimoku', {}).get('signal', 'neutral'),
                   result.get('stochastic', {}).get('signal', 'neutral'),
                   result.get('williamsR', {}).get('signal', 'neutral')]
        buy_cnt  = signals.count('buy')
        sell_cnt = signals.count('sell')
        result['summary'] = {
            'signal':   'buy' if buy_cnt > sell_cnt else ('sell' if sell_cnt > buy_cnt else 'neutral'),
            'buyCount': buy_cnt, 'sellCount': sell_cnt,
        }

        return result
    except Exception as e:
        print(f"  [ADV-IND] Hata: {e}")
        return {'error': str(e), 'summary': {'signal': 'neutral', 'buyCount': 0, 'sellCount': 0}}


# =====================================================================
# FEATURE 1: SIGNAL BACKTESTING & PERFORMANCE TRACKING

def calc_dynamic_thresholds(closes, highs, lows, volumes):
    """Her hisse icin tarihsel dagılıma gore adaptif RSI/BB/Volume esikleri"""
    try:
        n = len(closes)
        if n < 60:
            return {'rsi_oversold': 30, 'rsi_overbought': 70, 'vol_spike': 2.0, 'bb_std': 2.0}

        # Tum RSI degerlerini hesapla
        rsi_values = []
        for i in range(20, n):
            rv = calc_rsi_single(closes[:i+1])
            if rv is not None:
                rsi_values.append(rv)

        if len(rsi_values) < 20:
            return {'rsi_oversold': 30, 'rsi_overbought': 70, 'vol_spike': 2.0, 'bb_std': 2.0}

        rsi_arr = np.array(rsi_values)
        # Percentile bazli esikler: %10 ve %90
        rsi_oversold = sf(np.percentile(rsi_arr, 10))
        rsi_overbought = sf(np.percentile(rsi_arr, 90))

        # Minimum/maximum sinirlari
        rsi_oversold = max(20, min(40, float(rsi_oversold)))
        rsi_overbought = max(60, min(85, float(rsi_overbought)))

        # Volatilite bazli Bollinger genisligi
        if n >= 60:
            daily_returns = np.diff(closes[-60:]) / closes[-60:-1]
            vol = float(np.std(daily_returns))
            # Dusuk volatilite -> daha dar bantlar, yuksek -> daha genis
            bb_std = max(1.5, min(3.0, 2.0 * (vol / 0.02)))
        else:
            bb_std = 2.0

        # Hacim spike esigi: medyan bazli
        if n >= 30:
            vol_mean = float(np.mean(volumes[-30:]))
            vol_std = float(np.std(volumes[-30:]))
            vol_spike = max(1.5, min(3.0, (vol_mean + vol_std) / vol_mean if vol_mean > 0 else 2.0))
        else:
            vol_spike = 2.0

        return {
            'rsi_oversold': sf(rsi_oversold),
            'rsi_overbought': sf(rsi_overbought),
            'vol_spike': sf(vol_spike),
            'bb_std': sf(bb_std),
        }
    except Exception:
        return {'rsi_oversold': 30, 'rsi_overbought': 70, 'vol_spike': 2.0, 'bb_std': 2.0}


# =====================================================================
# FEATURE 3: CANDLESTICK PATTERN RECOGNITION
# =====================================================================
def calc_candlestick_patterns(opens, highs, lows, closes):
    """Mum formasyonlarini tespit et"""
    try:
        n = len(closes)
        if n < 5:
            return {'patterns': [], 'signal': 'neutral'}

        patterns = []
        o, h, l, c = [float(x) for x in opens[-5:]], [float(x) for x in highs[-5:]], [float(x) for x in lows[-5:]], [float(x) for x in closes[-5:]]
        body = [c[i] - o[i] for i in range(5)]
        body_abs = [abs(b) for b in body]
        upper_shadow = [h[i] - max(o[i], c[i]) for i in range(5)]
        lower_shadow = [min(o[i], c[i]) - l[i] for i in range(5)]
        total_range = [h[i] - l[i] for i in range(5)]

        # Son mum (index 4)
        i = 4
        avg_body = np.mean(body_abs[:4]) if np.mean(body_abs[:4]) > 0 else 0.01

        # 1. Doji - Gövde çok küçük
        if body_abs[i] < total_range[i] * 0.1 and total_range[i] > 0:
            patterns.append({'name': 'Doji', 'type': 'neutral', 'description': 'Kararsizlik formasyonu. Trend donusu habercisi olabilir.', 'strength': 2})

        # 2. Hammer (Cekic) - Uzun alt golge, kucuk govde, ust kisminda
        if (lower_shadow[i] > body_abs[i] * 2 and upper_shadow[i] < body_abs[i] * 0.5 and
            body[i-1] < 0):  # Onceki mum dusus
            patterns.append({'name': 'Cekic (Hammer)', 'type': 'bullish', 'description': 'Dusus sonrasi toparlanma sinyali. Guclu alis formasyonu.', 'strength': 3})

        # 3. Shooting Star (Kayan Yildiz) - Uzun ust golge, kucuk govde, alt kisminda
        if (upper_shadow[i] > body_abs[i] * 2 and lower_shadow[i] < body_abs[i] * 0.5 and
            body[i-1] > 0):  # Onceki mum yukselis
            patterns.append({'name': 'Kayan Yildiz (Shooting Star)', 'type': 'bearish', 'description': 'Yukselis sonrasi satis baskisi. Dusus sinyali.', 'strength': 3})

        # 4. Bullish Engulfing (Yukari Yutan)
        if (body[i] > 0 and body[i-1] < 0 and
            o[i] <= c[i-1] and c[i] >= o[i-1] and
            body_abs[i] > body_abs[i-1]):
            patterns.append({'name': 'Yukari Yutan (Bullish Engulfing)', 'type': 'bullish', 'description': 'Guclu alis formasyonu. Alicilar kontrolu ele aldi.', 'strength': 4})

        # 5. Bearish Engulfing (Asagi Yutan)
        if (body[i] < 0 and body[i-1] > 0 and
            o[i] >= c[i-1] and c[i] <= o[i-1] and
            body_abs[i] > body_abs[i-1]):
            patterns.append({'name': 'Asagi Yutan (Bearish Engulfing)', 'type': 'bearish', 'description': 'Guclu satis formasyonu. Saticilar kontrolu ele aldi.', 'strength': 4})

        # 6. Morning Star (Sabah Yildizi) - 3 mumlu boğa formasyonu
        if (n >= 3 and body[i-2] < 0 and body_abs[i-2] > avg_body and
            body_abs[i-1] < avg_body * 0.5 and
            body[i] > 0 and body_abs[i] > avg_body):
            patterns.append({'name': 'Sabah Yildizi (Morning Star)', 'type': 'bullish', 'description': 'Guclu 3 mumlu dip formasyonu. Trendin donusu bekleniyor.', 'strength': 5})

        # 7. Evening Star (Aksam Yildizi) - 3 mumlu ayı formasyonu
        if (n >= 3 and body[i-2] > 0 and body_abs[i-2] > avg_body and
            body_abs[i-1] < avg_body * 0.5 and
            body[i] < 0 and body_abs[i] > avg_body):
            patterns.append({'name': 'Aksam Yildizi (Evening Star)', 'type': 'bearish', 'description': 'Guclu 3 mumlu tepe formasyonu. Dusus bekleniyor.', 'strength': 5})

        # 8. Three White Soldiers (Uc Beyaz Asker)
        if (body[i] > 0 and body[i-1] > 0 and body[i-2] > 0 and
            c[i] > c[i-1] > c[i-2] and
            body_abs[i] > avg_body * 0.5 and body_abs[i-1] > avg_body * 0.5):
            patterns.append({'name': 'Uc Beyaz Asker', 'type': 'bullish', 'description': 'Art arda 3 guclu yukselis mumu. Guclu alis trendi.', 'strength': 4})

        # 9. Three Black Crows (Uc Kara Karga)
        if (body[i] < 0 and body[i-1] < 0 and body[i-2] < 0 and
            c[i] < c[i-1] < c[i-2] and
            body_abs[i] > avg_body * 0.5 and body_abs[i-1] > avg_body * 0.5):
            patterns.append({'name': 'Uc Kara Karga', 'type': 'bearish', 'description': 'Art arda 3 guclu dusus mumu. Guclu satis baskisi.', 'strength': 4})

        # 10. Marubozu (gölgesiz güçlü mum)
        if total_range[i] > 0:
            shadow_ratio = (upper_shadow[i] + lower_shadow[i]) / total_range[i]
            if shadow_ratio < 0.1 and body_abs[i] > avg_body * 1.5:
                mtype = 'bullish' if body[i] > 0 else 'bearish'
                patterns.append({'name': f'Marubozu ({"Yukari" if mtype == "bullish" else "Asagi"})', 'type': mtype,
                    'description': 'Golgesiz guclu mum. Tek yonlu baski. Trend devami beklenir.', 'strength': 3})

        # Genel sinyal
        bullish = sum(1 for p in patterns if p['type'] == 'bullish')
        bearish = sum(1 for p in patterns if p['type'] == 'bearish')
        signal = 'buy' if bullish > bearish else ('sell' if bearish > bullish else 'neutral')

        return {
            'patterns': patterns,
            'signal': signal,
            'bullishCount': bullish,
            'bearishCount': bearish,
        }
    except Exception as e:
        print(f"  [CANDLE] Hata: {e}")
        return {'patterns': [], 'signal': 'neutral'}


# Market regime cache (signals.py tarafindan kullanilir)
_market_regime_cache = {'regime': None, 'ts': 0}

def prepare_chart_data(hist):
    try:
        cs=[{'date':d.strftime('%Y-%m-%d'),'open':sf(r['Open']),'high':sf(r['High']),'low':sf(r['Low']),'close':sf(r['Close']),'volume':si(r['Volume'])} for d,r in hist.iterrows()]
        return {'candlestick':cs,'dates':[c['date'] for c in cs],'prices':[c['close'] for c in cs],'volumes':[c['volume'] for c in cs],'dataPoints':len(cs)}
    except Exception: return {'candlestick':[],'dates':[],'prices':[],'volumes':[],'dataPoints':0}


