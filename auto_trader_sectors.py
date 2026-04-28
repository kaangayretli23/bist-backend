"""
BIST Pro - BIST100 Sektor Mapping
Ayni sektorden cok pozisyon acmamak icin (korelasyon riski).
Mapping eksiltilmistir; bilinmeyen sembol 'DIGER' doner — cap uygulanmaz.
"""

# Ana sektor mapping — BIST100 ve yaygin BIST hisseleri
_SECTOR_MAP = {
    # BANKACILIK
    'AKBNK': 'BANKA', 'GARAN': 'BANKA', 'YKBNK': 'BANKA', 'ISCTR': 'BANKA',
    'HALKB': 'BANKA', 'VAKBN': 'BANKA', 'TSKB': 'BANKA', 'ALBRK': 'BANKA',
    'QNBFB': 'BANKA', 'ICBCT': 'BANKA',
    # HOLDING
    'KCHOL': 'HOLDING', 'SAHOL': 'HOLDING', 'GLYHO': 'HOLDING', 'DOHOL': 'HOLDING',
    'ALARK': 'HOLDING', 'AGHOL': 'HOLDING', 'NTHOL': 'HOLDING', 'IHLGM': 'HOLDING',
    'TKFEN': 'HOLDING', 'GUBRF': 'HOLDING',
    # DEMIR-CELIK
    'EREGL': 'DEMIR_CELIK', 'KRDMD': 'DEMIR_CELIK', 'ISDMR': 'DEMIR_CELIK',
    'BURCE': 'DEMIR_CELIK', 'KRDMA': 'DEMIR_CELIK', 'KRDMB': 'DEMIR_CELIK',
    # ENERJI / RAFINERI
    'TUPRS': 'ENERJI', 'ZOREN': 'ENERJI', 'AKSEN': 'ENERJI', 'AKENR': 'ENERJI',
    'AKFGY': 'ENERJI', 'ODAS': 'ENERJI', 'GESAN': 'ENERJI', 'AKFYE': 'ENERJI',
    'AYGAZ': 'ENERJI', 'ENKAI': 'ENERJI', 'ENJSA': 'ENERJI', 'AKSA': 'ENERJI',
    # GIDA / ICECEK
    'BIMAS': 'GIDA_ICECEK', 'MGROS': 'GIDA_ICECEK', 'SOKM': 'GIDA_ICECEK',
    'ULKER': 'GIDA_ICECEK', 'CCOLA': 'GIDA_ICECEK', 'AEFES': 'GIDA_ICECEK',
    'TUKAS': 'GIDA_ICECEK', 'TATGD': 'GIDA_ICECEK', 'KRVGD': 'GIDA_ICECEK',
    # GAYRIMENKUL (REIT)
    'EKGYO': 'GAYRIMENKUL', 'KZBGY': 'GAYRIMENKUL', 'ALGYO': 'GAYRIMENKUL',
    'TRGYO': 'GAYRIMENKUL', 'OZKGY': 'GAYRIMENKUL', 'AKSGY': 'GAYRIMENKUL',
    # OTOMOTIV
    'TOASO': 'OTOMOTIV', 'FROTO': 'OTOMOTIV', 'OTKAR': 'OTOMOTIV',
    'DOAS': 'OTOMOTIV', 'KARSN': 'OTOMOTIV', 'BRSAN': 'OTOMOTIV',
    # HAVACILIK
    'THYAO': 'HAVACILIK', 'PGSUS': 'HAVACILIK', 'TAVHL': 'HAVACILIK',
    # TELEKOM
    'TCELL': 'TELEKOM', 'TTKOM': 'TELEKOM',
    # PERAKENDE / GIYIM
    'MAVI': 'PERAKENDE', 'BIZIM': 'PERAKENDE', 'YATAS': 'PERAKENDE',
    # SAVUNMA / TEKNOLOJI
    'ASELS': 'SAVUNMA', 'KAREL': 'SAVUNMA', 'ALTNY': 'SAVUNMA',
    'LOGO': 'TEKNOLOJI', 'NETAS': 'TEKNOLOJI', 'INDES': 'TEKNOLOJI',
    'ARENA': 'TEKNOLOJI', 'ALCTL': 'TEKNOLOJI', 'KFEIN': 'TEKNOLOJI',
    # TEKSTIL
    'KORDS': 'TEKSTIL', 'ARSAN': 'TEKSTIL',
    # INSAAT / CIMENTO
    'SISE': 'INSAAT', 'ANELE': 'INSAAT', 'CIMSA': 'INSAAT',
    'PARSN': 'INSAAT', 'AKCNS': 'INSAAT',
    # KIMYA
    'HEKTS': 'KIMYA', 'BAGFS': 'KIMYA', 'PETKM': 'KIMYA',
    # SAGLIK / ILAC
    'DEVA': 'SAGLIK', 'MPARK': 'SAGLIK', 'SELEC': 'SAGLIK',
    'LKMNH': 'SAGLIK', 'ECILC': 'SAGLIK',
    # MEDYA
    'HURGZ': 'MEDYA', 'DOGUB': 'MEDYA',
    # MADENCILIK / MUCEVHER
    'IPEKE': 'MADENCILIK', 'KOZAL': 'MADENCILIK', 'KARTN': 'MADENCILIK',
}


def get_sector(symbol: str) -> str:
    """Sembol → sektor adi. Bilinmeyen 'DIGER'."""
    return _SECTOR_MAP.get((symbol or '').upper(), 'DIGER')


def count_open_per_sector(open_positions: list) -> dict:
    """Acik pozisyonlardan sektor sayilari. {'BANKA': 2, 'HOLDING': 1}"""
    out: dict = {}
    for p in open_positions or []:
        sec = get_sector(p.get('symbol', ''))
        if sec == 'DIGER':
            continue  # Bilinmeyen sembollerde cap uygulanmaz
        out[sec] = out.get(sec, 0) + 1
    return out


def sector_full(sym: str, open_positions: list, max_per_sector: int) -> tuple:
    """sym icin sektor cap'i dolu mu? (full_bool, sector_str, current_count)"""
    if max_per_sector <= 0:
        return False, '', 0
    sec = get_sector(sym)
    if sec == 'DIGER':
        return False, sec, 0  # Bilinmeyen sembol — cap uygulamayalim
    counts = count_open_per_sector(open_positions)
    cur = counts.get(sec, 0)
    return cur >= max_per_sector, sec, cur
