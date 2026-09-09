"""
Smart Money / Cash Flow Scanner — Streamlit Web App (Mobile Friendly)
يجلب كل أسهم NASDAQ المدرجة تلقائياً ثم يُبرز أعلى الفرص حسب Score
"""

import time
import re
import urllib.request
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ═══════════════════════════════════════════════
#  DEFAULT PARAMETERS
# ═══════════════════════════════════════════════
MFI_PERIOD         = 14
CMF_PERIOD         = 20
OBV_SLOPE_LEN      = 10
VWAP_PERIOD        = 20
LIQUIDITY_BINS     = 50
LIQUIDITY_LOOKBACK = 100
ATR_PERIOD         = 14

CHUNK_SIZE = 80
CACHE_TTL  = 300

# ═══════════════════════════════════════════════
#  جلب رموز NASDAQ كاملة من الموقع الرسمي
# ═══════════════════════════════════════════════

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

# قائمة احتياطية إذا فشل التحميل من الإنترنت
FALLBACK_NASDAQ = sorted(set([
    # NASDAQ-100
    "AAPL","ABNB","ADSK","ADP","ADBE","AEP","AMAT","AMGN","AMZN","ANSS",
    "APP","ARM","ASML","ASR","AVGO","AZN","BIIB","BKNG","BKR","BIDU",
    "CDNS","CDW","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CTAS","CTSH",
    "CSCO","DASH","DDOG","DLTR","DXCM","EA","EXC","FANG","FAST","FTNT",
    "GEHC","GILD","GOOG","GOOGL","HON","IDXX","ILMN","INTC","INTU","ISRG",
    "KDP","KHC","KLAC","KVUE","LRCX","LULU","MAR","MCHP","MDLZ","MELI",
    "META","MNST","MOH","MRVL","MSFT","MU","NFLX","NVDA","NXPI","ODFL",
    "ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PLTR","PYCR","QCOM",
    "REGN","ROP","ROST","SBUX","SMCI","SNPS","SWKS","TEAM","TMUS","TSLA",
    "TTWO","TXN","VRSK","VRTX","VTRS","WBD","WDAY","XEL","ZS",
    # أسهم إضافية شائعة
    "AAL","AAC","AADI","AAOI","AAT","AAWW","ABUS","ACAD","ACBI","ACGL",
    "ACHR","ACIW","ACLS","ACMR","ACNB","ACRS","ACTG","ADAG","ADIL","ADMA",
    "ADN","ADNT","ADOR","ADTX","ADVS","AEHR","AEL","AEMD","AEYE","AFRM",
    "AGEN","AGFS","AGIO","AGLE","AGTC","AHCO","AHH","AI","AIA","AIE",
    "AIRS","AKAM","AKBA","AKRX","ALCO","ALGM","ALGS","ALHC","ALIM","ALKT",
    "ALLK","ALPN","ALRM","ALT","ALTR","ALVR","AMAL","AMBA","AMCX","AMD",
    "AMKR","AMNB","AMPX","AMST","AMSW","AMWL","ANDE","ANDA","ANGI","ANIP",
    "ANTE","AOBC","AOSL","APLS","APO","APPF","APPH","APTO","APTX","APWC",
    "ARAY","ARBE","ARCC","ARCT","ARDX","ARES","ARGX","ARKG","ARKK","ARKW",
    "ARLP","ARMP","AROC","ARQT","ARR","ARTL","ARVN","ARWR","ASAN","ASND",
    "ASPN","ASRV","ASTC","ASUR","ATCX","ATEN","ATEX","ATH","ATHA","ATLC",
    "ATNI","ATOM","ATOS","ATRA","ATSG","ATXI","AUB","AUDA","AUS","AUTL",
    "AVAV","AVD","AVEO","AVGO","AVNW","AVPT","AVTR","AVXL","AWIN","AXAS",
    "AXDX","AXGN","AXLA","AXSM","AYLA","BABA","BAND","BANK","BASI","BATRA",
    "BBIG","BCEL","BCLI","BCML","BCNN","BCOV","BCRX","BDC","BEAM","BECN",
    "BELFA","BEST","BFST","BGNE","BHAT","BHE","BHF","BHVN","BIGC","BIGZ",
    "BIOL","BIOT","BIVI","BJRI","BKFI","BL","BLBD","BLDP","BLKB","BLRX",
    "BLTS","BMBL","BNR","BNTX","BOCN","BOKF","BOLT","BOMN","BPMC","BPOP",
    "BPT","BROG","BRPH","BSQR","BSBK","BSGM","BTAI","BTBT","BTEK","BTGN",
    "BTN","BTRS","BUDZ","BV","BVK","BW","BWMX","BYND","BYSI","CABA",
    "CACC","CADE","CAMP","CANG","CAPL","CAR","CARV","CASI","CASS","CATB",
    "CATC","CATY","CBAN","CBAT","CBBT","CBMG","CBPO","CBRN","CBSH","CBTX",
    "CCAP","CCBG","CCHW","CCCN","CCCR","CCLP","CCNE","CCOI","CCRC","CCRN",
    "CDE","CDNA","CECO","CELH","CEN","CENT","CERE","CEVA","CFBK","CFFI",
    "CFLT","CFMS","CGNT","CHCO","CHEK","CHRS","CHY","CIBR","CIIC","CINC",
    "CINT","CISN","CKPT","CLBK","CLDX","CLDR","CLFD","CLMT","CLNE","CLPR",
    "CLSK","CLW","CMPO","CMPS","CMRM","CNCE","CNCL","CNCR","CNDT","CNET",
    "CNFN","CNHI","CNOB","CNR","CNSL","CNSP","CNTB","CNTX","COCP","CODX",
    "COGT","COMM","COUP","COWN","CPHC","CPK","CPRX","CPT","CRBP","CRC",
    "CREG","CRIS","CRKR","CRLS","CRMT","CRNC","CRNX","CROX","CRSP","CRVL",
    "CRY","CSPI","CSTR","CSV","CTMX","CTNL","CTOS","CTSH","CURI","CVLY",
    "CVS","CVV","CXDC","CYBE","CYBR","CYN","CYRN","DAKT","DALI","DARE",
    "DBD","DBVT","DCBO","DCTH","DDMX","DECK","DECN","DEPO","DERM","DFPH",
    "DGIN","DGICA","DIOD","DIS","DLTH","DMRC","DNLI","DOCU","DORM","DPST",
    "DRCT","DRD2","DRIO","DTC","DTIL","DWIN","DXTR","EHT","EIGR","EL",
    "ELAN","ELLO","ENLV","ENOB","ENPH","ENS","EQIX","ERAS","ESEA","ETRN",
    "ETSY","EUCR","EVBG","EVI","EVLO","EVOK","EVTL","EXEL","EXFO","EXPD",
    "EYEN","EYPT","EZPW","FATE","FBIO","FCEL","FDBO","FEIM","FGEN","FHCO",
    "FISI","FLDM","FLGC","FLIR","FLNT","FLWS","FMBH","FMNI","FNCH","FNKO",
    "FNLC","FOLD","FORM","FOUR","FPRX","FRBA","FREQ","FRGI","FRME","FROG",
    "FRTX","FSEA","FSLR","FSTR","FTHM","FULC","FUND","FUV","FVRR","FWONK",
    "GATE","GBT","GECC","GEN","GERN","GH","GHSI","GIG","GIII","GLBS",
    "GLG","GLNG","GLSI","GLOB","GLOP","GMIN","GMVD","GNC","GNE","GNMK",
    "GNPX","GNTX","GNUS","GOSS","GPRO","GRAF","GRBK","GRFS","GRMN","GSBC",
    "GSKY","GTEC","GTHX","GTIM","GTS","GUG","GWRE","HAIN","HALO","HAPP",
    "HAPPY","HAYW","HBAN","HBB","HCA","HCHC","HDST","HELE","HIMX","HLNE",
    "HLIT","HLMN","HMHC","HMST","HOFV","HOFT","HOLI","HOTH","HOWL","HPK",
    "HPR","HSC","HSTM","HSII","HTBX","HTLD","HTZG","HUBG","HUIZ","HUNTW",
    "HWBK","HWKN","HX","HYFM","HZNP","IAC","IBCP","IBKR","ICCH","ICLK",
    "ICMT","ICUI","ID","IDE","IEA","IESC","IFGL","IFIN","IGIC","IGLD",
    "IKT","IMCR","IMMR","IMMX","IMNP","IMTE","IMTX","INBK","INBX","INOD",
    "INSG","INSM","INSP","INTF","IOSP","IPAR","IPDN","IPGP","IQ","IRBT",
    "IREN","IRMD","IRTC","ITCI","ITI","ITOS","IVAC","IVP","JAKK","JBHT",
    "JAZZ","JBL","JBLU","JCOM","JD","JFIN","JFU","JGW","JKHY","JMIA",
    "JMP","JNJX","JRSH","JUPW","KALA","KAMN","KAPP","KBAL","KDMN","KE",
    "KEM","KEN","KEQU","KEX","KEYS","KFY","KIDS","KLICY","KLTR","KMTS",
    "KNSL","KNTE","KOD","KOPN","KPLT","KRMD","KRNT","KSMT","KSPN","KTCC",
    "KURA","KVHI","KWHS","KYMR","LAES","LAKR","LAMR","LANC","LASR","LAT",
    "LBRCE","LBRCK","LCAP","LCII","LDOS","LECO","LEGH","LEGN","LFT","LGIH",
    "LGND","LIND","LITE","LIVN","LIXT","LLAP","LMB","LMAT","LMD","LULU",
    "LUNG","LUX","LXFR","LXRX","MACQ","MANH","MANT","MARK","MASI","MASI",
    "MATW","MAXR","MBIN","MBII","MBUU","MCFE","MCHP","MD","MDJD","MDRX",
    "MDVL","ME","MELI","MERC","METG","MGIC","MGNI","MGNX","MGT","MGYR",
    "MHLD","MIKR","MIMV","MINM","MIRC","MITK","MKSI","MLAB","MLNK","MLYN",
    "MMSI","MNST","MNTX","MOBG","MODG","MODN","MOMO","MOR","MORN","MOSY",
    "MPAD","MPLN","MPWR","MRCY","MRIN","MRMD","MRSN","MSGE","MSTR","MTCH",
    "MTLS","MTRN","MTRX","MUFG","MUSA","MVIS","MWHS","MXIM","MXL","MYRG",
    "NAAS","NABL","NAJR","NATI","NAVG","NAVX","NBEV","NBIX","NBL","NCBS",
    "NCMI","NCR","NDLS","NDSN","NDTK","NEGG","NEM","NEO","NEOG","NEPH",
    "NERV","NES","NET","NEU","NFE","NFEC","NFLX","NGS","NH","NHC","NICE",
    "NIR","NITE","NJWW","NKLA","NMBL","NMRD","NNVC","NOVN","NOW","NREF",
    "NRGS","NRO","NSIT","NSSR","NSTG","NTAP","NTCT","NTEC","NTGR","NTLA",
    "NTNX","NTRB","NTRS","NTST","NUAN","NUBG","NUVL","NVAX","NVCR","NVE",
    "NVMI","NVO","NVR","NWBI","NWLK","NWLI","NWN","NWPX","NXST","NXTC",
    "OBCI","OBIO","OBJ","OBSV","OCCI","OCDX","OCFT","OCSI","OCTA","ODP",
    "OESX","OFLX","OGFG","OIG","OIS","OLMN","OLTK","OMCL","OMER","OMEX",
    "ONCR","ONDS","ONEW","ONTO","ONTX","OPY","ORB","ORC","ORGO","ORLY",
    "ORMP","ORTX","OSIS","OSMT","OSTK","OTRK","OXYG","PAAS","PACB","PACI",
    "PAE","PAGP","PAHC","PAKM","PALK","PANL","PAR","PBCO","PCBC","PCGX",
    "PCLI","PCOM","PCT","PCTI","PDCO","PDEX","PDVW","PEAK","PED","PEGA",
    "PEIX","PENN","PEP","PERI","PETS","PFBC","PFGC","PGTI","PHAR","PHAT",
    "PHIO","PHMD","PINC","PIRS","PIX","PKOH","PLAB","PLAY","PLMI","PLNT",
    "PLOW","PLRX","PLSE","PLUS","PMVP","PNFP","PNRG","PODD","POL","POWL",
    "POWR","POZN","PPBI","PPLT","PRFT","PRGS","PRIM","PRKR","PRLB","PRPL",
    "PRTA","PRTK","PRTO","PSEC","PSTG","PSTX","PTCT","PTEN","PTGX","PTN",
    "PTON","PTRZ","PTVE","PUCK","PWFL","PWRW","PXMD","PYPD","QADA","QGEN",
    "QGLY","QH","QIDA","QK","QMCO","QRTEA","QRVO","QS","QSI","QSRV",
    "QTNT","QTWO","QUAD","QUCC","QUIK","RADA","RAIL","RAMP","RAPT","RARX",
    "RATE","RBBN","RCEL","RCII","RCKT","RCMT","RDCM","REBN","REDU","REFI",
    "REGI","REKR","RELV","REN","REPL","RESN","RGNX","RH","RHE","RIOT",
    "RMBI","RMBS","RMD","RNDB","RNET","ROCK","ROIC","ROL","ROOT","RPD",
    "RPRX","RRGR","RSMT","RSTN","RSVR","RTLR","RVLV","RVMD","RVNC","RYAAY",
    "SAFM","SAIL","SAIA","SANG","SANM","SASR","SATS","SAVE","SBAC","SBGI",
    "SBLK","SBRA","SBSI","SCLE","SCNX","SCON","SCOR","SCPH","SCVL","SDIG",
    "SEAC","SECO","SEED","SEEL","SEMI","SEN","SFBS","SFM","SFNC","SGAM",
    "SGBX","SGEN","SGH","SGMA","SGMO","SGMS","SGRY","SHEN","SHLS","SHOO",
    "SHSP","SIFY","SILC","SIMG","SITM","SIVB","SJW","SKIL","SKYW","SLAB",
    "SLNO","SMTC","SNDL","SNDR","SNEW","SNEX","SNFCA","SNGX","SNPS","SNR",
    "SNTG","SOFI","SONM","SORT","SOXI","SOXX","SOXL","SOXM","SOXQ","SPKE",
    "SPNE","SPRB","SPRC","SPTN","SPWH","SPWR","SQNS","SRAD","SRCE","SREV",
    "SRPT","SRRR","SRT","SRTS","SSNC","SSPK","SSYS","STBX","STC","STE",
    "STMP","STND","STOK","STRL","STRT","STSK","STTK","STWO","SUMO","SUNW",
    "SUPN","SVRA","SWAV","SWCH","SWET","SWIM","SWIR","SXC","SYBT","SYNH",
    "SYRS","SZLM","TAC","TARO","TATT","TAYD","TBGI","TCEHY","TCMD","TCRR",
    "TDC","TDOC","TECK","TECU","TELA","TEL","TENB","TER","TESS","TGLS",
    "TGTX","THRM","TIGO","TILE","TIMP","TIRX","TITN","TJX","TKNO","TLIS",
    "TLRY","TMDX","TMF","TNKS","TNXP","TOUR","TOWN","TPOX","TRDA","TREC",
    "TRIP","TRMB","TRMR","TRNS","TROW","TRTN","TRUP","TRVI","TRUE","TRUP",
    "TSCO","TSEM","TSLA","TSRI","TTAM","TTD","TTMI","TTRS","TTWO","TUSK",
    "TVAC","TWGI","TWNK","TXG","TXN","TXRH","TYHT","TYL","TZOO","UADA",
    "UAVS","UBFO","UBSI","UCBI","UCTT","UEIC","UFCS","UFI","UGRO","UIHC",
    "UK","UMBF","UNFI","UNH","UNIT","UNTY","UPBD","UPLD","UPST","USAK",
    "USPH","UTMD","UTSL","VCEL","VECO","VERX","VIRC","VITL","VIVE","VKTX",
    "VIRC","VRSN","VSAT","VSEC","VSH","VSTO","VTNR","VTRU","VVI","VVOS",
    "VVPR","VYNE","WALD","WASH","WATT","WAVD","WBA","WBRE","WCMS","WDC",
    "WEN","WERN","WHLR","WIGO","WIMI","WING","WIRE","WIT","WK","WKHS",
    "WKEY","WMC","WMK","WNC","WSTG","WTBA","WTER","WVFC","WW","WWW",
    "XAIR","XBIT","XNET","XNPD","XPER","XSPA","YELL","YORW","YQ","YTRA",
    "YUM","ZEAL","ZEN","ZETA","ZGNX","ZIM","ZION","ZIOP","ZJZZT","ZS",
    "ZTO","ZTS","ZUMZ","ZVRA","ZYME","ZYNE",
]))

def _parse_nasdaq_text(text: str) -> list:
    """يحلّل نص ملف NASDAQ ويرجع قائمة الرموز الصالحة."""
    symbols = []
    for line in text.strip().split("\n"):
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        sym = parts[0].strip()
        # تخطي السطر الأول (رأس الجدول) والسطر الأخير (وقت الإنشاء)
        if not sym or not sym[0].isalpha():
            continue
        if "File Creation Time" in sym:
            continue
        # تخطي أسهم الاختبار
        if parts[3].strip() == "Y":
            continue
        # تخطي الرموز الخاصة ($ = warrants)
        if "$" in sym:
            continue
        # رمز صالح
        if re.fullmatch(r"[A-Z][A-Z0-9.\-]*", sym):
            symbols.append(sym)
    return sorted(set(symbols))


def _fetch_nasdaq_primary() -> list:
    """الطريقة 1: ملف NASDAQ Trader الرسمي."""
    req = urllib.request.Request(
        NASDAQ_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
    symbols = _parse_nasdaq_text(text)
    if len(symbols) < 200:
        raise ValueError(f"فقط {len(symbols)} رمز — الملف قد يكون تالفًا")
    return symbols


def _fetch_nasdaq_secondary() -> list:
    """الطريقة 2: ملف otherlisted كملحق للأسهم المدرجة في بورصات أخرى متداولة على NASDAQ."""
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
    return _parse_nasdaq_text(text)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_all_nasdaq_symbols() -> tuple:
    """
    يجلب كل رموز NASDAQ المدرجة من الموقع الرسمي.
    يرجع tuple: (symbols_list, source_name, error_msg_or_None)
    """
    # ── المحاولة 1: nasdaqlisted.txt ─────────
    try:
        symbols = _fetch_nasdaq_primary()
        source  = f"nasdaqlisted.txt ({len(symbols)} رمز)"
        return tuple(symbols), source, None
    except Exception as e1:
        err1 = str(e1)

    # ── المحاولة 2: ملف إضافي ────────────────
    try:
        symbols = _fetch_nasdaq_secondary()
        if len(symbols) >= 200:
            source = f"otherlisted.txt ({len(symbols)} رمز)"
            return tuple(symbols), source, None
    except Exception:
        pass

    # ── الاحتياطي: قائمة مدمجة ───────────────
    symbols = sorted(set(FALLBACK_NASDAQ))
    source  = f"قائمة احتياطية ({len(symbols)} رمز)"
    return tuple(symbols), source, err1


# ══════════════════════════════════════════════════════════
#  TIMEFRAME CONFIG
# ══════════════════════════════════════════════════════════
TF_CONFIG = {
    "1h" : {"yf_interval": "1h",  "yf_period": "730d",
            "label": "⏰ Hourly (1H)",   "resample": None},
    "4h" : {"yf_interval": "1h",  "yf_period": "730d",
            "label": "🕓 4-Hours (4H)",  "resample": "4h"},
    "1d" : {"yf_interval": "1d",  "yf_period": "2y",
            "label": "📅 Daily (1D)",    "resample": None},
    "1wk": {"yf_interval": "1wk", "yf_period": "5y",
            "label": "📆 Weekly (1W)",   "resample": None},
}

# ═══════════════════════════════════════════════
#  DATA DOWNLOAD — دفعي (batch)
# ═══════════════════════════════════════════════

def download_raw_batch(tickers: tuple, yf_interval: str, yf_period: str) -> dict:
    tickers = list(tickers)
    result: dict = {}
    chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]

    for idx, chunk in enumerate(chunks):
        try:
            raw = yf.download(
                tickers  = chunk,
                period   = yf_period,
                interval = yf_interval,
                group_by = "ticker",
                threads  = True,
                progress = False, auto_adjust = True, actions = False,
            )
        except Exception:
            raw = None

        if raw is None or raw.empty:
            for t in chunk:
                result[t] = None
        else:
            for ticker in chunk:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if ticker not in raw.columns.get_level_values(0):
                            result[ticker] = None
                            continue
                        df = raw[ticker].copy()
                    else:
                        df = raw.copy()
                    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                    result[ticker] = df if len(df) > 0 else None
                except Exception:
                    result[ticker] = None

        if idx < len(chunks) - 1:
            time.sleep(0.5)

    return result

download_raw_batch_cached = st.cache_data(ttl=CACHE_TTL, show_spinner=False)(download_raw_batch)


def apply_resample(df, resample_rule):
    if df is None or not resample_rule:
        return df
    try:
        d = df.copy()
        if d.index.tz is not None:
            d.index = d.index.tz_convert("UTC").tz_localize(None)
        d = (d.resample(resample_rule)
               .agg({"Open": "first", "High": "max",
                     "Low": "min", "Close": "last", "Volume": "sum"})
               .dropna(subset=["Close"]))
        return d if len(d) > 0 else None
    except Exception:
        return None


def get_data_for_tf(tickers: list, tf: str) -> dict:
    cfg = TF_CONFIG.get(tf, TF_CONFIG["1d"])
    key = tuple(sorted(set(tickers)))
    raw_map = download_raw_batch_cached(key, cfg["yf_interval"], cfg["yf_period"])
    return {t: apply_resample(raw_map.get(t), cfg["resample"]) for t in tickers}


# ═══════════════════════════════════════════════
#  SMART MONEY INDICATORS
# ═══════════════════════════════════════════════

def _rolling_sum(arr, period):
    n = len(arr)
    if n < period:
        return np.full(n, np.nan)
    cs  = np.cumsum(arr)
    out = np.full(n, np.nan)
    out[period - 1] = cs[period - 1]
    if period < n:
        out[period:] = cs[period:] - cs[:-period]
    return out


def calc_mfi(high, low, close, volume, period=MFI_PERIOD):
    n = len(close)
    if n < period + 1:
        return None
    tp  = (high + low + close) / 3.0
    mf  = tp * volume
    pos = np.zeros(n)
    neg = np.zeros(n)
    for i in range(1, n):
        if   tp[i] > tp[i - 1]: pos[i] = mf[i]
        elif tp[i] < tp[i - 1]: neg[i] = mf[i]
    rp = _rolling_sum(pos, period)
    rn = _rolling_sum(neg, period)
    out = np.full(n, np.nan)
    for i in range(period, n):
        ns = rn[i]
        out[i] = 100.0 if ns < 1e-10 else 100.0 - 100.0 / (1.0 + rp[i] / ns)
    return out


def calc_cmf(high, low, close, volume, period=CMF_PERIOD):
    n = len(close)
    if n < period:
        return None
    hl  = np.maximum(high - low, 1e-10)
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * volume
    r_mfv = _rolling_sum(mfv, period)
    r_vol = _rolling_sum(volume, period)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        v = r_vol[i]
        out[i] = 0.0 if v < 1e-10 else r_mfv[i] / v
    return out


def calc_obv(close, volume):
    n = len(close)
    if n < 2:
        return None
    d   = np.sign(np.diff(close))
    obv = np.empty(n)
    obv[0] = volume[0]
    for i in range(1, n):
        obv[i] = obv[i - 1] + d[i - 1] * volume[i]
    return obv


def calc_vwap(high, low, close, volume, period=VWAP_PERIOD):
    n = len(close)
    if n < period:
        return None
    tp    = (high + low + close) / 3.0
    tpv   = tp * volume
    r_tpv = _rolling_sum(tpv, period)
    r_vol = _rolling_sum(volume, period)
    out   = np.full(n, np.nan)
    for i in range(period - 1, n):
        v = r_vol[i]
        out[i] = close[i] if v < 1e-10 else r_tpv[i] / v
    return out


def calc_atr(high, low, close, period=ATR_PERIOD):
    n = len(close)
    if n < period + 1:
        return None
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i]  - close[i - 1]))
    rs  = _rolling_sum(tr, period)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = rs[i] / period
    return out


def liquidity_levels(close, volume, n_bins=LIQUIDITY_BINS,
                     lookback=LIQUIDITY_LOOKBACK):
    n   = min(lookback, len(close))
    prc = close[-n:]
    vol = volume[-n:]
    lo, hi = prc.min(), prc.max()
    if hi - lo < 1e-10:
        return None, None
    edges    = np.linspace(lo, hi, n_bins + 1)
    centers  = (edges[:-1] + edges[1:]) / 2.0
    bins_vol = np.zeros(n_bins)
    idx      = np.clip(((prc - lo) / (hi - lo) * n_bins).astype(int), 0, n_bins - 1)
    for j, k in enumerate(idx):
        bins_vol[k] += vol[j]
    top_n   = max(1, n_bins // 5)
    top_idx = np.argsort(bins_vol)[::-1][:top_n]
    liq_prc = np.sort(centers[top_idx])
    cur     = close[-1]
    sup = res = None
    for p in liq_prc:
        if p < cur:
            sup = p
        elif p > cur and res is None:
            res = p
    return sup, res


# ═══════════════════════════════════════════════
#  SIGNAL GENERATOR — Smart Money / Cash Flow
# ═══════════════════════════════════════════════

def get_signal_from_df(ticker, df,
                       mfi_os, mfi_ob, score_min,
                       sl_atr, tp_atr,
                       min_price, min_vol_avg,
                       mfi_period=MFI_PERIOD,
                       cmf_period=CMF_PERIOD,
                       obv_len=OBV_SLOPE_LEN,
                       vwap_period=VWAP_PERIOD):
    try:
        needed = max(mfi_period, cmf_period, obv_len,
                     vwap_period, ATR_PERIOD, LIQUIDITY_LOOKBACK) + 30
        if df is None or len(df) < needed:
            return None

        close = df["Close"].values.astype(float)
        high  = df["High"].values.astype(float)
        low   = df["Low"].values.astype(float)
        vol   = np.maximum(df["Volume"].values.astype(float), 0.0)
        n     = len(close)

        price = close[-1]
        # ── فلتر سريع: سعر وحجم ────────────────
        if price < min_price:
            return None
        avg_vol = vol[-20:].mean() if n >= 20 else vol.mean()
        if avg_vol < min_vol_avg:
            return None

        mfi_a  = calc_mfi(high, low, close, vol, mfi_period)
        cmf_a  = calc_cmf(high, low, close, vol, cmf_period)
        obv_a  = calc_obv(close, vol)
        vwap_a = calc_vwap(high, low, close, vol, vwap_period)
        atr_a  = calc_atr(high, low, close, ATR_PERIOD)
        if any(x is None for x in (mfi_a, cmf_a, obv_a, vwap_a, atr_a)):
            return None

        i  = n - 1
        i1 = n - 2

        for arr in (mfi_a, cmf_a, vwap_a, atr_a):
            if np.isnan(arr[i]):
                return None

        mfi   = mfi_a[i]
        mfi_p = mfi_a[i1] if not np.isnan(mfi_a[i1]) else mfi
        cmf   = cmf_a[i]
        cmf_p = cmf_a[i1] if not np.isnan(cmf_a[i1]) else cmf
        vwap  = vwap_a[i]
        atr   = atr_a[i]

        ol       = min(obv_len, n - 1)
        obv_roc  = (obv_a[i] - obv_a[i - ol]) / (abs(obv_a[i - ol]) + 1.0)
        prc_roc  = (close[i] - close[i - ol]) / (close[i - ol] + 1e-10)

        # 1. MFI Score
        if   mfi < mfi_os and mfi_p >= mfi_os: s_mfi =  1.0
        elif mfi > mfi_ob and mfi_p <= mfi_ob: s_mfi = -1.0
        elif mfi < mfi_os:                     s_mfi =  0.8
        elif mfi > mfi_ob:                     s_mfi = -0.8
        elif mfi < 45:                         s_mfi =  (45 - mfi) / 45 * 0.4
        elif mfi > 55:                         s_mfi = -(mfi - 55) / 45 * 0.4
        else:                                  s_mfi =  0.0
        mfi_mom = np.clip((mfi - mfi_p) / 10.0, -0.2, 0.2)
        s_mfi   = np.clip(s_mfi + mfi_mom * 0.4, -1.0, 1.0)

        # 2. CMF Score
        if   cmf >  0.10: s_cmf =  min(1.0, cmf / 0.25)
        elif cmf >  0.00: s_cmf =  cmf / 0.10 * 0.3
        elif cmf < -0.10: s_cmf =  max(-1.0, cmf / 0.25)
        elif cmf <  0.00: s_cmf =  cmf / 0.10 * 0.3
        else:             s_cmf =  0.0
        cmf_mom = np.clip((cmf - cmf_p) * 10.0, -0.3, 0.3)
        s_cmf   = np.clip(s_cmf + cmf_mom * 0.3, -1.0, 1.0)

        # 3. OBV Score
        if   obv_roc > 0 and prc_roc < -0.005: s_obv =  0.85
        elif obv_roc < 0 and prc_roc >  0.005: s_obv = -0.85
        elif obv_roc > 0 and prc_roc > 0:      s_obv =  0.30
        elif obv_roc < 0 and prc_roc < 0:      s_obv = -0.30
        else:                                   s_obv =  0.0

        # 4. VWAP Score
        cross_up   = (not np.isnan(vwap_a[i1]) and
                      close[i1] <= vwap_a[i1] and price > vwap)
        cross_down = (not np.isnan(vwap_a[i1]) and
                      close[i1] >= vwap_a[i1] and price < vwap)
        if   cross_up:   s_vwap =  1.0
        elif cross_down: s_vwap = -1.0
        elif price > vwap:
            s_vwap = min( 0.5, (price - vwap) / (atr + 1e-10) * 0.2)
        else:
            s_vwap = max(-0.5, -(vwap - price) / (atr + 1e-10) * 0.2)
        s_vwap = np.clip(s_vwap, -1.0, 1.0)

        # 5. Liquidity Score
        sup, res = liquidity_levels(close, vol)
        s_liq = 0.0
        if sup and atr > 0:
            d = (price - sup) / atr
            if d < 1.5:
                s_liq = max(s_liq, 1.0 - d / 1.5)
        if res and atr > 0:
            d = (res - price) / atr
            if d < 1.5:
                s_liq = min(s_liq, -(1.0 - d / 1.5))
        s_liq = np.clip(s_liq, -1.0, 1.0)

        # Composite
        W = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
        S = np.array([s_mfi, s_cmf, s_obv, s_vwap, s_liq])
        composite = float(W @ S)

        sig = None
        if   composite >=  score_min: sig = "BUY"
        elif composite <= -score_min: sig = "SELL"
        if sig is None:
            return None

        sl    = round(price - atr * sl_atr if sig == "BUY" else price + atr * sl_atr, 2)
        tp    = round(price + atr * tp_atr if sig == "BUY" else price - atr * tp_atr, 2)
        price = round(price, 2)

        return dict(
            Ticker  = ticker,
            Signal  = sig,
            Price   = price,
            TP      = tp,
            SL      = sl,
            MFI     = round(mfi, 1),
            CMF     = round(cmf, 3),
            OBV     = "↑" if obv_roc > 0 else "↓",
            VWAP    = "Above" if price > vwap else "Below",
            Score   = round(composite, 3),
            _score  = composite,
            _sig    = sig,
        )
    except Exception:
        return None


# ═══════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="Smart Money Scanner 💰",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stButton > button {
    height: 3.2rem !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}
.card-buy {
    background: linear-gradient(135deg,#0a3d0a,#145214);
    border-left: 5px solid #00e676;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 7px 0;
    color: #e0ffe0;
    font-size: 1rem;
    line-height: 1.7;
}
.card-sell {
    background: linear-gradient(135deg,#3d0a0a,#521414);
    border-left: 5px solid #ff5252;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 7px 0;
    color: #ffe0e0;
    font-size: 1rem;
    line-height: 1.7;
}
.card-title { font-size:1.25rem; font-weight:800; letter-spacing:.5px; }
.tag-buy  { background:#00e676; color:#000; border-radius:5px;
            padding:2px 9px; font-weight:700; font-size:.85rem; }
.tag-sell { background:#ff5252; color:#fff; border-radius:5px;
            padding:2px 9px; font-weight:700; font-size:.85rem; }
.tf-badge { background:#1e3a5f; color:#7dd3fc; border-radius:6px;
            padding:3px 10px; font-size:.82rem; font-weight:700;
            margin-right:6px; }
.custom-tag {
    background: #1e3a5f; color: #7dd3fc; border-radius: 6px;
    padding: 3px 10px; font-size: .82rem; font-weight: 700;
}
.rank-badge {
    background: #ff9800; color: #000; border-radius: 6px;
    padding: 2px 8px; font-size: .78rem; font-weight: 800;
    margin-right: 6px;
}
.info-box {
    background: #1a1a2e; border: 1px solid #16213e;
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────
st.title("💰 Smart Money Scanner")
st.caption(f"NASDAQ Full Scan  •  {datetime.now():%d %b %Y  %H:%M}")
st.divider()

# ═══════════════════════════════════════════════
#  تحميل رموز NASDAQ
# ═══════════════════════════════════════════════
with st.spinner("📡 يجلب رموز NASDAQ من الموقع الرسمي..."):
    nasdaq_tuple, nasdaq_source, nasdaq_err = fetch_all_nasdaq_symbols()
    nasdaq_all = list(nasdaq_tuple)

if nasdaq_err:
    st.warning(f"⚠️ لم يتّصل بالموقع الرسمي ({nasdaq_err}) — يُستخدم احتياطي محلي")

st.markdown(
    f'<div class="info-box">'
    f'📊 <b>{len(nasdaq_all):,}</b> رمز NASDAQ محمّل  •  '
    f'المصدر: <i>{nasdaq_source}</i></div>',
    unsafe_allow_html=True,
)

# ── Settings ─────────────────────────────────────
with st.expander("⚙️ Settings", expanded=False):

    st.markdown("#### 📊 مؤشرات تدفق المال")
    c1, c2 = st.columns(2)
    with c1:
        mfi_period = st.slider("MFI Period", 8, 30, MFI_PERIOD, 1)
        mfi_os     = st.slider("MFI Oversold", 10, 35, 20, 5)
    with c2:
        cmf_period = st.slider("CMF Period", 10, 40, CMF_PERIOD, 2)
        mfi_ob     = st.slider("MFI Overbought", 65, 90, 80, 5)

    st.markdown("#### 🎯 عتبة الإشارة والخروج")
    score_min = st.slider("Score Threshold", 0.20, 0.80, 0.40, 0.05)
    c3, c4 = st.columns(2)
    with c3:
        sl_atr = st.slider("Stop Loss  × ATR", 0.5, 4.0, 1.5, 0.5)
    with c4:
        tp_atr = st.slider("Take Profit × ATR", 0.5, 6.0, 2.5, 0.5)

    st.markdown("#### 🧹 فلترة مبدئية (لتسريع المسح)")
    c5, c6 = st.columns(2)
    with c5:
        min_price  = st.slider("Min Price ($)", 1, 50, 3, 1,
                               help="تخطّي الأسهم أرخص من هذا الحد")
    with c6:
        min_vol_avg = st.slider("Min Avg Volume (K)", 50, 1000, 200, 50,
                                help="تخطّي الأسهم بأقل حجم تداول (بألف)")

    tf = st.selectbox(
        "Timeframe",
        options       = list(TF_CONFIG.keys()),
        index         = 2,
        format_func   = lambda x: TF_CONFIG[x]["label"],
    )

    if tf == "1h":
        st.info("⏰ **Hourly** — بيانات آخر 730 يوم · مناسب للصفقات اليومية")
    elif tf == "4h":
        st.info("🕓 **4-Hours** — يُعاد تجميعها من بيانات الساعة")
    elif tf == "1d":
        st.info("📅 **Daily** — بيانات سنتين · مناسب للصفقات الأسبوعية")
    else:
        st.info("📆 **Weekly** — بيانات 5 سنوات · مناسب للصفقات الشهرية")

    st.info(f"R:R Ratio = **1 : {tp_atr / sl_atr:.1f}**")

# ═══════════════════════════════════════════════
#  🔍 بحث + حد المسح + عدد النتائج
# ═══════════════════════════════════════════════
with st.expander("🔍 بحث وتحكم", expanded=True):

    search_input = st.text_input(
        "🔑 أضف رموز إضافية",
        placeholder="مثال:  AAPL, TSLA, BTC-USD",
    )

    custom_tickers = []
    if search_input.strip():
        raw = search_input.strip().upper()
        for sep in [",", "،", ";", "؛", " "]:
            raw = raw.replace(sep, " ")
        for p in raw.split():
            p = p.strip()
            if re.fullmatch(r"[A-Z][A-Z0-9.\-]*", p):
                custom_tickers.append(p)

    if custom_tickers:
        tags_html = "".join(f'<span class="custom-tag">{t}</span>' for t in custom_tickers)
        st.markdown(f'{tags_html} &nbsp;✅ {len(custom_tickers)} رمز إضافي', unsafe_allow_html=True)

    # ── حدود المسح ─────────────────────────────
    st.markdown("---")
    sc1, sc2 = st.columns(2)
    with sc1:
        max_stocks = st.selectbox(
            "📊 عدد الأسهم للمسح",
            options  = [100, 200, 500, 1000, 2000, 5000],
            index    = 2,    # default 500
            format_func = lambda x: f"{x:,} سهم",
        )
    with sc2:
        top_n = st.selectbox(
            "🏆 أعلى النتائج المعروضة",
            options  = [10, 20, 30, 50, 100, 200],
            index    = 2,    # default 30
            format_func = lambda x: f"أعلى {x}",
        )

    # تقدير الوقت
    est_batches = min(max_stocks, len(nasdaq_all)) // CHUNK_SIZE + 1
    est_sec     = est_batches * 3
    st.caption(
        f"⏱️ تقدير الوقت: ~{est_sec//60}د {est_sec%60:02d}ث "
        f"لأول مسح (الكاش يُسرّع التكرار)"
    )

# ── بناء القائمة النهائية ───────────────────────
stocks = list(dict.fromkeys(custom_tickers + nasdaq_all[:max_stocks]))
tf_label = TF_CONFIG[tf]["label"]

st.markdown(
    f'<span class="tf-badge">{tf_label}</span>'
    f'**{len(stocks):,} stocks** جاهزة للمسح',
    unsafe_allow_html=True,
)

# ── SCAN BUTTON ──────────────────────────────────
if st.button("🔍  SCAN NOW", type="primary", use_container_width=True):

    if not stocks:
        st.warning("لا توجد أسهم للمسح")
        st.stop()

    min_vol_actual = min_vol_avg * 1000  # تحويل من K

    # ── تحميل البيانات ─────────────────────────
    with st.spinner(f"📡 تحميل بيانات {len(stocks):,} سهم  [{tf_label}]..."):
        data_map = get_data_for_tf(stocks, tf)

    buys, sells, skipped, failed = [], [], 0, []
    bar = st.progress(0, text="جاري التحليل...")

    for i, ticker in enumerate(stocks):
        # عرض مباشر لعدد الإشارات
        n_sig = len(buys) + len(sells)
        bar.progress(
            (i + 1) / len(stocks),
            text=f"🔍 {ticker}  ({i+1}/{len(stocks):,})  |  "
                 f"إشارات: {n_sig}  |  "
                 f"تخطّي: {skipped}",
        )
        df = data_map.get(ticker)
        if df is None:
            failed.append(ticker)
            continue
        r = get_signal_from_df(
            ticker, df, mfi_os, mfi_ob, score_min,
            sl_atr, tp_atr, min_price, min_vol_actual,
            mfi_period, cmf_period,
            OBV_SLOPE_LEN, VWAP_PERIOD,
        )
        if r:
            (buys if r["_sig"] == "BUY" else sells).append(r)
        else:
            # نميّز بين تخطّي (فلتر) وفشل تحميل
            if df is not None and len(df) > 0:
                skipped += 1

    bar.empty()

    # ── ترتيب حسب |Score| ──────────────────────
    buys  = sorted(buys,  key=lambda x: x["_score"], reverse=True)
    sells = sorted(sells, key=lambda x: x["_score"])    # الأكثر سلبًا أولًا

    # ── تقرير الفشل ────────────────────────────
    if failed:
        fail_rate = len(failed) / len(stocks)
        if fail_rate > 0.3:
            st.error(
                f"⚠️ تعذّر تحميل {len(failed):,} من {len(stocks):,} سهم "
                f"({fail_rate:.0%}) — حظر مؤقت من ياهو. جرّب بعد شوية."
            )
        else:
            with st.expander(
                f"⚠️ تعذّر تحميل {len(failed)} رمز (حظر ياهو أو بيانات غير كافية)"
            ):
                st.write(", ".join(failed[:200]))
                if len(failed) > 200:
                    st.caption(f"... و {len(failed)-200} آخر")

    # ── Summary ──────────────────────────────────
    total = len(buys) + len(sells)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Signals", total)
    c2.metric("📈 BUY",     len(buys))
    c3.metric("📉 SELL",   len(sells))
    c4.metric("⏭️ Skipped",  skipped)

    st.caption(
        f"Completed {datetime.now():%H:%M:%S}  •  "
        f"{tf_label}  •  Score≥{score_min}  •  "
        f"MFI({mfi_period}) {mfi_os}/{mfi_ob}  •  "
        f"CMF({cmf_period})  •  "
        f"Min ${min_price}  •  Min Vol {min_vol_avg}K  •  "
        f"SL:{sl_atr}×ATR  •  TP:{tp_atr}×ATR"
    )
    st.divider()

    if total == 0:
        st.warning("⬜ لا توجد إشارات — لا إجماع كافٍ بين مؤشرات تدفق المال")
        st.stop()

    # ── اقتصار على أعلى N ───────────────────────
    buys_show  = buys[:top_n]
    sells_show = sells[:top_n]

    if total > top_n:
        st.info(
            f"🏆 يُعرض أعلى **{top_n}** نتيجة من أصل **{total}** إشارة  "
            f"(غيّر العدد من ⚙️ التحكم)"
        )

    # ── BUY Cards ────────────────────────────────
    if buys_show:
        extra_b = f" من أصل {len(buys)}" if len(buys) > top_n else ""
        st.markdown(f"### 📈 BUY &nbsp;<small>({len(buys_show)}{extra_b})</small>",
                    unsafe_allow_html=True)
        for rank, r in enumerate(buys_show, 1):
            st.markdown(f"""
            <div class="card-buy">
              <span class="rank-badge">#{rank}</span>
              <span class="card-title">{r['Ticker']}</span>
              &nbsp;&nbsp;<span class="tag-buy">BUY</span>
              &nbsp;&nbsp;<span style="font-size:.82rem;opacity:.8">
                Score: <b>{r['Score']}</b></span><br>
              💰 Price : <b>${r['Price']:,.2f}</b><br>
              🎯 TP &nbsp;&nbsp;&nbsp;: <b>${r['TP']:,.2f}</b><br>
              🛑 SL &nbsp;&nbsp;&nbsp;: <b>${r['SL']:,.2f}</b><br>
              ━━━━━━━━━━━━━━━━━━━━━━━━<br>
              📊 MFI &nbsp;&nbsp;: <b>{r['MFI']}</b>
              &nbsp;|&nbsp; 💧 CMF : <b>{r['CMF']}</b><br>
              📦 OBV &nbsp;&nbsp;: <b>{r['OBV']}</b>
              &nbsp;|&nbsp; 📐 VWAP : <b>{r['VWAP']}</b>
            </div>""", unsafe_allow_html=True)

    # ── SELL Cards ───────────────────────────────
    if sells_show:
        extra_s = f" من أصل {len(sells)}" if len(sells) > top_n else ""
        st.markdown(f"### 📉 SELL &nbsp;<small>({len(sells_show)}{extra_s})</small>",
                    unsafe_allow_html=True)
        for rank, r in enumerate(sells_show, 1):
            st.markdown(f"""
            <div class="card-sell">
              <span class="rank-badge">#{rank}</span>
              <span class="card-title">{r['Ticker']}</span>
              &nbsp;&nbsp;<span class="tag-sell">SELL</span>
              &nbsp;&nbsp;<span style="font-size:.82rem;opacity:.8">
                Score: <b>{r['Score']}</b></span><br>
              💰 Price : <b>${r['Price']:,.2f}</b><br>
              🎯 TP &nbsp;&nbsp;&nbsp;: <b>${r['TP']:,.2f}</b><br>
              🛑 SL &nbsp;&nbsp;&nbsp;: <b>${r['SL']:,.2f}</b><br>
              ━━━━━━━━━━━━━━━━━━━━━━━━<br>
              📊 MFI &nbsp;&nbsp;: <b>{r['MFI']}</b>
              &nbsp;|&nbsp; 💧 CMF : <b>{r['CMF']}</b><br>
              📦 OBV &nbsp;&nbsp;: <b>{r['OBV']}</b>
              &nbsp;|&nbsp; 📐 VWAP : <b>{r['VWAP']}</b>
            </div>""", unsafe_allow_html=True)

    # ── Download CSV ──────────────────────────────
    st.divider()
    df_out = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in buys + sells   # كل الإشارات، مش بس المعروضة
    ])
    df_out.insert(0, "Rank", range(1, len(df_out) + 1))
    df_out.insert(3, "Timeframe", tf_label)
    st.download_button(
        "💾  تنزيل كل النتائج CSV",
        data      = df_out.to_csv(index=False).encode("utf-8"),
        file_name = f"smart_money_nasdaq_{tf}_{datetime.now():%Y%m%d_%H%M}.csv",
        mime      = "text/csv",
        use_container_width=True,
    )
