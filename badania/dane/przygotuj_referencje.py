# -*- coding: utf-8 -*-

"""Przygotowuje warstwę referencyjną do oceny dokładności; etap weryfikacji punktów wykonywany jest ręcznie.

WEJŚCIE:      warstwy BDOT10k w podkatalogu SHP, scena Sentinel-2, obszary treningowe
WYJŚCIE:      dane_testowe/referencja2/referencja_FINAL.shp oraz warstwy pośrednie
URUCHOMIENIE: otwarcie pliku w edytorze konsoli Pythona QGIS i ustawienie zmiennej MODE
"""

import os
import random
from itertools import accumulate
import processing
from qgis.core import (
    QgsVectorLayer, QgsRasterLayer, QgsProject, QgsField,
    QgsVectorFileWriter, QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsGeometry, QgsPoint, QgsPointXY, QgsFeature, QgsRaster
)
from qgis.PyQt.QtCore import QVariant

MODE = 'prepare'

try:
    _BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DOMYSLNY = os.path.join(_BADANIA, 'dane_testowe')
except NameError:
    _DOMYSLNY = ''
BASE     = os.environ.get('DANE_DIR', _DOMYSLNY)
if not BASE:
    raise SystemExit('Nie ustalono katalogu danych. Podaj go zmienną DANE_DIR '
                     'albo wpisz ścieżkę w zmiennej BASE poniżej sekcji CONFIG.')
OUT      = os.path.join(BASE, 'referencja2')
RASTER   = os.path.join(BASE, 'L1C/L1C_przyciety.tif')
ZASIEG   = os.path.join(OUT, 'maska2.shp')
ROI_TRAIN = os.path.join(BASE, 'ROI/roi_1234567_duzeSHP.shp')

CRS_TARGET = 'EPSG:32634'
PIXEL      = 10.0
MIN_DIST   = 50

ROI_BUFOR = 15

BDOT = os.path.join(BASE, 'SHP')

WARSTWY = {
    'pole':     os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_PTTR_A.shp'),
    'las':      os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_PTLZ_A.shp'),
    'woda':     os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_PTWP_A.shp'),
    'plaza':    os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_PTGN_A.shp'),
    'zabudowa': os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_BUBD_A.shp'),
    'droga_l':  os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_SKDR_L.shp'),
    'droga_a':  os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_PTKM_A.shp'),
}

WERSJA_BDOT = None

KLASY = {
    'pole':     1,
    'droga':    2,
    'plaza':    3,
    'las':      4,
    'zabudowa': 5,
    'woda':     6,
}

LICZBA_PUNKTOW = {
    'pole':     500,
    'droga':    188,
    'plaza':    134,
    'las':      80,
    'zabudowa': 78,
    'woda':     75,
}

FILTR_POLE = "\"RODZAJ\" = 'uprawa na gruntach ornych'"
FILTR_DROGA = (
    "\"KLASA_DROGI\" IN ("
    "'droga ekspresowa',"
    "'droga główna ruchu przyspieszonego',"
    "'droga główna',"
    "'droga zbiorcza')"
)
KANAL_GREEN = 3
KANAL_RED   = 4
KANAL_NIR   = 8

NDVI_POLE_GOLE = 0.30
NDVI_POLE_ZIEL = 0.40

MIN_POW_BUDYNKU = 1000
BUFOR_BUDYNKU   = -5
BUFOR_DROGI     = 4
PAS_PLAZY       = 200

def _p(name):
    return os.path.join(OUT, name)

def _log(msg):
    print(f"  {msg}")

def _run(alg, params):
    return processing.run(alg, params)['OUTPUT']

def _wczytaj(path, nazwa):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Brak pliku: {path}")
    lyr = QgsVectorLayer(path, nazwa, 'ogr')
    if not lyr.isValid():
        raise RuntimeError(f"Nie mozna wczytac: {path}")
    return lyr

def _filtruj(lyr, wyrazenie):
    if not wyrazenie:
        return lyr
    return _run('native:extractbyexpression',
                {'INPUT': lyr, 'EXPRESSION': wyrazenie, 'OUTPUT': 'memory:'})

def _dodaj_wersje(wyrazenie):
    if not WERSJA_BDOT:
        return wyrazenie
    w = f"\"WERSJA\" = '{WERSJA_BDOT}'"
    return f"({wyrazenie}) AND {w}" if wyrazenie else w

def _reproj(lyr):
    return _run('native:reprojectlayer',
                {'INPUT': lyr, 'TARGET_CRS': QgsCoordinateReferenceSystem(CRS_TARGET),
                 'OUTPUT': 'memory:'})

def _przytnij(lyr, maska):
    return _run('native:clip', {'INPUT': lyr, 'OVERLAY': maska, 'OUTPUT': 'memory:'})

def _dissolve(lyr):
    return _run('native:dissolve', {'INPUT': lyr, 'FIELD': [], 'OUTPUT': 'memory:'})

def _bufor(lyr, dist, styl_plaski=False):
    return _run('native:buffer', {
        'INPUT': lyr, 'DISTANCE': dist, 'SEGMENTS': 8,
        'END_CAP_STYLE': 1 if styl_plaski else 0,
        'JOIN_STYLE': 0, 'MITER_LIMIT': 2,
        'DISSOLVE': True, 'OUTPUT': 'memory:'})

def _zapisz(lyr, sciezka):
    crs = QgsCoordinateReferenceSystem(CRS_TARGET)
    if hasattr(QgsVectorFileWriter, 'writeAsVectorFormatV3'):
        ctx = QgsProject.instance().transformContext()
        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = 'ESRI Shapefile'
        opts.fileEncoding = 'utf-8'
        if lyr.crs() != crs:
            opts.ct = QgsCoordinateTransform(lyr.crs(), crs, ctx)
        wynik = QgsVectorFileWriter.writeAsVectorFormatV3(lyr, sciezka, ctx, opts)
    else:
        wynik = QgsVectorFileWriter.writeAsVectorFormat(
            lyr, sciezka, 'utf-8', crs, 'ESRI Shapefile')
    kod = wynik[0] if isinstance(wynik, (tuple, list)) else wynik
    try:
        blad = (kod != QgsVectorFileWriter.NoError)
    except Exception:
        blad = bool(kod)
    if blad or not os.path.exists(sciezka):
        raise RuntimeError(f"ZAPIS NIEUDANY: {sciezka} (wynik: {wynik})")
    return sciezka

def _wykluczenia_roi():
    roi = _wczytaj(ROI_TRAIN, 'roi_train')
    crs = QgsCoordinateReferenceSystem(CRS_TARGET)
    tr = None
    if roi.crs() != crs:
        tr = QgsCoordinateTransform(roi.crs(), crs,
                                    QgsProject.instance().transformContext())
    silniki = []
    for f in roi.getFeatures():
        g = f.geometry()
        if g is None or g.isEmpty():
            continue
        if tr:
            g.transform(tr)
        if ROI_BUFOR > 0:
            g = g.buffer(ROI_BUFOR, 8)
        eng = QgsGeometry.createGeometryEngine(g.constGet())
        eng.prepareGeometry()
        silniki.append((g, eng))
    _log(f"Wykluczenie ROI treningowego: {len(silniki)} poligonow "
         f"(+ bufor {ROI_BUFOR} m)")
    return silniki

def _zasieg_rastra():
    if os.path.exists(ZASIEG):
        return _wczytaj(ZASIEG, 'zasieg')
    r = QgsRasterLayer(RASTER, 'raster')
    if not r.isValid():
        raise RuntimeError(f"Nie mozna wczytac rastra: {RASTER}")
    ext = r.extent()
    ext_str = f"{ext.xMinimum()},{ext.xMaximum()},{ext.yMinimum()},{ext.yMaximum()}"
    lyr = _run('native:extenttolayer',
               {'INPUT': f"{ext_str} [{r.crs().authid()}]", 'OUTPUT': 'memory:'})
    _zapisz(lyr, ZASIEG)
    _log(f"Zasieg sceny: {ext.width()/1000:.1f} x {ext.height()/1000:.1f} km")
    return lyr

def _origin_rastra():
    r = QgsRasterLayer(RASTER, 'raster')
    ext = r.extent()
    return ext.xMinimum(), ext.yMaximum()

def etap1_prepare():
    print("\n=== ETAP 1: przygotowanie warstw BDOT ===\n")
    os.makedirs(OUT, exist_ok=True)
    zasieg = _zasieg_rastra()

    proste = {
        'pole': (WARSTWY['pole'], FILTR_POLE),
        'las':  (WARSTWY['las'],  None),
        'woda': (WARSTWY['woda'], None),
    }
    for nazwa, (sciezka, filtr) in proste.items():
        _log(f"[{nazwa}]")
        lyr = _wczytaj(sciezka, nazwa)
        lyr = _filtruj(lyr, _dodaj_wersje(filtr))
        _log(f"  po filtrze: {lyr.featureCount()} obiektow")
        lyr = _reproj(lyr)
        lyr = _przytnij(lyr, zasieg)
        lyr = _dissolve(lyr)
        _zapisz(lyr, _p(f'bdot_{nazwa}.shp'))
        _log(f"  -> bdot_{nazwa}.shp")

    _log("[zabudowa]")
    lyr = _wczytaj(WARSTWY['zabudowa'], 'zabudowa')
    lyr = _filtruj(lyr, _dodaj_wersje(None))
    lyr = _reproj(lyr)
    lyr = _run('native:extractbyexpression',
               {'INPUT': lyr, 'EXPRESSION': f'"pow_m2" > {MIN_POW_BUDYNKU}',
                'OUTPUT': 'memory:'})
    _log(f"  budynkow > {MIN_POW_BUDYNKU} m2: {lyr.featureCount()}")
    lyr = _przytnij(lyr, zasieg)
    lyr = _bufor(lyr, BUFOR_BUDYNKU)
    lyr = _run('native:multiparttosingleparts', {'INPUT': lyr, 'OUTPUT': 'memory:'})
    lyr = _run('native:extractbyexpression',
               {'INPUT': lyr, 'EXPRESSION': 'area($geometry) > 100', 'OUTPUT': 'memory:'})
    lyr = _dissolve(lyr)
    _zapisz(lyr, _p('bdot_zabudowa.shp'))
    _log(f"  -> bdot_zabudowa.shp (bufor {BUFOR_BUDYNKU} m)")

    _log("[plaza]")
    woda = _wczytaj(_p('bdot_woda.shp'), 'woda')
    pas = _bufor(woda, PAS_PLAZY)
    lyr = _wczytaj(WARSTWY['plaza'], 'plaza')
    lyr = _filtruj(lyr, _dodaj_wersje(None))
    lyr = _reproj(lyr)
    lyr = _przytnij(lyr, zasieg)
    lyr = _przytnij(lyr, pas)
    lyr = _dissolve(lyr)
    _zapisz(lyr, _p('bdot_plaza.shp'))
    _log(f"  -> bdot_plaza.shp (pas {PAS_PLAZY} m od wody)")

    _log("[droga]")
    linie = _wczytaj(WARSTWY['droga_l'], 'droga_l')
    linie = _filtruj(linie, _dodaj_wersje(FILTR_DROGA))
    _log(f"  odcinkow po filtrze klasy drogi: {linie.featureCount()}")
    linie = _reproj(linie)
    linie = _przytnij(linie, zasieg)
    linie = _bufor(linie, BUFOR_DROGI, styl_plaski=True)

    czesci = [linie]
    if os.path.exists(WARSTWY['droga_a']):
        pol = _wczytaj(WARSTWY['droga_a'], 'droga_a')
        pol = _filtruj(pol, _dodaj_wersje(None))
        pol = _reproj(pol)
        pol = _przytnij(pol, zasieg)
        czesci.append(pol)
        _log(f"  PTKM_A: {pol.featureCount()} obiektow")
    else:
        _log("  PTKM_A nie znaleziony -- pomijam")

    lyr = _run('native:mergevectorlayers',
               {'LAYERS': czesci, 'CRS': QgsCoordinateReferenceSystem(CRS_TARGET),
                'OUTPUT': 'memory:'})
    lyr = _dissolve(lyr)
    _zapisz(lyr, _p('bdot_droga.shp'))
    _log(f"  -> bdot_droga.shp (bufor {BUFOR_DROGI} m od osi)")

    print("\nGOTOWE. Obejrzyj warstwy bdot_*.shp w QGIS przed dalszym krokiem.")
    print("Nastepny krok: MODE = 'pilot'\n")

def _losuj(nazwa, n, plik_wy, wyklucz=None):
    zrodlo = _p(f'bdot_{nazwa}.shp')
    lyr = _wczytaj(zrodlo, nazwa)

    czesci = []
    for f in lyr.getFeatures():
        g = f.geometry()
        if g is None or g.isEmpty():
            continue
        for part in g.asGeometryCollection():
            a = part.area()
            if a <= 0:
                continue
            eng = QgsGeometry.createGeometryEngine(part.constGet())
            eng.prepareGeometry()
            czesci.append((part, eng, part.boundingBox(), a))
    if not czesci:
        raise RuntimeError(f"{zrodlo}: brak poligonow do losowania")

    cum_wagi = list(accumulate(c[3] for c in czesci))
    rng = random.Random(42)
    pkt = []
    proby, limit = 0, max(20000, n * 500)
    while len(pkt) < n and proby < limit:
        proby += 1
        part, eng, bb, _a = rng.choices(czesci, cum_weights=cum_wagi)[0]
        x = rng.uniform(bb.xMinimum(), bb.xMaximum())
        y = rng.uniform(bb.yMinimum(), bb.yMaximum())
        p = QgsPoint(x, y)
        if not eng.contains(p):
            continue
        if wyklucz and any(e.intersects(p) for _g, e in wyklucz):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < MIN_DIST ** 2 for px, py in pkt):
            continue
        pkt.append((x, y))

    pts = QgsVectorLayer(f'Point?crs={CRS_TARGET}&field=ref_7:integer',
                         nazwa, 'memory')
    feats = []
    for x, y in pkt:
        f = QgsFeature(pts.fields())
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        f['ref_7'] = KLASY[nazwa]
        feats.append(f)
    pts.dataProvider().addFeatures(feats)
    pts.updateExtents()

    _zapisz(pts, plik_wy)
    _log(f"[{nazwa}] wylosowano {pts.featureCount()} / {n} -> {os.path.basename(plik_wy)}")
    if pts.featureCount() < n:
        _log(f"  UWAGA: mniej niz zadano. Za malo miejsca przy MIN_DISTANCE={MIN_DIST} m.")
    return pts

def etap2_pilot():
    print("\n=== ETAP 2: kalibracja pilotazowa (30 pkt/klase) ===\n")
    os.makedirs(_p('pilot'), exist_ok=True)
    for nazwa in ['droga', 'zabudowa', 'plaza', 'pole', 'las', 'woda']:
        _losuj(nazwa, 30, _p(f'pilot/pilot_{nazwa}.shp'))

    print("""
KALIBRACJA LICZEBNOŚCI PRÓBY:
  1. Warstwy pilot_*.shp wczytuje się w QGIS na tle ortofoto.
  2. Dla każdej klasy zlicza się, ile z 30 punktów leży faktycznie na obiekcie
     tej klasy (droga = asfalt, zabudowa = dach, plaża = piasek).
  3. p = trafienia / 30
  4. n = 75 / p, zaokrąglone w górę
  5. Wartość n trafia do LICZBA_PUNKTOW w sekcji CONFIG.
  6. MODE = 'sample' i ponowne uruchomienie.

PRZYKŁAD: droga 12/30 -> p = 0,40 -> n = 75/0,40 = 188
""")

def etap3_sample():
    print("\n=== ETAP 3: losowanie wlasciwe ===\n")
    brak = [k for k, v in LICZBA_PUNKTOW.items() if v is None]
    if brak:
        raise ValueError(
            f"Uzupelnij LICZBA_PUNKTOW dla: {brak}. Najpierw uruchom MODE='pilot'.")

    os.makedirs(_p('punkty'), exist_ok=True)
    wyklucz = _wykluczenia_roi()
    warstwy = []
    for nazwa, n in LICZBA_PUNKTOW.items():
        plik = _p(f'punkty/punkty_{nazwa}.shp')
        _losuj(nazwa, n, plik, wyklucz=wyklucz)
        warstwy.append(plik)

    scalone = _run('native:mergevectorlayers', {
        'LAYERS': warstwy,
        'CRS': QgsCoordinateReferenceSystem(CRS_TARGET),
        'OUTPUT': 'memory:'})
    _zapisz(scalone, _p('punkty_do_weryfikacji.shp'))

    print(f"\nRAZEM: {scalone.featureCount()} punktow")
    print(f"-> {_p('punkty_do_weryfikacji.shp')}")
    print("""
ETAP 4 -- WERYFIKACJA RĘCZNA:
  1. Warstwę punkty_do_weryfikacji.shp wczytuje się na tle ortofoto
     Gdańsk 2024.
  2. Po włączeniu edycji przechodzi się punkt po punkcie.
  3. W polu ref_7 zapisuje się klasę widoczną na ortofoto:
       1 pole (zaorane)      2 droga     3 plaża
       4 las                 5 zabudowa  6 woda     7 pole z roślinnością
  4. Punkt na poboczu, w ogrodzie lub na wydmie otrzymuje zmienioną
     etykietę, nie jest usuwany.
  5. Usuwane są wyłącznie punkty niejednoznaczne (cień, granica obiektów).
     Ich liczba wchodzi do metodyki.
  6. Wynik zapisuje się jako referencja_zweryfikowana.shp
  7. MODE = 'polygons' i ponowne uruchomienie.
""")

def etap4_assist():
    print("\n=== ETAP 4a: wspomaganie weryfikacji (NDVI) ===\n")
    zrodlo = _p('punkty_do_weryfikacji.shp')
    if not os.path.exists(zrodlo):
        raise FileNotFoundError(f"Brak {zrodlo}. Najpierw MODE='sample'.")

    lyr = _wczytaj(zrodlo, 'punkty')
    r = QgsRasterLayer(RASTER, 'raster')
    if not r.isValid():
        raise RuntimeError(f"Nie mozna wczytac rastra: {RASTER}")
    prov = r.dataProvider()

    out = QgsVectorLayer(
        f'Point?crs={CRS_TARGET}'
        '&field=ref_7:integer&field=ndvi:double&field=ndwi:double'
        '&field=sugestia:integer&field=flaga:string(30)',
        'assist', 'memory')

    def _idx(a, b):
        return (a - b) / (a + b) if (a is not None and b is not None
                                     and (a + b) != 0) else None

    NAZWY = {1: 'pole zaorane', 2: 'droga', 3: 'plaza', 4: 'las',
             5: 'zabudowa', 6: 'woda', 7: 'pole z rosl.'}

    def _sugestia(k, ndvi, ndwi):
        if ndvi is None:
            return None
        if ndwi is not None and ndwi > 0.10 and ndvi < 0.05:
            return 6
        if ndvi >= NDVI_POLE_ZIEL:
            return 4 if k == 4 else 7
        if ndvi <= NDVI_POLE_GOLE:
            if k in (1, 2, 3, 5):
                return 1 if k == 1 else k
            return None
        if k == 4:
            return 4
        return None

    feats, n_flagi, pole_1, pole_7, pole_np = [], 0, 0, 0, 0
    for f in lyr.getFeatures():
        pt = f.geometry().asPoint()
        wart = prov.identify(pt, QgsRaster.IdentifyFormatValue).results()
        green = wart.get(KANAL_GREEN)
        red   = wart.get(KANAL_RED)
        nir   = wart.get(KANAL_NIR)
        ndvi = _idx(nir, red)
        ndwi = _idx(green, nir)

        k = f['ref_7']
        sug = _sugestia(k, ndvi, ndwi)
        if k == 1:
            if sug == 1:
                pole_1 += 1
            elif sug == 7:
                pole_7 += 1
            else:
                pole_np += 1

        if ndvi is None:
            flaga = 'poza rastrem'
        elif sug is None:
            flaga = 'spektralnie niepewne'
        elif k == 1 and sug in (1, 7):
            flaga = ''
        elif sug != k:
            flaga = f'wyglada jak {NAZWY[sug]}'
        else:
            flaga = ''
        if flaga:
            n_flagi += 1

        nf = QgsFeature(out.fields())
        nf.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(pt)))
        nf['ref_7'] = k
        nf['ndvi'] = round(ndvi, 4) if ndvi is not None else None
        nf['ndwi'] = round(ndwi, 4) if ndwi is not None else None
        nf['sugestia'] = sug
        nf['flaga'] = flaga
        feats.append(nf)

    out.dataProvider().addFeatures(feats)
    out.updateExtents()
    _zapisz(out, _p('punkty_do_weryfikacji_assist.shp'))

    print(f"  Punktow: {len(feats)}, oflagowanych: {n_flagi}")
    print(f"  Pole -- sugestie: {pole_1} x zaorane (1), {pole_7} x roslinnosc (7), "
          f"{pole_np} niepewnych")
    print(f"""
-> punkty_do_weryfikacji_assist.shp

SPOSÓB UŻYCIA (etap 4 pozostaje ręczny):
  1. Warstwę wczytuje się na tle ortofoto.
  2. Styl warstwy: kategoryzacja po polu 'flaga' -- oflagowane na czerwono.
  3. Weryfikacja przebiega punkt po punkcie jak dotąd; przy polach kolumna
     'sugestia' podpowiada podział 1/7, potwierdzany lub korygowany
     wyłącznie według ortofoto.
  4. Wynik zapisuje się jako referencja_zweryfikowana.shp; kolumny
     ndvi/ndwi/sugestia/flaga można usunąć albo zostawić, nie przeszkadzają.
""")

def etap5_polygons():
    print("\n=== ETAP 5: punkty -> kwadraty 8x8 m ===\n")
    zrodlo = _p('referencja_zweryfikowana3.shp')
    if not os.path.exists(zrodlo):
        raise FileNotFoundError(
            f"Brak {zrodlo}. Najpierw wykonaj ETAP 4 (weryfikacja na ortofoto).")

    lyr = _wczytaj(zrodlo, 'ref')
    x0, y0 = _origin_rastra()
    _log(f"Origin rastra: X0={x0}, Y0={y0}")

    expr = (f"make_point("
            f"floor((x($geometry)-{x0})/{PIXEL})*{PIXEL} + {x0} + {PIXEL/2}, "
            f"floor((y($geometry)-{y0})/{PIXEL})*{PIXEL} + {y0} + {PIXEL/2})")
    snapped = _run('native:geometrybyexpression', {
        'INPUT': lyr, 'OUTPUT_GEOMETRY': 2,
        'WITH_Z': False, 'WITH_M': False,
        'EXPRESSION': expr, 'OUTPUT': 'memory:'})
    _log("Punkty przesuniete do srodkow pikseli")

    kwadraty = _run('native:buffer', {
        'INPUT': snapped, 'DISTANCE': PIXEL * 0.4,
        'SEGMENTS': 1, 'END_CAP_STYLE': 2,
        'JOIN_STYLE': 2, 'MITER_LIMIT': 2,
        'DISSOLVE': False, 'OUTPUT': 'memory:'})

    _zapisz(kwadraty, _p('referencja_FINAL.shp'))
    _log(f"-> referencja_FINAL.shp ({kwadraty.featureCount()} poligonow)")

    eps = 1e-6
    zle = 0
    for f in kwadraty.getFeatures():
        bb = f.geometry().boundingBox()
        kol1 = int((bb.xMinimum() - x0 + eps) // PIXEL)
        kol2 = int((bb.xMaximum() - x0 - eps) // PIXEL)
        w1 = int((y0 - bb.yMaximum() + eps) // PIXEL)
        w2 = int((y0 - bb.yMinimum() - eps) // PIXEL)
        if kol1 != kol2 or w1 != w2:
            zle += 1

    print("\n  Kontrola -- wpasowanie kwadratow w piksele:")
    if zle:
        print(f"    >>> BLAD: {zle} kwadratow przekracza granice piksela <<<")
        print("    Sprawdz origin rastra i wartosc PIXEL w CONFIG.")
    else:
        print(f"    OK -- wszystkie {kwadraty.featureCount()} kwadratow "
              f"lezy w calosci w jednym pikselu.")

    print("\nNastepny krok: MODE = 'check'\n")

def etap6_check():
    print("\n=== ETAP 6: kontrola jakosci ===\n")
    ref = _wczytaj(_p('referencja_FINAL.shp'), 'ref')

    print("[Test 1] Nakladanie na ROI treningowe")
    if os.path.exists(ROI_TRAIN):
        train = _wczytaj(ROI_TRAIN, 'train')
        train = _reproj(train)
        inter = _run('native:intersection',
                     {'INPUT': ref, 'OVERLAY': train,
                      'INPUT_FIELDS': [], 'OVERLAY_FIELDS': [], 'OUTPUT': 'memory:'})
        n = inter.featureCount()
        if n == 0:
            print("  OK -- brak nakladania. Referencja niezalezna od treningu.\n")
        else:
            print(f"  >>> UWAGA: {n} poligonow referencyjnych nakłada sie na ROI! <<<")
            _zapisz(inter, _p('KOLIZJE_z_treningiem.shp'))
            print(f"  Zapisano: KOLIZJE_z_treningiem.shp -- usun te punkty.\n")
    else:
        print(f"  Pominieto -- brak pliku {ROI_TRAIN}\n")

    print("[Test 2] Liczebnosc klas")
    NAZWY = {1: 'pole zaorane', 2: 'droga', 3: 'plaza', 4: 'las',
             5: 'zabudowa', 6: 'woda', 7: 'pole z roslinnoscia'}
    licz = {}
    for f in ref.getFeatures():
        k = f['ref_7']
        licz[k] = licz.get(k, 0) + 1

    ok = True
    for k in sorted(licz):
        flaga = "OK" if licz[k] >= 75 else ">>> < 75 <<<"
        if licz[k] < 75:
            ok = False
        print(f"  {k} {NAZWY.get(k, '?'):22s} {licz[k]:4d}   {flaga}")
    print(f"  {'RAZEM':26s} {sum(licz.values()):4d}")
    if not ok:
        print("\n  Dolosuj punkty dla klas ponizej 75 i zweryfikuj je na ortofoto.")

    print("\n[Test 3] Typ pola ref_7")
    idx = ref.fields().indexOf('ref_7')
    typ = ref.fields().at(idx).typeName()
    print(f"  ref_7: {typ}   {'OK' if 'nt' in typ else '>>> musi byc Integer <<<'}")

    print("""
Dalszy krok: warianty zestawu klas powstają ze zweryfikowanej warstwy
referencja_FINAL.shp skryptem badania/dane/generuj_warianty_referencja.py,
który wytwarza 8 plików referencja_<wariant>.shp z polem 'klasa'.
""")

TRYBY = {
    'prepare':  etap1_prepare,
    'pilot':    etap2_pilot,
    'sample':   etap3_sample,
    'assist':   etap4_assist,
    'polygons': etap5_polygons,
    'check':    etap6_check,
}

if MODE not in TRYBY:
    raise ValueError(f"Nieznany MODE: {MODE}. Dostepne: {list(TRYBY)}")

os.makedirs(OUT, exist_ok=True)
TRYBY[MODE]()
