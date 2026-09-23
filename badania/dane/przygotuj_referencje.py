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
    _DEFAULT = os.path.join(_BADANIA, 'dane_testowe')
except NameError:
    _DEFAULT = ''
BASE     = os.environ.get('DANE_DIR', _DEFAULT)
if not BASE:
    raise SystemExit('Nie ustalono katalogu danych. Podaj go zmienną DANE_DIR '
                     'albo wpisz ścieżkę w zmiennej BASE poniżej sekcji CONFIG.')
OUT      = os.path.join(BASE, 'referencja2')
RASTER   = os.path.join(BASE, 'L1C/L1C_przyciety.tif')
EXTENT_SHP   = os.path.join(OUT, 'maska2.shp')
ROI_TRAIN = os.path.join(BASE, 'ROI/roi_1234567_duzeSHP.shp')

CRS_TARGET = 'EPSG:32634'
PIXEL      = 10.0
MIN_DIST   = 50

ROI_BUFFER = 15

BDOT = os.path.join(BASE, 'SHP')

LAYERS = {
    'pole':     os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_PTTR_A.shp'),
    'las':      os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_PTLZ_A.shp'),
    'woda':     os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_PTWP_A.shp'),
    'plaza':    os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_PTGN_A.shp'),
    'zabudowa': os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_BUBD_A.shp'),
    'droga_l':  os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_SKDR_L.shp'),
    'droga_a':  os.path.join(BDOT, 'PL.PZGiK.336.BDOT10k.2261__OT_PTKM_A.shp'),
}

BDOT_VERSION = None

CLASSES = {
    'pole':     1,
    'droga':    2,
    'plaza':    3,
    'las':      4,
    'zabudowa': 5,
    'woda':     6,
}

POINT_COUNTS = {
    'pole':     500,
    'droga':    188,
    'plaza':    134,
    'las':      80,
    'zabudowa': 78,
    'woda':     75,
}

FIELD_FILTER = "\"RODZAJ\" = 'uprawa na gruntach ornych'"
ROAD_FILTER = (
    "\"KLASA_DROGI\" IN ("
    "'droga ekspresowa',"
    "'droga główna ruchu przyspieszonego',"
    "'droga główna',"
    "'droga zbiorcza')"
)
BAND_GREEN = 3
BAND_RED   = 4
BAND_NIR   = 8

NDVI_BARE_FIELD = 0.30
NDVI_GREEN_FIELD = 0.40

MIN_BUILDING_AREA = 1000
BUILDING_BUFFER   = -5
ROAD_BUFFER     = 4
BEACH_STRIP       = 200

def _p(name):
    return os.path.join(OUT, name)

def _log(msg):
    print(f"  {msg}")

def _run(alg, params):
    return processing.run(alg, params)['OUTPUT']

def _load(path, layer_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Brak pliku: {path}")
    lyr = QgsVectorLayer(path, layer_name, 'ogr')
    if not lyr.isValid():
        raise RuntimeError(f"Nie mozna wczytac: {path}")
    return lyr

def _filter_layer(lyr, expression):
    if not expression:
        return lyr
    return _run('native:extractbyexpression',
                {'INPUT': lyr, 'EXPRESSION': expression, 'OUTPUT': 'memory:'})

def _add_version(expression):
    if not BDOT_VERSION:
        return expression
    w = f"\"WERSJA\" = '{BDOT_VERSION}'"
    return f"({expression}) AND {w}" if expression else w

def _reproj(lyr):
    return _run('native:reprojectlayer',
                {'INPUT': lyr, 'TARGET_CRS': QgsCoordinateReferenceSystem(CRS_TARGET),
                 'OUTPUT': 'memory:'})

def _clip(lyr, mask_layer):
    return _run('native:clip', {'INPUT': lyr, 'OVERLAY': mask_layer, 'OUTPUT': 'memory:'})

def _dissolve(lyr):
    return _run('native:dissolve', {'INPUT': lyr, 'FIELD': [], 'OUTPUT': 'memory:'})

def _buffer(lyr, dist, styl_plaski=False):
    return _run('native:buffer', {
        'INPUT': lyr, 'DISTANCE': dist, 'SEGMENTS': 8,
        'END_CAP_STYLE': 1 if styl_plaski else 0,
        'JOIN_STYLE': 0, 'MITER_LIMIT': 2,
        'DISSOLVE': True, 'OUTPUT': 'memory:'})

def _save(lyr, dest_path):
    crs = QgsCoordinateReferenceSystem(CRS_TARGET)
    if hasattr(QgsVectorFileWriter, 'writeAsVectorFormatV3'):
        ctx = QgsProject.instance().transformContext()
        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = 'ESRI Shapefile'
        opts.fileEncoding = 'utf-8'
        if lyr.crs() != crs:
            opts.ct = QgsCoordinateTransform(lyr.crs(), crs, ctx)
        result = QgsVectorFileWriter.writeAsVectorFormatV3(lyr, dest_path, ctx, opts)
    else:
        result = QgsVectorFileWriter.writeAsVectorFormat(
            lyr, dest_path, 'utf-8', crs, 'ESRI Shapefile')
    code = result[0] if isinstance(result, (tuple, list)) else result
    try:
        has_error = (code != QgsVectorFileWriter.NoError)
    except Exception:
        has_error = bool(code)
    if has_error or not os.path.exists(dest_path):
        raise RuntimeError(f"ZAPIS NIEUDANY: {dest_path} (wynik: {result})")
    return dest_path

def _roi_exclusions():
    roi = _load(ROI_TRAIN, 'roi_train')
    crs = QgsCoordinateReferenceSystem(CRS_TARGET)
    tr = None
    if roi.crs() != crs:
        tr = QgsCoordinateTransform(roi.crs(), crs,
                                    QgsProject.instance().transformContext())
    engines = []
    for f in roi.getFeatures():
        g = f.geometry()
        if g is None or g.isEmpty():
            continue
        if tr:
            g.transform(tr)
        if ROI_BUFFER > 0:
            g = g.buffer(ROI_BUFFER, 8)
        eng = QgsGeometry.createGeometryEngine(g.constGet())
        eng.prepareGeometry()
        engines.append((g, eng))
    _log(f"Wykluczenie ROI treningowego: {len(engines)} poligonow "
         f"(+ bufor {ROI_BUFFER} m)")
    return engines

def _raster_extent():
    if os.path.exists(EXTENT_SHP):
        return _load(EXTENT_SHP, 'zasieg')
    r = QgsRasterLayer(RASTER, 'raster')
    if not r.isValid():
        raise RuntimeError(f"Nie mozna wczytac rastra: {RASTER}")
    ext = r.extent()
    ext_str = f"{ext.xMinimum()},{ext.xMaximum()},{ext.yMinimum()},{ext.yMaximum()}"
    lyr = _run('native:extenttolayer',
               {'INPUT': f"{ext_str} [{r.crs().authid()}]", 'OUTPUT': 'memory:'})
    _save(lyr, EXTENT_SHP)
    _log(f"Zasieg sceny: {ext.width()/1000:.1f} x {ext.height()/1000:.1f} km")
    return lyr

def _raster_origin():
    r = QgsRasterLayer(RASTER, 'raster')
    ext = r.extent()
    return ext.xMinimum(), ext.yMaximum()

def step1_prepare():
    print("\n=== ETAP 1: przygotowanie warstw BDOT ===\n")
    os.makedirs(OUT, exist_ok=True)
    extent_layer = _raster_extent()

    simple_layers = {
        'pole': (LAYERS['pole'], FIELD_FILTER),
        'las':  (LAYERS['las'],  None),
        'woda': (LAYERS['woda'], None),
    }
    for layer_name, (dest_path, filter_expr) in simple_layers.items():
        _log(f"[{layer_name}]")
        lyr = _load(dest_path, layer_name)
        lyr = _filter_layer(lyr, _add_version(filter_expr))
        _log(f"  po filtrze: {lyr.featureCount()} obiektow")
        lyr = _reproj(lyr)
        lyr = _clip(lyr, extent_layer)
        lyr = _dissolve(lyr)
        _save(lyr, _p(f'bdot_{layer_name}.shp'))
        _log(f"  -> bdot_{layer_name}.shp")

    _log("[zabudowa]")
    lyr = _load(LAYERS['zabudowa'], 'zabudowa')
    lyr = _filter_layer(lyr, _add_version(None))
    lyr = _reproj(lyr)
    lyr = _run('native:extractbyexpression',
               {'INPUT': lyr, 'EXPRESSION': f'"pow_m2" > {MIN_BUILDING_AREA}',
                'OUTPUT': 'memory:'})
    _log(f"  budynkow > {MIN_BUILDING_AREA} m2: {lyr.featureCount()}")
    lyr = _clip(lyr, extent_layer)
    lyr = _buffer(lyr, BUILDING_BUFFER)
    lyr = _run('native:multiparttosingleparts', {'INPUT': lyr, 'OUTPUT': 'memory:'})
    lyr = _run('native:extractbyexpression',
               {'INPUT': lyr, 'EXPRESSION': 'area($geometry) > 100', 'OUTPUT': 'memory:'})
    lyr = _dissolve(lyr)
    _save(lyr, _p('bdot_zabudowa.shp'))
    _log(f"  -> bdot_zabudowa.shp (bufor {BUILDING_BUFFER} m)")

    _log("[plaza]")
    water = _load(_p('bdot_woda.shp'), 'woda')
    strip = _buffer(water, BEACH_STRIP)
    lyr = _load(LAYERS['plaza'], 'plaza')
    lyr = _filter_layer(lyr, _add_version(None))
    lyr = _reproj(lyr)
    lyr = _clip(lyr, extent_layer)
    lyr = _clip(lyr, strip)
    lyr = _dissolve(lyr)
    _save(lyr, _p('bdot_plaza.shp'))
    _log(f"  -> bdot_plaza.shp (pas {BEACH_STRIP} m od wody)")

    _log("[droga]")
    lines_layer = _load(LAYERS['droga_l'], 'droga_l')
    lines_layer = _filter_layer(lines_layer, _add_version(ROAD_FILTER))
    _log(f"  odcinkow po filtrze klasy drogi: {lines_layer.featureCount()}")
    lines_layer = _reproj(lines_layer)
    lines_layer = _clip(lines_layer, extent_layer)
    lines_layer = _buffer(lines_layer, ROAD_BUFFER, styl_plaski=True)

    parts = [lines_layer]
    if os.path.exists(LAYERS['droga_a']):
        pol = _load(LAYERS['droga_a'], 'droga_a')
        pol = _filter_layer(pol, _add_version(None))
        pol = _reproj(pol)
        pol = _clip(pol, extent_layer)
        parts.append(pol)
        _log(f"  PTKM_A: {pol.featureCount()} obiektow")
    else:
        _log("  PTKM_A nie znaleziony, pomijam")

    lyr = _run('native:mergevectorlayers',
               {'LAYERS': parts, 'CRS': QgsCoordinateReferenceSystem(CRS_TARGET),
                'OUTPUT': 'memory:'})
    lyr = _dissolve(lyr)
    _save(lyr, _p('bdot_droga.shp'))
    _log(f"  -> bdot_droga.shp (bufor {ROAD_BUFFER} m od osi)")

    print("\nGOTOWE. Obejrzyj warstwy bdot_*.shp w QGIS przed dalszym krokiem.")
    print("Nastepny krok: MODE = 'pilot'\n")

def _sample_points(layer_name, n, out_path, exclude=None):
    source = _p(f'bdot_{layer_name}.shp')
    lyr = _load(source, layer_name)

    parts = []
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
            parts.append((part, eng, part.boundingBox(), a))
    if not parts:
        raise RuntimeError(f"{source}: brak poligonow do losowania")

    cum_wagi = list(accumulate(c[3] for c in parts))
    rng = random.Random(42)
    points = []
    attempts, limit = 0, max(20000, n * 500)
    while len(points) < n and attempts < limit:
        attempts += 1
        part, eng, bb, _a = rng.choices(parts, cum_weights=cum_wagi)[0]
        x = rng.uniform(bb.xMinimum(), bb.xMaximum())
        y = rng.uniform(bb.yMinimum(), bb.yMaximum())
        p = QgsPoint(x, y)
        if not eng.contains(p):
            continue
        if exclude and any(e.intersects(p) for _g, e in exclude):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < MIN_DIST ** 2 for px, py in points):
            continue
        points.append((x, y))

    pts = QgsVectorLayer(f'Point?crs={CRS_TARGET}&field=ref_7:integer',
                         layer_name, 'memory')
    feats = []
    for x, y in points:
        f = QgsFeature(pts.fields())
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        f['ref_7'] = CLASSES[layer_name]
        feats.append(f)
    pts.dataProvider().addFeatures(feats)
    pts.updateExtents()

    _save(pts, out_path)
    _log(f"[{layer_name}] wylosowano {pts.featureCount()} / {n} -> {os.path.basename(out_path)}")
    if pts.featureCount() < n:
        _log(f"  UWAGA: mniej niz zadano. Za malo miejsca przy MIN_DISTANCE={MIN_DIST} m.")
    return pts

def step2_pilot():
    print("\n=== ETAP 2: kalibracja pilotazowa (30 pkt/klase) ===\n")
    os.makedirs(_p('pilot'), exist_ok=True)
    for layer_name in ['droga', 'zabudowa', 'plaza', 'pole', 'las', 'woda']:
        _sample_points(layer_name, 30, _p(f'pilot/pilot_{layer_name}.shp'))

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

def step3_sample():
    print("\n=== ETAP 3: losowanie wlasciwe ===\n")
    missing = [k for k, v in POINT_COUNTS.items() if v is None]
    if missing:
        raise ValueError(
            f"Uzupelnij LICZBA_PUNKTOW dla: {missing}. Najpierw uruchom MODE='pilot'.")

    os.makedirs(_p('punkty'), exist_ok=True)
    exclude = _roi_exclusions()
    layers = []
    for layer_name, n in POINT_COUNTS.items():
        file_path = _p(f'punkty/punkty_{layer_name}.shp')
        _sample_points(layer_name, n, file_path, exclude=exclude)
        layers.append(file_path)

    merged = _run('native:mergevectorlayers', {
        'LAYERS': layers,
        'CRS': QgsCoordinateReferenceSystem(CRS_TARGET),
        'OUTPUT': 'memory:'})
    _save(merged, _p('punkty_do_weryfikacji.shp'))

    print(f"\nRAZEM: {merged.featureCount()} punktow")
    print(f"-> {_p('punkty_do_weryfikacji.shp')}")
    print("""
ETAP 4: WERYFIKACJA RĘCZNA
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

def step4_assist():
    print("\n=== ETAP 4a: wspomaganie weryfikacji (NDVI) ===\n")
    source = _p('punkty_do_weryfikacji.shp')
    if not os.path.exists(source):
        raise FileNotFoundError(f"Brak {source}. Najpierw MODE='sample'.")

    lyr = _load(source, 'punkty')
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

    CLASS_NAMES = {1: 'pole zaorane', 2: 'droga', 3: 'plaza', 4: 'las',
             5: 'zabudowa', 6: 'woda', 7: 'pole z rosl.'}

    def _suggestion(k, ndvi, ndwi):
        if ndvi is None:
            return None
        if ndwi is not None and ndwi > 0.10 and ndvi < 0.05:
            return 6
        if ndvi >= NDVI_GREEN_FIELD:
            return 4 if k == 4 else 7
        if ndvi <= NDVI_BARE_FIELD:
            if k in (1, 2, 3, 5):
                return 1 if k == 1 else k
            return None
        if k == 4:
            return 4
        return None

    feats, n_flagi, pole_1, pole_7, pole_np = [], 0, 0, 0, 0
    for f in lyr.getFeatures():
        pt = f.geometry().asPoint()
        value = prov.identify(pt, QgsRaster.IdentifyFormatValue).results()
        green = value.get(BAND_GREEN)
        red   = value.get(BAND_RED)
        nir   = value.get(BAND_NIR)
        ndvi = _idx(nir, red)
        ndwi = _idx(green, nir)

        k = f['ref_7']
        hint = _suggestion(k, ndvi, ndwi)
        if k == 1:
            if hint == 1:
                pole_1 += 1
            elif hint == 7:
                pole_7 += 1
            else:
                pole_np += 1

        if ndvi is None:
            flag = 'poza rastrem'
        elif hint is None:
            flag = 'spektralnie niepewne'
        elif k == 1 and hint in (1, 7):
            flag = ''
        elif hint != k:
            flag = f'wyglada jak {CLASS_NAMES[hint]}'
        else:
            flag = ''
        if flag:
            n_flagi += 1

        nf = QgsFeature(out.fields())
        nf.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(pt)))
        nf['ref_7'] = k
        nf['ndvi'] = round(ndvi, 4) if ndvi is not None else None
        nf['ndwi'] = round(ndwi, 4) if ndwi is not None else None
        nf['sugestia'] = hint
        nf['flaga'] = flag
        feats.append(nf)

    out.dataProvider().addFeatures(feats)
    out.updateExtents()
    _save(out, _p('punkty_do_weryfikacji_assist.shp'))

    print(f"  Punktow: {len(feats)}, oflagowanych: {n_flagi}")
    print(f"  Pole, sugestie: {pole_1} x zaorane (1), {pole_7} x roslinnosc (7), "
          f"{pole_np} niepewnych")
    print(f"""
-> punkty_do_weryfikacji_assist.shp

SPOSÓB UŻYCIA (etap 4 pozostaje ręczny):
  1. Warstwę wczytuje się na tle ortofoto.
  2. Styl warstwy: kategoryzacja po polu 'flaga', oflagowane na czerwono.
  3. Weryfikacja przebiega punkt po punkcie jak dotąd; przy polach kolumna
     'sugestia' podpowiada podział 1/7, potwierdzany lub korygowany
     wyłącznie według ortofoto.
  4. Wynik zapisuje się jako referencja_zweryfikowana.shp; kolumny
     ndvi/ndwi/sugestia/flaga można usunąć albo zostawić, nie przeszkadzają.
""")

def step5_polygons():
    print("\n=== ETAP 5: punkty -> kwadraty 8x8 m ===\n")
    source = _p('referencja_zweryfikowana3.shp')
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"Brak {source}. Najpierw wykonaj ETAP 4 (weryfikacja na ortofoto).")

    lyr = _load(source, 'ref')
    x0, y0 = _raster_origin()
    _log(f"Origin rastra: X0={x0}, Y0={y0}")

    expr = (f"make_point("
            f"floor((x($geometry)-{x0})/{PIXEL})*{PIXEL} + {x0} + {PIXEL/2}, "
            f"floor((y($geometry)-{y0})/{PIXEL})*{PIXEL} + {y0} + {PIXEL/2})")
    snapped = _run('native:geometrybyexpression', {
        'INPUT': lyr, 'OUTPUT_GEOMETRY': 2,
        'WITH_Z': False, 'WITH_M': False,
        'EXPRESSION': expr, 'OUTPUT': 'memory:'})
    _log("Punkty przesuniete do srodkow pikseli")

    squares = _run('native:buffer', {
        'INPUT': snapped, 'DISTANCE': PIXEL * 0.4,
        'SEGMENTS': 1, 'END_CAP_STYLE': 2,
        'JOIN_STYLE': 2, 'MITER_LIMIT': 2,
        'DISSOLVE': False, 'OUTPUT': 'memory:'})

    _save(squares, _p('referencja_FINAL.shp'))
    _log(f"-> referencja_FINAL.shp ({squares.featureCount()} poligonow)")

    eps = 1e-6
    bad = 0
    for f in squares.getFeatures():
        bb = f.geometry().boundingBox()
        kol1 = int((bb.xMinimum() - x0 + eps) // PIXEL)
        kol2 = int((bb.xMaximum() - x0 - eps) // PIXEL)
        w1 = int((y0 - bb.yMaximum() + eps) // PIXEL)
        w2 = int((y0 - bb.yMinimum() - eps) // PIXEL)
        if kol1 != kol2 or w1 != w2:
            bad += 1

    print("\n  Kontrola wpasowania kwadratow w piksele:")
    if bad:
        print(f"    >>> BLAD: {bad} kwadratow przekracza granice piksela <<<")
        print("    Sprawdz origin rastra i wartosc PIXEL w CONFIG.")
    else:
        print(f"    OK, wszystkie {squares.featureCount()} kwadratow "
              f"lezy w calosci w jednym pikselu.")

    print("\nNastepny krok: MODE = 'check'\n")

def step6_check():
    print("\n=== ETAP 6: kontrola jakosci ===\n")
    ref = _load(_p('referencja_FINAL.shp'), 'ref')

    print("[Test 1] Nakladanie na ROI treningowe")
    if os.path.exists(ROI_TRAIN):
        train = _load(ROI_TRAIN, 'train')
        train = _reproj(train)
        inter = _run('native:intersection',
                     {'INPUT': ref, 'OVERLAY': train,
                      'INPUT_FIELDS': [], 'OVERLAY_FIELDS': [], 'OUTPUT': 'memory:'})
        n = inter.featureCount()
        if n == 0:
            print("  OK, brak nakladania. Referencja niezalezna od treningu.\n")
        else:
            print(f"  >>> UWAGA: {n} poligonow referencyjnych nakłada sie na ROI! <<<")
            _save(inter, _p('KOLIZJE_z_treningiem.shp'))
            print(f"  Zapisano: KOLIZJE_z_treningiem.shp, usun te punkty.\n")
    else:
        print(f"  Pominieto, brak pliku {ROI_TRAIN}\n")

    print("[Test 2] Liczebnosc klas")
    CLASS_NAMES = {1: 'pole zaorane', 2: 'droga', 3: 'plaza', 4: 'las',
             5: 'zabudowa', 6: 'woda', 7: 'pole z roslinnoscia'}
    counts = {}
    for f in ref.getFeatures():
        k = f['ref_7']
        counts[k] = counts.get(k, 0) + 1

    ok = True
    for k in sorted(counts):
        flag = "OK" if counts[k] >= 75 else ">>> < 75 <<<"
        if counts[k] < 75:
            ok = False
        print(f"  {k} {CLASS_NAMES.get(k, '?'):22s} {counts[k]:4d}   {flag}")
    print(f"  {'RAZEM':26s} {sum(counts.values()):4d}")
    if not ok:
        print("\n  Dolosuj punkty dla klas ponizej 75 i zweryfikuj je na ortofoto.")

    print("\n[Test 3] Typ pola ref_7")
    idx = ref.fields().indexOf('ref_7')
    field_type = ref.fields().at(idx).typeName()
    print(f"  ref_7: {field_type}   {'OK' if 'nt' in field_type else '>>> musi byc Integer <<<'}")

    print("""
Dalszy krok: warianty zestawu klas powstają ze zweryfikowanej warstwy
referencja_FINAL.shp skryptem badania/dane/generuj_warianty_referencja.py,
który wytwarza 8 plików referencja_<wariant>.shp z polem 'klasa'.
""")

MODES = {
    'prepare':  step1_prepare,
    'pilot':    step2_pilot,
    'sample':   step3_sample,
    'assist':   step4_assist,
    'polygons': step5_polygons,
    'check':    step6_check,
}

if MODE not in MODES:
    raise ValueError(f"Nieznany MODE: {MODE}. Dostepne: {list(MODES)}")

os.makedirs(OUT, exist_ok=True)
MODES[MODE]()
