# Raster L1C

Tutaj należy umieścić plik **`L1C_przyciety.tif`**, czyli scenę Sentinel-2
poziomu L1C (Top-Of-Atmosphere, bez korekcji atmosferycznej) przyciętą do obszaru badań.

Plik nie jest przechowywany w repozytorium ze względu na rozmiar. Bez niego
skrypty badawcze zgłoszą brak rastra i przerwą działanie.

## Scena źródłowa

    S2B_MSIL1C_20240814T094549_N0511_R079_T34UCF_20240814T114912.SAFE

Satelita Sentinel-2B, kafelek T34UCF, orbita R079, akwizycja 14 sierpnia 2024
o godzinie 09:45 UTC. Produkt pobiera się z Copernicus Data Space Ecosystem.

## Jak przyciąć

Granicę obszaru badań zawiera maska `badania/dane_testowe/maska/maska2.shp`.
Po złożeniu trzynastu pasm sceny w jeden raster wielopasmowy wystarczy:

    gdalwarp -cutline badania/dane_testowe/maska/maska2.shp -crop_to_cutline \
      pelna_scena_L1C.tif badania/dane_testowe/L1C/L1C_przyciety.tif

## Oczekiwane parametry

    rozmiar        5774 x 3720 pikseli
    pasma          13
    typ danych     UInt16
    rozdzielczosc  10 m
    uklad          EPSG:32634 (WGS 84 / UTM strefa 34N)

Wartości pikseli należy pozostawić w skali natywnej produktu. Skrypty badawcze
pracują na wartościach surowych i nie wykonują żadnego przeliczenia, wobec
czego przycinanie nie może zmieniać typu danych ani skalować wartości.
