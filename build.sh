#!/bin/bash
set -e

QGIS_PYTHON="/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3.9"
LRELEASE="/opt/homebrew/opt/qt/bin/lrelease"

if [ ! -f "$QGIS_PYTHON" ]; then
    echo "ERROR: QGIS Python not found: $QGIS_PYTHON"
    exit 1
fi

if [ ! -f "$LRELEASE" ]; then
    echo "WARNING: lrelease not found: $LRELEASE"
    echo "Translations will not be compiled. Install Qt: brew install qt"
    SKIP_TRANSLATIONS=1
fi

echo "Building resources_rc.py..."
"$QGIS_PYTHON" -m PyQt5.pyrcc_main ui/resources.qrc -o ui/resources_rc.py
echo "OK - ui/resources_rc.py"

echo "Building UI files..."

compile_ui() {
    local ui_file="$1"
    local py_file="${ui_file%.ui}.py"
    "$QGIS_PYTHON" -m PyQt5.uic.pyuic "$ui_file" -o "$py_file" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERR $(basename "$ui_file")"
        return
    fi
    sed -i '' 's/^import resources_rc$/from . import resources_rc/' "$py_file"
    echo "OK - $(basename "$ui_file")"
}

compile_ui "ui/ui_semiautomaticclassificationplugin.ui"
compile_ui "ui/ui_semiautomaticclassificationplugin_dock_class.ui"
compile_ui "ui/ui_semiautomaticclassificationplugin_scatter_plot.ui"
compile_ui "ui/ui_semiautomaticclassificationplugin_signature_plot.ui"
compile_ui "ui/ui_semiautomaticclassificationplugin_widget.ui"

if [ -z "$SKIP_TRANSLATIONS" ]; then
    echo "Building translations..."
    for ts_file in i18n/*.ts; do
        "$LRELEASE" "$ts_file" -qm "${ts_file%.ts}.qm" 2>/dev/null \
            && echo "OK - $(basename "$ts_file")" \
            || echo "ERR - $(basename "$ts_file")"
    done
else
    echo "Translations skipped (lrelease not found)."
fi

echo ""
echo "DONE"
