#!/bin/bash
# Kompiluje pliki .ui do .py dla pluginu SCP.
# Uruchom z katalogu głównego pluginu: bash compile_ui.sh

QGIS_PYTHON="/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3.9"
UI_DIR="$(dirname "$0")/ui"

compile() {
    local ui_file="$1"
    local py_file="${ui_file%.ui}.py"
    "$QGIS_PYTHON" -m PyQt5.uic.pyuic "$ui_file" -o "$py_file" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERR $(basename "$ui_file")"
        return
    fi
    # pyuic5 generuje "import resources_rc" zamiast "from . import resources_rc"
    sed -i '' 's/^import resources_rc$/from . import resources_rc/' "$py_file"
    echo "OK  $(basename "$ui_file")"
}

compile "$UI_DIR/ui_semiautomaticclassificationplugin.ui"
compile "$UI_DIR/ui_semiautomaticclassificationplugin_dock_class.ui"
compile "$UI_DIR/ui_semiautomaticclassificationplugin_scatter_plot.ui"
compile "$UI_DIR/ui_semiautomaticclassificationplugin_signature_plot.ui"
compile "$UI_DIR/ui_semiautomaticclassificationplugin_widget.ui"
