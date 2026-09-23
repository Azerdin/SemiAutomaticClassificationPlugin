# Notice of modification

This is a modified version of the Semi-Automatic Classification Plugin (SCP).

SCP is Copyright (C) 2012-2024 by Luca Congedo and is distributed under the
GNU General Public License, either version 3 of the License, or (at your
option) any later version. The full text of the license is in `COPYING`.

Modifications in this fork are Copyright (C) 2026 by Krzysztof Tyszko and are
distributed under the same license.

Upstream project: https://github.com/semiautomaticgit/SemiAutomaticClassificationPlugin

## What was modified, and when

Modified between 2025-12-24 and 2026-09-22, based on upstream version 8.5.0.

Added an optional tool that detects and removes outlying pixels from training
ROIs, and fixed a defect that corrupted the output classification raster.

### New files

    core/outlier_filters.py                detection methods, pipelines, voting
    core/outlier_catalog.py                applying the result to the catalog
    interface/remove_outliers.py           orchestration inside QGIS
    interface/remove_outliers_dialog.py    pipeline builder, report, chart
    interface/remove_outliers_use_case.py  entry points
    tests/test_outlier_filters.py          unit tests
    ui/icons/semiautomaticclassificationplugin_remove_outliers_*.svg

### Modified files

    core/messages.py                       new messages
    core/util_gdal.py                      wrappers for clipping and
                                           polygonising ROI rasters
    core/util_qgis.py                      helper used by the tool
    interface/classification_tab.py        fix of the classification defect,
                                           checkbox for cleaning before a run
    interface/input_interface.py           signal connections
    interface/scp_dock.py                  context menu and toolbar entries
    semiautomaticclassificationplugin.py   registration of the new actions
    ui/resources.qrc                       three new icons
    ui/resources_rc.py                     regenerated resource file
    ui/ui_*.ui, ui/ui_*.py                 interface definitions and the
                                           files regenerated from them
    i18n/*.ts, i18n/*.qm                   strings of the new tool
    metadata.txt                           marking of the modified version

The fix of the classification defect was submitted to the upstream project,
accepted and merged there.
