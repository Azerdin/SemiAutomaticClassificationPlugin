cfg = __import__(str(__name__).split(".")[0] + ".core.config", fromlist=[""])


def remove_outliers_drawing_roi():
    cfg.remove_outliers_dialog.PipelineDialog(
        cfg.remove_outliers.remove_outliers_drawing_roi
    ).exec_()


def remove_outliers_drawed_roi():
    raise ValueError("Method not implemented yet")


def remove_outliers_signature():
    raise ValueError("Method not implemented yet")
