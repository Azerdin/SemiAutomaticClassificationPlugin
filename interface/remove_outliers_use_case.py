cfg = __import__(str(__name__).split(".")[0] + ".core.config", fromlist=[""])


def remove_outliers_drawing_roi():
    cfg.remove_outliers_dialog.PipelineDialog(
        cfg.remove_outliers.remove_outliers_drawing_roi
    ).exec_()


def remove_outliers_selected_signatures():
    if cfg.scp_training is None or cfg.scp_training.signature_catalog is None:
        cfg.mx.msg_war_outliers_no_training()
        return

    sig_ids = cfg.scp_training.get_highlighted_ids()

    catalog = cfg.scp_training.signature_catalog
    tbl = catalog.table
    sig_ids = [
        s
        for s in sig_ids
        if (tbl["signature_id"] == s).any()
        and tbl[tbl["signature_id"] == s]["geometry"][0] == 1
    ]

    if not sig_ids:
        cfg.mx.msg_war_outliers_no_selection()
        return

    def _callback(pipeline, voting, threshold):
        return cfg.remove_outliers.remove_outliers_selected_signatures(
            sig_ids, pipeline, voting, threshold
        )

    cfg.remove_outliers_dialog.PipelineDialog(_callback).exec_()


def remove_outliers_all_signatures():
    if cfg.scp_training is None or cfg.scp_training.signature_catalog is None:
        cfg.mx.msg_war_outliers_no_training()
        return

    cfg.remove_outliers_dialog.PipelineDialog(
        cfg.remove_outliers.remove_outliers_all_signatures
    ).exec_()


def get_pipeline_config():
    captured = {}

    def _capture(pipeline, use_majority_voting, vote_threshold):
        captured["pipeline"] = pipeline
        captured["voting"] = use_majority_voting
        captured["threshold"] = vote_threshold
        return None

    cfg.remove_outliers_dialog.PipelineDialog(_capture, signature_mode=True).exec_()

    if "pipeline" not in captured:
        return None

    return captured["pipeline"], captured["voting"], captured["threshold"]
