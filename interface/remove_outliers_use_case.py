cfg = __import__(str(__name__).split(".")[0] + ".core.config", fromlist=[""])


def remove_outliers_drawing_roi():
    if cfg.temporary_roi is None:
        cfg.mx.msg_war_4()
        return

    # drawn ROI: the reference scope is forced to a single area
    def _callback(pipeline, use_majority_voting, vote_threshold, _scope):
        return cfg.remove_outliers.remove_outliers_drawing_roi(
            pipeline, use_majority_voting, vote_threshold
        )

    cfg.remove_outliers_dialog.PipelineDialog(
        _callback, scope_options=("roi",)
    ).exec_()


def remove_outliers_selected_signatures():
    if cfg.scp_training is None or cfg.scp_training.signature_catalog is None:
        cfg.mx.msg_war_outliers_no_training()
        return

    signatures_selected, macroclass_ids = (
        cfg.scp_training.get_highlighted_selection_types()
    )

    # only macroclass nodes are selected: the scope is forced to macroclass
    if macroclass_ids and not signatures_selected:
        def _mc_callback(pipeline, use_majority_voting, vote_threshold,
                         _scope):
            return cfg.remove_outliers.remove_outliers_macroclasses(
                macroclass_ids, pipeline, use_majority_voting, vote_threshold
            )

        cfg.remove_outliers_dialog.PipelineDialog(
            _mc_callback, scope_options=("macroclass",)
        ).exec_()
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

    def _callback(pipeline, use_majority_voting, vote_threshold, scope):
        return cfg.remove_outliers.remove_outliers_selected_signatures(
            sig_ids, pipeline, use_majority_voting, vote_threshold, scope
        )

    cfg.remove_outliers_dialog.PipelineDialog(
        _callback, scope_options=("roi", "class", "macroclass")
    ).exec_()


def remove_outliers_all_signatures():
    if cfg.scp_training is None or cfg.scp_training.signature_catalog is None:
        cfg.mx.msg_war_outliers_no_training()
        return

    def _callback(pipeline, use_majority_voting, vote_threshold, scope):
        return cfg.remove_outliers.remove_outliers_all_signatures(
            pipeline, use_majority_voting, vote_threshold, scope
        )

    cfg.remove_outliers_dialog.PipelineDialog(
        _callback, scope_options=("roi", "class", "macroclass")
    ).exec_()


def get_pipeline_config():
    dialog = cfg.remove_outliers_dialog.PipelineDialog(
        signature_mode=True, configure_only=True,
        scope_options=("roi", "class", "macroclass"),
    )
    if not dialog.exec_():
        return None

    pipeline = dialog.get_pipeline()
    if not pipeline:
        return None

    return (
        pipeline, dialog.get_voting(), dialog.get_vote_threshold(),
        dialog.get_scope(),
    )
