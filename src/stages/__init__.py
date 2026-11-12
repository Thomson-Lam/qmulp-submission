"""The stage drivers the notebooks and ``scripts/run_all.py`` call.

One module per stage with a ``run(dataset_name, config)`` entry point that
reads the frozen artifacts, writes the stage's intermediates under
``outputs/notebook_pipeline/<stage>/`` and its result tables under
``outputs/results/<dataset>/``, and ends by writing the stage manifest. The notebooks
are thin: each calls one of these and displays what it returns.
"""
