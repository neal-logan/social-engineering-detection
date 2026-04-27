"""Analysis module for the social-engineering detection pipeline.

Public entry points:
    schema           — column / key constants
    loading          — load + validate detection metadata + conversations
    preliminaries    — build conversation-level and turn-level tables
    query            — ibis aggregate queries on those tables
    roc              — ROC / AUC computation for the four scenarios
    figures          — histograms, heatmaps, Sankey, ROC curves + save helpers
"""

from . import figures, loading, preliminaries, query, roc, schema  # noqa: F401
