"""The analysis sub-package: paired statistics, per-cell result summarization,
and the three-regime human audit.

Import order matters here: results.py imports statistics, so statistics must
exist on the package namespace before results is loaded.

Usage:

    from analysis import statistics, results, audit
    from analysis import mcnemar_test, MatchedPair, AuditReport
"""

from analysis import statistics     # must be first (results.py imports it)
from analysis import results
from analysis import audit

# Re-export the full public surface of all three modules so callers can import
# directly from the sub-package instead of drilling into the individual files.

from analysis.statistics import (
    PairedContingencyTable,
    build_paired_table,
    McNemarResult,
    mcnemar_test,
    mcnemar_sample_size,
    audit_sample_size,
    ConfidenceInterval,
    bootstrap_confidence_interval_paired,
    summarize_cell,
)

from analysis.results import (
    CELL_DIMENSION_KEYS,
    MatchedPair,
    join_matched_pairs,
    group_pairs_into_cells,
    summarize_all_cells,
    write_cell_table,
)

from analysis.audit import (
    KAPPA_GATE,
    AuditRating,
    ItemAuditOutcome,
    fleiss_kappa,
    stratified_sample,
    resolve_item,
    AuditReport,
    audit_report,
)
