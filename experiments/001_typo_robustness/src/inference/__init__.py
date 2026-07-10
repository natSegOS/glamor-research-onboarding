"""The inference sub-package: the model roster and the generation engines.

Re-exports the full public surface of both modules:

    from inference import MODEL_ROSTER, get_model_specification
    from inference import VllmEngine
"""

from inference.roster import (
    REVISION_PLACEHOLDER,
    ModelSpecification,
    MODEL_ROSTER,
    get_model_specification,
    resolve_current_revision,
    assert_revisions_pinned,
)

from inference.engines import VllmEngine
