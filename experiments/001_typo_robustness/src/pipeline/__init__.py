"""The pipeline sub-package: the idempotent generation runner and the
config-driven experiment orchestrator.

Re-exports the full public surface of both modules:

    from pipeline import ExperimentConfiguration, run_experiment
    from pipeline import GenerationRequest, DeterministicDummyEngine
    from pipeline import run_shard, load_generation_rows
"""

from pipeline.runner import (
    SCHEMA_VERSION,
    GenerationRequest,
    DeterministicDummyEngine,
    ShardManifest,
    chat_exemplar_turns_for_family,
    deterministic_row_id,
    run_shard,
    load_generation_rows,
)

from pipeline.experiment import (
    PerturbationCondition,
    ExperimentConfiguration,
    load_task_items,
    build_requests,
    required_context_length,
    run_experiment,
)
