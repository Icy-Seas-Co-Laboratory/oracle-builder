"""Dependency-light, versioned Oracle data and artifact contracts."""

from oracle_data_contracts.datasets import SCHEMA_NAME, SCHEMA_VERSION
from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WORK_UNIT_SCHEMA_NAME, WORK_UNIT_SCHEMA_VERSION, WorkUnit, WorkUnitError

__all__ = [
    "SCHEMA_NAME", "SCHEMA_VERSION", "ArtifactRef", "WORK_UNIT_SCHEMA_NAME",
    "WORK_UNIT_SCHEMA_VERSION", "WorkUnit", "WorkUnitError",
]
