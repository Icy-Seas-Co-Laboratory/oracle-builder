"""Portable, location-free references to durable Oracle artifacts.

An :class:`ArtifactRef` deliberately contains no filesystem path or storage
credential.  It is safe to put in a work unit that may execute on another
machine; a store is responsible for resolving the reference locally.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


def _token(value: str, field: str) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ValueError(
            f"{field} must be a non-path token containing only letters, numbers, '.', '_' or '-'"
        )
    return value


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Stable identity for an immutable artifact revision.

    ``kind`` names the artifact namespace (for example ``dataset`` or
    ``model_run``), while ``artifact_id`` is the producer-owned identity.
    ``revision`` is optional for existing artifacts that have no separate
    revision identifier.  The optional fingerprint pins the exact bytes or
    semantic content expected by a consumer.
    """

    kind: str
    artifact_id: str
    revision: str | None = None
    fingerprint_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _token(self.kind, "kind"))
        object.__setattr__(self, "artifact_id", _token(self.artifact_id, "artifact_id"))
        if self.revision is not None:
            object.__setattr__(self, "revision", _token(self.revision, "revision"))
        if self.fingerprint_sha256 is not None:
            fingerprint = self.fingerprint_sha256.lower()
            if not _FINGERPRINT.fullmatch(fingerprint):
                raise ValueError("fingerprint_sha256 must be a 64-character lowercase SHA-256 hex digest")
            object.__setattr__(self, "fingerprint_sha256", fingerprint)

    @property
    def uri(self) -> str:
        """Compact, stable text form: ``kind:id@revision#fingerprint``."""
        value = f"{self.kind}:{self.artifact_id}"
        if self.revision is not None:
            value += f"@{self.revision}"
        if self.fingerprint_sha256 is not None:
            value += f"#{self.fingerprint_sha256}"
        return value

    def __str__(self) -> str:
        return self.uri

    def to_dict(self) -> dict[str, str]:
        value: dict[str, str] = {"kind": self.kind, "artifact_id": self.artifact_id}
        if self.revision is not None:
            value["revision"] = self.revision
        if self.fingerprint_sha256 is not None:
            value["fingerprint_sha256"] = self.fingerprint_sha256
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            kind=str(value["kind"]),
            artifact_id=str(value["artifact_id"]),
            revision=str(value["revision"]) if value.get("revision") is not None else None,
            fingerprint_sha256=(
                str(value["fingerprint_sha256"])
                if value.get("fingerprint_sha256") is not None
                else None
            ),
        )

    @classmethod
    def parse(cls, value: str) -> "ArtifactRef":
        """Parse an :attr:`uri`; paths and credentials are intentionally invalid."""
        if not isinstance(value, str) or value.count(":") != 1:
            raise ValueError("artifact reference must use 'kind:artifact_id[@revision][#fingerprint]' form")
        kind, identity = value.split(":", 1)
        if identity.count("#") > 1:
            raise ValueError("artifact reference has more than one fingerprint separator")
        core, separator, fingerprint = identity.partition("#")
        if core.count("@") > 1:
            raise ValueError("artifact reference has more than one revision separator")
        artifact_id, revision_separator, revision = core.partition("@")
        return cls(
            kind=kind,
            artifact_id=artifact_id,
            revision=revision if revision_separator else None,
            fingerprint_sha256=fingerprint if separator else None,
        )
