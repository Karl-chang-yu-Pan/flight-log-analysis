"""Per-file source-fact disk cache (Layer 1).

Cache the profiler's single-file extraction so downstream analyses don't
re-parse the same PX4 file for every question or every flight. Keyed on
``(source_hash, file_path)`` where ``source_hash`` is the git commit
SHA of the source tree at extraction time.

Cross-file joins — pointer-output routing, recursive helper resolution —
are NOT part of Layer 1; they are re-run by the discovery loop after
loading each involved file's Layer 1 entry.

**Cross-hash reuse**: entries under different git hashes coexist. On a
cache miss for the current ``source_hash``, :func:`get_source_facts_for_file`
walks other cached hashes and asks git whether the file's contents changed
between them. If not, the older entry is reused in place. When we
rebuild, the new entry is stored under the current ``source_hash`` — old
entries stay put so historical rollbacks don't trigger a rebuild.

Payload shape mirrors what
:meth:`MechanismSourceProfiler.extract_*_from_source([file])` produces
when invoked with a single-file list. See ``memory/dag_design.md`` for
where Layer 1 fits among Layers 2 (unresolved DAG per terminal), 3
(flight-annotated DAG), and 4 (question → terminal mapping).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from flight_log_agent.px4.mechanism_source_profiler import (
    BranchConditionRef,
    FieldRef,
    FunctionCallRef,
    HelperExpressionRef,
    MechanismSourceProfiler,
    ParameterPredicateRef,
    ParameterRef,
    SourceAssignmentRef,
    SourceCallableRef,
    SourceClassRef,
    SourceIncludeRef,
    SourceMemberRef,
    TopicRef,
)


class SourceFileFacts(BaseModel):
    """Complete single-file extraction cached at Layer 1.

    Each list holds the facts the profiler extracts when invoked with
    ``[file]`` as its file list — no cross-file joins. The Layer 1
    cache preserves these primary facts so a re-run doesn't reparse the
    same file.
    """

    file: str
    source_hash: str
    published_topics: List[TopicRef] = Field(default_factory=list)
    subscribed_topics: List[TopicRef] = Field(default_factory=list)
    unknown_direction_topics: List[TopicRef] = Field(default_factory=list)
    referenced_parameters: List[ParameterRef] = Field(default_factory=list)
    assigned_fields: List[FieldRef] = Field(default_factory=list)
    read_fields: List[FieldRef] = Field(default_factory=list)
    source_assignments: List[SourceAssignmentRef] = Field(default_factory=list)
    function_calls: List[FunctionCallRef] = Field(default_factory=list)
    helper_expressions: List[HelperExpressionRef] = Field(default_factory=list)
    branch_conditions: List[BranchConditionRef] = Field(default_factory=list)
    parameter_predicates: List[ParameterPredicateRef] = Field(default_factory=list)
    classes: List[SourceClassRef] = Field(default_factory=list)
    members: List[SourceMemberRef] = Field(default_factory=list)
    callables: List[SourceCallableRef] = Field(default_factory=list)
    includes: List[SourceIncludeRef] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


_FS_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@lru_cache(maxsize=1)
def extractor_fingerprint() -> str:
    """Content fingerprint of the extraction code.

    Hashes the source bytes of the profiler module and every first-party
    module it (transitively) references, so ANY edit — committed or not —
    yields a new fingerprint and Layer 1 entries produced by older code
    become unreachable. No manually-bumped version constant: the cache
    keys on what the code IS, not on what someone remembered to label it.
    """
    package_prefix = __name__.split(".", 1)[0]
    root = sys.modules[MechanismSourceProfiler.__module__]
    seen: set[str] = set()
    queue: list[str] = [root.__name__]
    files: set[Path] = set()
    while queue:
        name = queue.pop()
        if name in seen or not name.startswith(package_prefix):
            continue
        seen.add(name)
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if module is None or not module_file:
            continue
        files.add(Path(module_file))
        for attribute in vars(module).values():
            if isinstance(attribute, ModuleType):
                queue.append(attribute.__name__)
            else:
                dependency = getattr(attribute, "__module__", None)
                if isinstance(dependency, str):
                    queue.append(dependency)
    digest = hashlib.sha256()
    for path in sorted(files):
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(str(path).encode("utf-8"))
    return digest.hexdigest()[:16]


_PRUNED_ROOTS: set[tuple[str, str]] = set()


def _prune_stale_fingerprints(cache_root: Path, current: str) -> None:
    """Delete Layer 1 trees written by other extractor versions.

    Entries under a stale fingerprint are fully regenerable garbage — no
    migration is possible (the old output simply lacks whatever the new
    code extracts), so purge-and-lazily-rebuild is the semantics. Runs
    once per (cache_root, fingerprint) per process.
    """
    key = (str(cache_root), current)
    if key in _PRUNED_ROOTS:
        return
    _PRUNED_ROOTS.add(key)
    source_dir = Path(cache_root) / "source"
    if not source_dir.is_dir():
        return
    for entry in source_dir.iterdir():
        if entry.is_dir() and entry.name != current:
            shutil.rmtree(entry, ignore_errors=True)


def _file_path_slug(file_path: str) -> str:
    """Filesystem-safe slug for a source file path.

    ``src/modules/navigator/rtl.cpp`` → ``src__modules__navigator__rtl.cpp``.
    """
    slug = file_path.strip("/").replace("/", "__").replace("\\", "__")
    slug = _FS_SANITIZE_RE.sub("_", slug)
    return slug or "unknown"


def layer1_cache_path(cache_root: Path, source_hash: str, file_path: str) -> Path:
    """Layer 1 path:
    ``{cache_root}/source/{extractor_fp}/{source_hash}/{file_slug}.json``.

    The extractor fingerprint level makes entries from older extraction
    code unreachable the moment the code changes.
    """
    return (
        Path(cache_root)
        / "source"
        / extractor_fingerprint()
        / source_hash
        / f"{_file_path_slug(file_path)}.json"
    )


def write_source_facts(facts: SourceFileFacts, path: Path) -> None:
    """Serialize ``facts`` to ``path`` atomically.

    Writes to ``path.tmp`` and renames, so a partial write cannot corrupt
    an existing cache entry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(facts.model_dump_json(), encoding="utf-8")
    tmp.replace(path)


def read_source_facts(path: Path) -> Optional[SourceFileFacts]:
    """Deserialize ``SourceFileFacts`` from ``path``.

    Returns ``None`` on missing file or unparseable payload. Callers
    treat that as a cache miss.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        return SourceFileFacts.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def get_source_facts_for_file(
    cache_root: Union[str, Path],
    file_path: str,
    source_hash: str,
    *,
    source_root: Optional[Union[str, Path]] = None,
    git_path: str = "git",
) -> Optional[SourceFileFacts]:
    """Look up cached facts for ``file_path`` at ``source_hash``.

    Tries a direct-hit at ``{cache_root}/source/{source_hash}/{file_slug}.json``
    first. On miss, walks other cached hashes and asks git whether
    ``file_path`` is unchanged between that hash and ``source_hash``.
    An unchanged entry is reused in place (no copy, no overwrite).

    Returns ``None`` when no reusable entry exists; the caller should
    build a fresh entry via :func:`extract_facts_for_file` and write it
    under the current ``source_hash``.

    ``source_root`` (path to the git working tree) is required for
    cross-hash reuse. Without it, only direct-hit is attempted.
    """
    cache_root = Path(cache_root)
    direct = read_source_facts(layer1_cache_path(cache_root, source_hash, file_path))
    if direct is not None:
        return direct

    if source_root is None:
        return None

    source_dir = cache_root / "source" / extractor_fingerprint()
    if not source_dir.is_dir():
        return None

    for other_dir in source_dir.iterdir():
        if not other_dir.is_dir() or other_dir.name == source_hash:
            continue
        other_hash = other_dir.name
        candidate_path = layer1_cache_path(cache_root, other_hash, file_path)
        if not candidate_path.exists():
            continue
        if not _file_unchanged_between(source_root, other_hash, source_hash, file_path, git_path):
            continue
        return read_source_facts(candidate_path)

    return None


def _file_unchanged_between(
    source_root: Union[str, Path],
    hash_a: str,
    hash_b: str,
    file_path: str,
    git_path: str,
) -> bool:
    """True iff ``file_path`` is byte-identical between ``hash_a`` and ``hash_b``.

    Uses ``git diff --quiet`` — exit 0 means no diff, exit 1 means diff,
    anything else is a git error and we conservatively treat as "changed"
    so the caller rebuilds.
    """
    try:
        result = subprocess.run(
            [git_path, "diff", "--quiet", hash_a, hash_b, "--", file_path],
            cwd=str(source_root),
            capture_output=True,
        )
    except (FileNotFoundError, OSError):
        return False
    return result.returncode == 0


def get_or_extract_facts(
    profiler: MechanismSourceProfiler,
    cache_root: Union[str, Path],
    file_path: str,
    source_hash: str,
    *,
    source_root: Optional[Union[str, Path]] = None,
    git_path: str = "git",
) -> SourceFileFacts:
    """Layer 1 read-path: cache hit if available, extract and write on miss.

    First calls :func:`get_source_facts_for_file`, which tries the direct
    hit under ``source_hash`` and falls back to git-diff-based cross-hash
    reuse. On genuine miss, runs :func:`extract_facts_for_file` and
    writes the result to Layer 1 for next time.

    **Cross-file joins are NOT covered.** ``SourceFileFacts.source_assignments``
    contains only what falls out of ``extract_source_assignments_from_source([file])``
    for a single-file input — pointer-output routing across caller/callee
    files, recursive helper resolution across files, and any other
    cross-file pass must be reapplied by the caller on the union of loaded
    Layer 1 entries. The discovery loop is the intended owner of that
    reassembly.
    """
    _prune_stale_fingerprints(Path(cache_root), extractor_fingerprint())
    facts = get_source_facts_for_file(
        cache_root,
        file_path,
        source_hash,
        source_root=source_root,
        git_path=git_path,
    )
    if facts is not None:
        return facts
    facts = extract_facts_for_file(profiler, file_path, source_hash)
    write_source_facts(
        facts,
        layer1_cache_path(cache_root, source_hash, file_path),
    )
    return facts


def extract_facts_for_file(
    profiler: MechanismSourceProfiler,
    file_path: str,
    source_hash: str,
) -> SourceFileFacts:
    """Run every per-file profiler extractor and return a bundled result.

    Invokes each ``extract_*_from_source`` method with the single-file
    list ``[file_path]``. The profiler's cross-file join steps
    (pointer-output routing, recursive helper resolution) may still fire
    but only against the single input file, so they collapse to the
    trivial single-file case — cross-file joins are the discovery
    loop's responsibility to reassemble across Layer 1 entries.
    """
    files = [file_path]

    def from_exact_file(items):
        return [item for item in items if str(getattr(item, "file", "")) == file_path]

    uorb = profiler.extract_uorb_io_from_source(files)
    params = profiler.extract_params_from_source(files)
    structure = profiler.extract_source_structure_from_source(
        files, expand_companions=False
    )

    return SourceFileFacts(
        file=file_path,
        source_hash=source_hash,
        published_topics=from_exact_file(uorb.get("published_topics", []) or []),
        subscribed_topics=from_exact_file(uorb.get("subscribed_topics", []) or []),
        unknown_direction_topics=from_exact_file(
            uorb.get("unknown_direction_topics", []) or []
        ),
        referenced_parameters=from_exact_file(params or []),
        assigned_fields=from_exact_file(
            profiler.extract_assigned_fields_from_source(files)
        ),
        read_fields=from_exact_file(
            profiler.extract_read_fields_from_source(files)
        ),
        source_assignments=from_exact_file(
            profiler.extract_source_assignments_from_source(files)
        ),
        function_calls=from_exact_file(
            profiler.extract_function_calls_from_source(files)
        ),
        helper_expressions=from_exact_file(
            profiler.extract_helper_expressions_from_source(files)
        ),
        branch_conditions=from_exact_file(
            profiler.extract_branch_conditions_from_source(files)
        ),
        parameter_predicates=from_exact_file(
            profiler.extract_parameter_predicates_from_source(files)
        ),
        classes=list(structure.get("classes") or []),
        members=list(structure.get("members") or []),
        callables=list(structure.get("callables") or []),
        includes=list(structure.get("includes") or []),
    )
