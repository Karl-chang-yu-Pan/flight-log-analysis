"""
mechanism_source_profiler.py

Deterministic source-code profiler for PX4 mechanism discovery.

This module is intended to sit between a broad source-search stage and a
mechanism resolver agent. It does not call an LLM. It extracts structured facts
from PX4 source files:

- related source files for a user/mechanism query
- uORB publications/subscriptions
- PX4 parameters referenced by source
- assigned message/struct fields that may correspond to log fields

Typical use:

    profiler = MechanismSourceProfiler("/path/to/PX4-Autopilot")
    profile = profiler.profile_mechanism("fixed wing takeoff pitch")
    print(json.dumps(profile, indent=2))

Or from CLI:

    python3 mechanism_source_profiler.py /path/to/PX4-Autopilot "FW_TKO_PITCH_MIN takeoff pitch" \
        --out mechanism_profile.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, Field

from flight_log_agent.expression_math import (
    canonical_math_function_name,
    is_safe_math_function_name,
    normalize_expression_function_names,
)
from flight_log_agent.px4.source_snapshot import SourceHandle, SourceInput, source_handle


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SourceMatch(BaseModel):
    file: str
    line: int
    text: str
    query: str


class SourceFileHit(BaseModel):
    file: str
    score: float
    matched_queries: List[str] = Field(default_factory=list)
    matches: List[SourceMatch] = Field(default_factory=list)


class SourceStorageRef(BaseModel):
    """Declaration-derived identity for one C++ storage location."""

    kind: Literal["local", "member", "global", "unknown"]
    symbol: str
    root: str
    file: str = ""
    callable_id: str = ""
    class_owner: str = ""
    declaring_class: str = ""
    namespace_owner: str = ""
    declaration_id: str = ""
    declaration_proven: bool = False


class SourceCallResultRef(BaseModel):
    """One value projection rooted at a source-proven call expression."""

    call_source_site_id: str
    result_path: str = ""
    text: str = ""


class SourceExpressionRef(BaseModel):
    """Structured dependencies for one source expression.

    ``text`` preserves the source spelling while ``lowered_text`` carries the
    source-derived alias substitution used by the DAG. ``input_symbols``
    contains only storage reads; call results are represented separately so a
    projection such as ``object.read().field`` cannot become storage owned by
    ``object``. ``exact`` is true only when the syntax backend proved that the
    dependency lists are complete.
    """

    text: str
    lowered_text: Optional[str] = None
    input_symbols: List[str] = Field(default_factory=list)
    input_identities: Dict[str, SourceStorageRef] = Field(default_factory=dict)
    call_results: List[SourceCallResultRef] = Field(default_factory=list)
    exact: bool = False


class TopicRef(BaseModel):
    topic: str
    direction: str  # "publish", "subscribe", or "unknown"
    file: str
    line: int
    evidence: str
    struct: Optional[str] = None
    variable: Optional[str] = None
    api: Optional[str] = None
    instance: Optional[int] = None
    function: Optional[str] = None
    callable_id: Optional[str] = None
    # Structurally proven owner when ``variable`` is a class member. Locals
    # deliberately leave this unset so downstream scope checks cannot widen
    # them across methods based on spelling.
    variable_owner: Optional[str] = None
    # Structural endpoint identity emitted by syntax-aware backends. ``member``
    # and ``local`` endpoints are resolved by owner/callable scope; ``base``
    # denotes an initialized base subobject and uses ``this`` as its receiver.
    endpoint_kind: Optional[str] = None
    variable_identity: Optional[SourceStorageRef] = None
    source_site_id: Optional[str] = None
    control_predicates: List[str] = Field(default_factory=list)
    control_expression_refs: List[SourceExpressionRef] = Field(default_factory=list)
    reachability_exact: bool = True


class ParameterRef(BaseModel):
    name: Optional[str]
    file: str
    line: int
    evidence: str
    access_pattern: str
    member: Optional[str] = None
    confidence: str = "high"


class FieldRef(BaseModel):
    field: str
    file: str
    line: int
    evidence: str
    variable: Optional[str] = None
    topic: Optional[str] = None
    struct: Optional[str] = None
    assignment_operator: Optional[str] = None


class FunctionCallRef(BaseModel):
    name: str
    file: str
    line: int
    evidence: str
    receiver: Optional[str] = None
    receiver_identity: Optional[SourceStorageRef] = None
    receiver_type: Optional[str] = None
    resolved_callable_id: Optional[str] = None
    resolved_callable_file: Optional[str] = None
    resolved_callable_owner: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    argument_topics: Dict[str, str] = Field(default_factory=dict)
    control_predicates: List[str] = Field(default_factory=list)
    # Source line of each governing control statement, aligned with
    # ``control_predicates`` — the branch's own SITE identity, distinct
    # from this ref's ``line`` (the gated statement).
    control_predicate_lines: List[int] = Field(default_factory=list)
    control_predicate_site_ids: List[str] = Field(default_factory=list)
    # False when a governing construct the extractor does not model
    # (switch, loops) makes ``control_predicates`` incomplete — the
    # reachability is then explicitly unresolved, never silently partial.
    reachability_exact: bool = True
    symbol_bindings: Dict[str, str] = Field(default_factory=dict)
    # Call argument root -> owning class, populated only when the root is
    # declared as a member of the callable's class.
    argument_owners: Dict[str, str] = Field(default_factory=dict)
    function: Optional[str] = None
    callable_id: Optional[str] = None
    source_site_id: Optional[str] = None
    argument_expressions: List[SourceExpressionRef] = Field(default_factory=list)
    control_expression_refs: List[SourceExpressionRef] = Field(default_factory=list)


class SourceAssignmentRef(BaseModel):
    target: str
    expression: str
    file: str
    line: int
    evidence: str
    function: Optional[str] = None
    callable_id: Optional[str] = None
    owner: Optional[str] = None
    target_identity: Optional[SourceStorageRef] = None
    function_parameters: List[str] = Field(default_factory=list)
    target_topic: Optional[str] = None
    target_field: Optional[str] = None
    assignment_operator: Optional[str] = None
    declaration_kind: Optional[str] = None
    control_predicates: List[str] = Field(default_factory=list)
    # Source line of each governing control statement, aligned with
    # ``control_predicates`` — the branch's own SITE identity, distinct
    # from this ref's ``line`` (the gated statement).
    control_predicate_lines: List[int] = Field(default_factory=list)
    control_predicate_site_ids: List[str] = Field(default_factory=list)
    # False when a governing construct the extractor does not model
    # (switch, loops) makes ``control_predicates`` incomplete — the
    # reachability is then explicitly unresolved, never silently partial.
    reachability_exact: bool = True
    symbol_bindings: Dict[str, str] = Field(default_factory=dict)
    # Struct-typed variable → C++ struct type in scope at this assignment's
    # site. Lets the DAG derive ``var.field → topic.field`` bindings
    # graph-natively via :func:`_derive_topic_from_return_type` instead of
    # depending on the pre-baked ``symbol_bindings`` dict.
    struct_variables: Dict[str, str] = Field(default_factory=dict)
    source_site_id: Optional[str] = None
    expression_ref: Optional[SourceExpressionRef] = None
    control_expression_refs: List[SourceExpressionRef] = Field(default_factory=list)


class HelperExpressionRef(BaseModel):
    name: str
    owner: Optional[str] = None
    file: str
    line: int
    evidence: str
    callable_id: Optional[str] = None
    parameters: List[str] = Field(default_factory=list)
    # Source-derived default expressions aligned with formal positions.
    # ``None`` means the corresponding formal has no default. The tree-sitter
    # backend can merge companion-header defaults into the definition's helper
    # record without changing the callable's source identity.
    parameter_defaults: List[Optional[str]] = Field(default_factory=list)
    statements: List[Dict[str, Any]] = Field(default_factory=list)
    assignments: Dict[str, str] = Field(default_factory=dict)
    assignment_operators: Dict[str, str] = Field(default_factory=dict)
    assignment_expression_refs: Dict[str, SourceExpressionRef] = Field(
        default_factory=dict
    )
    assignment_sites: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    return_expression: Optional[str] = None
    return_expression_ref: Optional[SourceExpressionRef] = None
    lowered_return_expression: Optional[str] = None
    branches: List[Dict[str, Any]] = Field(default_factory=list)
    symbol_bindings: Dict[str, str] = Field(default_factory=dict)
    call_resolutions: List[Dict[str, Any]] = Field(default_factory=list)
    helper_calls: List[str] = Field(default_factory=list)
    # Alias-output writes carried on the helper record so the DAG builder
    # can emit graph-native operations at each call site (target derived
    # from the caller's actual arg). Each entry is
    # ``{"param": <formal>, "field": <dotted field>, "expression": <RHS>}``.
    # The historical field name is retained for schema compatibility; entries
    # may be proven through pointer or reference parameters.
    pointer_output_writes: List[Dict[str, str]] = Field(default_factory=list)
    # Raw C++ return type extracted from the function signature so the DAG
    # can derive source→logged bindings without the flat ``symbol_bindings``
    # table. Preserves pointer/reference qualifiers (``vehicle_status_s *``)
    # so downstream can distinguish struct-return helpers from scalar ones.
    return_type: Optional[str] = None
    # Struct-typed variable name → C++ struct type in scope of this helper
    # (both local variables declared in the body and class-member fields
    # visible via ``this->``). Lets the DAG derive ``var.field →
    # topic.field`` bindings graph-natively using the same
    # :func:`_derive_topic_from_return_type` utility as helper return types.
    struct_variables: Dict[str, str] = Field(default_factory=dict)
    unresolved_reason: Optional[str] = None


class BranchConditionRef(BaseModel):
    kind: str
    condition: str
    file: str
    line: int
    evidence: str
    source_site_id: Optional[str] = None
    condition_ref: Optional[SourceExpressionRef] = None


class ParameterPredicateRef(BaseModel):
    name: Optional[str]
    predicate: str
    file: str
    line: int
    evidence: str
    member: Optional[str] = None
    operator: Optional[str] = None
    compared_value: Optional[str] = None


class SourceClassRef(BaseModel):
    """One source-declared class and its direct inheritance relation."""

    name: str
    file: str
    line: int
    end_line: int
    bases: List[str] = Field(default_factory=list)


class SourceMemberRef(BaseModel):
    """A direct data-member declaration owned by a class."""

    name: str
    owner: str
    type: Optional[str] = None
    file: str
    line: int


class SourceCallableRef(BaseModel):
    """A source-defined callable with stable ownership and signature data."""

    name: str
    owner: Optional[str] = None
    file: str
    line: int
    end_line: int
    callable_id: str
    parameters: List[str] = Field(default_factory=list)
    parameter_types: List[str] = Field(default_factory=list)
    parameter_defaults: List[Optional[str]] = Field(default_factory=list)
    return_type: Optional[str] = None


def callable_parameter_count(record: Any) -> int:
    """Return the structurally known formal count for a callable record."""
    parameters = (
        record.get("parameters")
        if isinstance(record, dict)
        else getattr(record, "parameters", None)
    )
    parameter_types = (
        record.get("parameter_types")
        if isinstance(record, dict)
        else getattr(record, "parameter_types", None)
    )
    defaults = (
        record.get("parameter_defaults")
        if isinstance(record, dict)
        else getattr(record, "parameter_defaults", None)
    )
    count = max(
        len(parameters or []),
        len(parameter_types or []),
        len(defaults or []),
    )
    if count:
        return count

    raw_evidence = (
        record.get("evidence")
        if isinstance(record, dict)
        else getattr(record, "evidence", "")
    )
    evidence = str(raw_evidence or "")
    match = re.search(r"\((.*)\)", evidence)
    if not match or not match.group(1).strip() or match.group(1).strip() == "void":
        return 0
    return len(
        [value for value in split_top_level_args(match.group(1)) if value.strip()]
    )


def callable_arguments_with_defaults(
    record: Any,
    arguments: Sequence[str],
) -> Optional[List[str]]:
    """Return a complete positional argument list or fail closed.

    Defaults are accepted only when source extraction supplied an expression
    for every omitted formal. Older records without default metadata therefore
    retain exact-arity behavior.
    """
    explicit = [str(value).strip() for value in arguments]
    total = callable_parameter_count(record)
    if len(explicit) > total:
        return None

    raw_defaults = (
        record.get("parameter_defaults")
        if isinstance(record, dict)
        else getattr(record, "parameter_defaults", None)
    )
    defaults: List[Optional[str]] = list(raw_defaults or [])
    if len(defaults) < total:
        defaults.extend([None] * (total - len(defaults)))

    omitted = defaults[len(explicit) : total]
    if any(value is None or not str(value).strip() for value in omitted):
        return None
    return [*explicit, *(str(value).strip() for value in omitted)]


def callable_accepts_argument_count(record: Any, argument_count: int) -> bool:
    """Whether ``argument_count`` is valid under source-declared defaults."""
    return callable_arguments_with_defaults(record, [""] * argument_count) is not None


class SourceIncludeRef(BaseModel):
    """A source include resolved to a file in the configured source tree."""

    file: str
    included_file: str
    line: int


class MechanismSourceProfile(BaseModel):
    query: str
    source_root: str
    related_files: List[SourceFileHit]
    published_topics: List[TopicRef]
    subscribed_topics: List[TopicRef]
    unknown_direction_topics: List[TopicRef]
    referenced_parameters: List[ParameterRef]
    assigned_fields: List[FieldRef]
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
    notes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class _HelperLoweringFailed(Exception):
    """Raised by a lowering step that decided the helper cannot be lowered.

    The exception's ``reason`` carries the precise message the resolution
    attempt produced (e.g. "for loop bound is not statically resolvable").
    :meth:`_translate_helper_body` catches it and surfaces ``reason`` as
    the helper's ``unresolved_reason``.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class MechanismSourceProfiler:
    """
    Deterministic PX4 source profiler.

    The class intentionally uses regex/ripgrep-style static extraction. It is
    not a compiler and will not perfectly understand every C++ construct, but it
    gives the resolver an explicit, auditable list of files/topics/params/fields
    to check against the log.
    """

    SOURCE_GLOBS = (
        "*.c",
        "*.cc",
        "*.cpp",
        "*.cxx",
        "*.h",
        "*.hpp",
        "*.hh",
        "*.hxx",
    )

    DEFAULT_EXCLUDES = (
        ".git",
        "build",
        "build_*",
        "cmake-build-*",
        "Tools/sitl_gazebo",
        "Tools/simulation",
        "platforms/nuttx/NuttX",
    )
    LOW_VALUE_QUERY_TOKENS = {
        "about",
        "above",
        "active",
        "around",
        "branch",
        "case",
        "chain",
        "class",
        "complete",
        "compute",
        "covering",
        "defines",
        "definition",
        "definitions",
        "determine",
        "determines",
        "especially",
        "extract",
        "fallback",
        "function",
        "handling",
        "including",
        "logic",
        "mechanism",
        "metadata",
        "method",
        "minimum",
        "parameter",
        "parameters",
        "profile",
        "references",
        "relative",
        "return",
        "selected",
        "selection",
        "source",
        "values",
        "whether",
    }
    MAX_SCORE_PER_FILE_QUERY = 12.0

    # PX4/uORB declaration patterns.
    _UORB_DECL_PATTERNS: Sequence[Tuple[str, str, re.Pattern]] = (
        (
            "publish",
            "uORB::Publication",
            re.compile(
                r"\buORB::Publication(?:Data|Multi|Queued)?\s*<\s*(?P<struct>[A-Za-z_][A-Za-z0-9_]*_s)\s*>\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)?"
            ),
        ),
        (
            "subscribe",
            "uORB::Subscription",
            re.compile(
                r"\buORB::Subscription(?:Data|Interval|CallbackWorkItem)?\s*<\s*(?P<struct>[A-Za-z_][A-Za-z0-9_]*_s)\s*>\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)?"
            ),
        ),
        (
            "subscribe",
            "uORB::SubscriptionCallbackWorkItem",
            re.compile(
                r"\buORB::SubscriptionCallbackWorkItem\s*<\s*(?P<struct>[A-Za-z_][A-Za-z0-9_]*_s)\s*>\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)?"
            ),
        ),
    )

    _ORB_CALL_PATTERNS: Sequence[Tuple[str, str, re.Pattern]] = (
        (
            "publish",
            "orb_advertise/orb_publish",
            re.compile(
                r"\borb_(?:advertise(?:_multi)?|publish)\s*\([^;\n]*?ORB_ID\s*\(\s*(?P<topic>[A-Za-z_][A-Za-z0-9_]*)\s*\)"
            ),
        ),
        (
            "subscribe",
            "orb_subscribe/orb_copy",
            re.compile(
                r"\borb_(?:subscribe(?:_multi)?|copy)\s*\([^;\n]*?ORB_ID\s*\(\s*(?P<topic>[A-Za-z_][A-Za-z0-9_]*)\s*\)"
            ),
        ),
        (
            "unknown",
            "ORB_ID",
            re.compile(r"\bORB_ID\s*\(\s*(?P<topic>[A-Za-z_][A-Za-z0-9_]*)\s*\)"),
        ),
    )
    _UORB_OBJECT_DECL_PATTERN = re.compile(
        r"\buORB::(?P<api>(?:Publication|Subscription)[A-Za-z0-9_]*)"
        r"(?:\s*<\s*(?P<template>[^;{}]+?)\s*>)?\s+"
        r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*[\{(]"
        r"(?P<initializer>[^;\n]*)"
    )
    _ORB_ID_REFERENCE_PATTERN = re.compile(
        r"\bORB_ID\s*(?:\(\s*(?P<call>[A-Za-z_][A-Za-z0-9_]*)\s*\)|"
        r"::\s*(?P<scope>[A-Za-z_][A-Za-z0-9_]*))"
    )
    _CLASS_DECL_PATTERN = re.compile(
        r"\b(?:class|struct)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        r"(?:\s+final)?(?:\s*:\s*(?P<bases>[^{]+))?\s*\{"
    )

    # Parameter reference patterns.
    _PARAM_DECL_PATTERN = re.compile(
        r"\(?\s*\bParam(?:Bool|Int|Float|ExtBool|ExtInt|ExtFloat|Custom)?\s*<\s*"
        r"px4::params::(?P<name>[A-Z][A-Z0-9_]*)\s*>\s*\)?\s*"
        r"(?P<member>[A-Za-z_][A-Za-z0-9_]*)?"
    )
    _PX4_PARAM_PATTERN = re.compile(r"\bpx4::params::(?P<name>[A-Z][A-Z0-9_]*)\b")
    _PARAM_FIND_PATTERN = re.compile(r"\bparam_find\s*\(\s*\"(?P<name>[A-Z][A-Z0-9_]*)\"\s*\)")
    _PARAM_GET_MEMBER_PATTERN = re.compile(
        r"\b(?P<member>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*\.\s*get\s*\(\s*\)"
    )

    # C/C++ variable declaration of uORB struct/message instances.
    _STRUCT_VAR_PATTERNS: Sequence[re.Pattern] = (
        re.compile(
            r"\b(?P<struct>[A-Za-z_][A-Za-z0-9_]*_s)\s+(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\{|=|;|\[|\))"
        ),
        re.compile(
            r"\b(?P<struct>[A-Za-z_][A-Za-z0-9_]*_s)\s*[&*]+\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\b"
        ),
    )

    # Assignment to a struct field, e.g. attitude_sp.roll_body = ...
    _FIELD_ASSIGN_PATTERN = re.compile(
        r"\b(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<access>\.|->)\s*"
        r"(?P<field>[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*"
        r"(?P<op>\+=|-=|\*=|/=|%=|\|=|&=|\^=|=(?!=))"
    )
    _FIELD_ACCESS_PATTERN = re.compile(
        r"\b(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<access>\.|->)\s*"
        r"(?P<field>[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*)*)"
    )
    _FUNCTION_CALL_PATTERN = re.compile(
        r"(?<![#A-Za-z0-9_])(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Za-z_][A-Za-z0-9_]*)\s*\("
    )
    # ``=(?!=)`` rejects the first ``=`` of an equality comparison —
    # ``if (mode == 1) x = 2;`` must extract ``x <- 2``, never the junk
    # binding ``mode <- = 1) x = 2`` (comparison-as-assignment).
    _SOURCE_ASSIGNMENT_PATTERN = re.compile(
        r"(?P<target>[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*)*)"
        r"\s*=(?!=)\s*(?P<expr>[^;]+);"
    )
    # C++ reference declaration: ``[const] Type &name = expression;``.
    # Captured per-function so subsequent uses of ``name.X`` in the body can
    # be rewritten to ``expression.X`` before assignments / field accesses are
    # extracted. The alias is only applied when the RHS ends in a member
    # access (``.foo`` or ``->foo`` without trailing ``()``) — otherwise the
    # existing struct-var binding path (which handles cases like
    # ``Type &name = *getter();``) is the better resolution.
    _REFERENCE_ALIAS_PATTERN = re.compile(
        r"(?:const\s+)?"
        r"(?P<type>[A-Za-z_][A-Za-z0-9_:<>]*)"
        r"\s*&\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        r"\s*=\s*(?P<expr>[^;]+);"
    )
    _ALIAS_TRAILING_MEMBER_ACCESS_RE = re.compile(
        r"(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*\s*$"
    )
    _SOURCE_COMPOUND_ASSIGNMENT_PATTERN = re.compile(
        r"(?P<target>[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*)*)"
        r"\s*(?P<op>\+=|-=|\*=|/=|%=|\|=|&=|\^=)\s*(?P<expr>[^;]+);"
    )
    # C/C++ enum block: ``enum [class] [Name] [: base] { body };``. Used to
    # extract NAME = VALUE entries inside the body so PX4-defined bitmask
    # constants flow through source_assignments and become resolvable by
    # the slicer + predicate parser like any other symbol.
    _ENUM_BLOCK_PATTERN = re.compile(
        r"enum\s+(?:class\s+|struct\s+)?"
        r"(?:[A-Za-z_][A-Za-z0-9_]*\s*)?"
        r"(?::\s*[A-Za-z_][A-Za-z0-9_:\s]*\s*)?"
        r"\{(?P<body>[^{}]*)\}",
        re.DOTALL,
    )
    _ENUM_ENTRY_PATTERN = re.compile(
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>[^,\n]+?)\s*(?:,|$)",
        re.MULTILINE,
    )
    # Simple object-like ``#define NAME VALUE`` macro. Function-like macros
    # (``#define NAME(args) body``) don't match because they lack the
    # required whitespace between NAME and the body.
    _DEFINE_PATTERN = re.compile(
        r"^\s*#\s*define\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<value>[^/\n]+?)(?:\s*//.*)?\s*$",
        re.MULTILINE,
    )
    # The separator before the name is ``[\s*&]+`` (not just ``\s+``) so a
    # pointer/reference that hugs the name is accepted — PX4 writes uORB
    # accessors as ``vehicle_status_s *get_vstatus()`` with the ``*`` against
    # the name. The non-greedy prefix leaves the ``*``/``&`` in the separator,
    # so ``return_type`` stays the clean struct name.
    _FUNCTION_SIGNATURE_PATTERN = re.compile(
        r"(?P<prefix>[A-Za-z_][A-Za-z0-9_:<>,~*&\s]*?)[\s*&]+"
        r"(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Za-z_][A-Za-z0-9_]*)"
        r"\s*\((?P<params>[^()]*)\)\s*(?:const\s*)?$"
    )
    _BRANCH_CONDITION_PATTERN = re.compile(
        r"\b(?P<kind>if|else\s+if|while|switch)\s*\((?P<condition>[^;\n]*)\)"
    )
    _CASE_CONDITION_PATTERN = re.compile(r"\bcase\s+(?P<condition>[^:\n]+)\s*:")
    _PARAM_COMPARISON_PATTERN = re.compile(
        r"(?P<left>[A-Za-z_][A-Za-z0-9_.:]*\s*(?:\.\s*get\s*\(\s*\))?)\s*"
        r"(?P<op>>=|<=|==|!=|>|<)\s*"
        r"(?P<right>-?[A-Za-z_][A-Za-z0-9_:]*|-?\d+(?:\.\d+)?|true|false)"
    )

    def __init__(
        self,
        source_path: SourceInput,
        rg_path: str = "rg",
        read_limit_bytes: int = 2_000_000,
        excludes: Optional[Sequence[str]] = None,
        source_parser_backend: str = "legacy",
    ) -> None:
        source = source_handle(source_path)
        if source is None:
            raise FileNotFoundError("PX4 source is unavailable.")
        self.source: SourceHandle = source
        self.rg_path = rg_path
        self.read_limit_bytes = read_limit_bytes
        self.excludes = tuple(excludes) if excludes is not None else self.DEFAULT_EXCLUDES
        backend = str(source_parser_backend or "legacy").strip().lower()
        if backend not in {"legacy", "tree_sitter", "compare"}:
            raise ValueError(
                "source_parser_backend must be 'legacy', 'tree_sitter', or 'compare'"
            )
        self.source_parser_backend = backend
        # Per-instance cache: the same file is hit by 10+ extraction
        # methods across multiple source_discovery iterations. Keyed by
        # the rel path string so the path-resolution variants converge.
        self._text_cache: dict[str, Optional[str]] = {}
        self._search_cache: dict[str, List[SourceMatch]] = {}
        # Several fact extractors consume the same call-site inventory. Keep
        # this parse memo local to one profiler/run; it is not a persisted DAG
        # or source-facts cache and cannot survive a source snapshot change.
        self._function_call_cache: dict[tuple[str, ...], List[FunctionCallRef]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_related_source_files(
        self,
        queries: Union[str, Sequence[str]],
        max_files: Optional[int] = 12,
        max_matches_per_file: int = 8,
        expand_query_tokens: bool = True,
    ) -> List[SourceFileHit]:
        """
        Search the PX4 source tree for files related to one or more query terms.

        Args:
            queries: Single query string or list of query strings. Use concrete
                terms when possible, e.g. "FW_TKO_PITCH_MIN", "takeoff pitch",
                "position_setpoint_triplet".
            max_files: Maximum number of ranked source files to return. ``None``
                returns every match so deterministic expansion can validate
                candidates without a correctness-affecting retrieval cap.
            max_matches_per_file: Number of evidence lines to keep per file.
            expand_query_tokens: Also search strong identifier tokens parsed
                from prose queries. Deterministic definition lookup disables
                this so an exact ``::callable(`` query is not broadened.

        Returns:
            Ranked source file hits with evidence lines.
        """
        query_list = self._normalize_queries(
            queries, expand_tokens=expand_query_tokens
        )
        by_file: Dict[str, SourceFileHit] = {}
        score_by_file_query: Dict[Tuple[str, str], float] = {}

        for query in query_list:
            matches = self._ripgrep_or_python_search(query)
            for match in matches:
                hit = by_file.setdefault(
                    match.file,
                    SourceFileHit(file=match.file, score=0.0, matched_queries=[], matches=[]),
                )
                if query not in hit.matched_queries:
                    hit.matched_queries.append(query)

                if len(hit.matches) < max_matches_per_file:
                    hit.matches.append(match)

                score = self._score_match(match, query)
                score_key = (match.file, query)
                current_score = score_by_file_query.get(score_key, 0.0)
                remaining_score = max(self.MAX_SCORE_PER_FILE_QUERY - current_score, 0.0)
                applied_score = min(score, remaining_score)
                if applied_score > 0:
                    hit.score += applied_score
                    score_by_file_query[score_key] = current_score + applied_score

        # Prefer real flight-stack source over tests/examples when scores tie.
        for hit in by_file.values():
            hit.score += self._file_path_boost(hit.file)

        ranked = sorted(by_file.values(), key=lambda x: (-x.score, x.file))
        return ranked if max_files is None else ranked[:max_files]

    def extract_uorb_io_from_source(
        self,
        files: Sequence[Union[str, Path]],
    ) -> Dict[str, List[TopicRef]]:
        """
        Extract uORB publications/subscriptions from source files.

        Returns:
            {
                "published_topics": [...],
                "subscribed_topics": [...],
                "unknown_direction_topics": [...]
            }
        """
        refs: List[TopicRef] = []
        expanded_files = self._expand_companion_files(files)

        for path in expanded_files:
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            definitions = self._extract_function_definitions(text, rel_file)
            class_definitions = self._extract_class_definitions(text)
            for line_no, line in self._iter_code_lines(text):
                object_declarations = list(self._UORB_OBJECT_DECL_PATTERN.finditer(line))
                for match in object_declarations:
                    api = match.group("api")
                    direction = "publish" if api.startswith("Publication") else "subscribe"
                    topic_matches = list(
                        self._ORB_ID_REFERENCE_PATTERN.finditer(match.group("initializer"))
                    )
                    topics = {
                        topic_match.group("call") or topic_match.group("scope")
                        for topic_match in topic_matches
                    }
                    if len(topics) != 1:
                        continue
                    topic = next(iter(topics))
                    topic_match = topic_matches[0]
                    initializer_tail = match.group("initializer")[topic_match.end():]
                    instance_match = re.match(r"\s*,\s*(?P<instance>\d+)\b", initializer_tail)
                    instance = instance_match.group("instance") if instance_match else None
                    template = str(match.group("template") or "")
                    struct_match = re.search(
                        r"\b(?P<struct>[A-Za-z_][A-Za-z0-9_]*_s)\b", template
                    )
                    definition = self._function_definition_for_line(definitions, line_no)
                    owner = None if definition else self._class_owner_for_line(
                        class_definitions, line_no
                    )
                    refs.append(
                        TopicRef(
                            topic=topic,
                            struct=(
                                struct_match.group("struct")
                                if struct_match
                                else self._struct_from_topic(topic)
                            ),
                            variable=match.group("var"),
                            direction=direction,
                            api=f"uORB::{api}",
                            instance=int(instance) if instance is not None else None,
                            variable_owner=owner,
                            file=rel_file,
                            line=line_no,
                            evidence=line.strip(),
                        )
                    )
                for direction, api, pattern in self._UORB_DECL_PATTERNS:
                    for match in pattern.finditer(line):
                        struct = match.group("struct")
                        topic = self._topic_from_struct(struct)
                        variable = match.groupdict().get("var")
                        if any(
                            declaration.group("var") == variable
                            for declaration in object_declarations
                        ):
                            continue
                        refs.append(
                            TopicRef(
                                topic=topic,
                                struct=struct,
                                variable=variable,
                                direction=direction,
                                api=api,
                                file=rel_file,
                                line=line_no,
                                evidence=line.strip(),
                            )
                        )

                # First record specific call-site direction. Then record generic
                # ORB_ID only if this line did not already produce a specific ref.
                specific_seen_spans: List[Tuple[int, int]] = []

                for direction, api, pattern in self._ORB_CALL_PATTERNS:
                    if direction == "unknown":
                        continue
                    for match in pattern.finditer(line):
                        specific_seen_spans.append(match.span())
                        refs.append(
                            TopicRef(
                                topic=match.group("topic"),
                                struct=self._struct_from_topic(match.group("topic")),
                                direction=direction,
                                api=api,
                                file=rel_file,
                                line=line_no,
                                evidence=line.strip(),
                            )
                        )

                for direction, api, pattern in self._ORB_CALL_PATTERNS:
                    if direction != "unknown":
                        continue
                    for match in pattern.finditer(line):
                        # Skip generic ORB_ID if already captured through a more
                        # specific orb_publish/orb_subscribe/orb_copy pattern.
                        if any(start <= match.start() <= end for start, end in specific_seen_spans):
                            continue
                        refs.append(
                            TopicRef(
                                topic=match.group("topic"),
                                struct=self._struct_from_topic(match.group("topic")),
                                direction="unknown",
                                api=api,
                                file=rel_file,
                                line=line_no,
                                evidence=line.strip(),
                            )
                        )

        # C uORB APIs carry the message object in the call arguments instead
        # of a wrapper declaration. Reuse the token-aware call extraction so
        # multi-line calls and callable identity stay consistent.
        for call in self.extract_function_calls_from_source(files):
            short_name = call.name.rsplit("::", 1)[-1]
            topic_arg: Optional[int] = None
            data_arg: Optional[int] = None
            direction = ""
            if short_name == "orb_copy" and len(call.args) >= 3:
                direction, topic_arg, data_arg = "subscribe", 0, 2
            elif short_name.startswith("orb_publish") and len(call.args) >= 3:
                direction, topic_arg, data_arg = "publish", 0, 2
            elif short_name.startswith("orb_advertise") and len(call.args) >= 2:
                direction, topic_arg, data_arg = "publish", 0, 1
            if topic_arg is None or data_arg is None:
                continue
            topic = self._topic_from_orb_reference(call.args[topic_arg])
            source_symbol = self._boundary_argument_symbol(call.args[data_arg])
            if not topic or not source_symbol:
                continue
            root = source_symbol.replace("->", ".").split(".", 1)[0]
            refs.append(
                TopicRef(
                    topic=topic,
                    struct=self._struct_from_topic(topic),
                    variable=source_symbol,
                    direction=direction,
                    api=short_name,
                    file=call.file,
                    line=call.line,
                    evidence=call.evidence,
                    function=call.function,
                    callable_id=call.callable_id,
                    variable_owner=call.argument_owners.get(root),
                )
            )

        refs = self._dedupe_topic_refs(refs)
        return {
            "published_topics": [r for r in refs if r.direction == "publish"],
            "subscribed_topics": [r for r in refs if r.direction == "subscribe"],
            "unknown_direction_topics": [r for r in refs if r.direction == "unknown"],
        }

    def extract_params_from_source(
        self,
        files: Sequence[Union[str, Path]],
    ) -> List[ParameterRef]:
        """
        Extract PX4 parameters referenced in source files.

        Handles common styles:
        - ParamFloat<px4::params::FW_AIRSPD_TRIM> _param_fw_airspd_trim
        - px4::params::FW_AIRSPD_TRIM
        - param_find("FW_AIRSPD_TRIM")
        - _param_fw_airspd_trim.get(), if the declaration was visible in one of
          the profiled files or companion header/source files.
        """
        refs: List[ParameterRef] = []
        member_to_param: Dict[str, str] = {}

        expanded_files = self._expand_companion_files(files)

        # First pass: collect declarations so member.get() can be normalized.
        for path in expanded_files:
            text = self._read_text(path)
            if text is None:
                continue
            for _, line in self._iter_code_lines(text):
                for match in self._PARAM_DECL_PATTERN.finditer(line):
                    name = match.group("name")
                    member = match.groupdict().get("member")
                    if member:
                        member_to_param[member] = name
                        # Common use: DEFINE_PARAMETERS macro puts member names
                        # right after a closing parenthesis. Keep both variants.
                        member_to_param[member.lstrip("_ ")] = name

        # Second pass: emit parameter references.
        for path in expanded_files:
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            for line_no, line in self._iter_code_lines(text):
                for match in self._PARAM_DECL_PATTERN.finditer(line):
                    refs.append(
                        ParameterRef(
                            name=match.group("name"),
                            member=match.groupdict().get("member"),
                            access_pattern="Param<px4::params::PARAM>",
                            file=rel_file,
                            line=line_no,
                            evidence=line.strip(),
                            confidence="high",
                        )
                    )

                for match in self._PX4_PARAM_PATTERN.finditer(line):
                    refs.append(
                        ParameterRef(
                            name=match.group("name"),
                            access_pattern="px4::params::PARAM",
                            file=rel_file,
                            line=line_no,
                            evidence=line.strip(),
                            confidence="high",
                        )
                    )

                for match in self._PARAM_FIND_PATTERN.finditer(line):
                    refs.append(
                        ParameterRef(
                            name=match.group("name"),
                            access_pattern='param_find("PARAM")',
                            file=rel_file,
                            line=line_no,
                            evidence=line.strip(),
                            confidence="high",
                        )
                    )

                for match in self._PARAM_GET_MEMBER_PATTERN.finditer(line):
                    member_expr = match.group("member")
                    member = member_expr.split(".")[-1]
                    name = member_to_param.get(member_expr) or member_to_param.get(member)

                    # Avoid treating normal container/value .get() calls as PX4
                    # params unless the name looks parameter-like or is mapped.
                    if name is None and not self._looks_like_param_member(member):
                        continue

                    refs.append(
                        ParameterRef(
                            name=name,
                            member=member_expr,
                            access_pattern="member.get()",
                            file=rel_file,
                            line=line_no,
                            evidence=line.strip(),
                            confidence="high" if name else "low",
                        )
                    )

        return self._dedupe_param_refs(refs)

    def extract_assigned_fields_from_source(
        self,
        files: Sequence[Union[str, Path]],
        *,
        relevance_terms: Optional[Sequence[str]] = None,
    ) -> List[FieldRef]:
        """
        Extract assigned fields from source files.

        The method maps fields to a uORB topic when it can infer the variable's
        struct type. For example:

            vehicle_attitude_setpoint_s attitude_sp{};
            attitude_sp.pitch_body = pitch_sp;

        becomes:

            topic=vehicle_attitude_setpoint, field=pitch_body

        It also keeps unknown fields only when they match the active
        question/search context, because unmapped variable names are not source
        proof by themselves.
        """
        refs: List[FieldRef] = []
        dynamic_terms = self._dynamic_relevance_terms(relevance_terms or [])

        for path in self._expand_companion_files(files):
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            var_to_struct = self._extract_struct_variables(text)

            for line_no, line in self._iter_code_lines(text):
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue

                for match in self._FIELD_ASSIGN_PATTERN.finditer(line):
                    var = match.group("var")
                    field_name = self._clean_field_path(match.group("field"))
                    op = match.group("op")

                    # Filter obvious non-message assignments. Keep mapped uORB
                    # struct fields. Unknown structs must match the current
                    # question/search context instead of a global flight-domain
                    # keyword list.
                    struct = var_to_struct.get(var)
                    if struct is None and not self._matches_dynamic_relevance(var, field_name, dynamic_terms):
                        continue

                    topic = self._topic_from_struct(struct) if struct else None
                    refs.append(
                        FieldRef(
                            variable=var,
                            field=field_name,
                            topic=topic,
                            struct=struct,
                            assignment_operator=op,
                            file=rel_file,
                            line=line_no,
                            evidence=stripped,
                        )
                    )

        return self._dedupe_field_refs(refs)

    def extract_read_fields_from_source(
        self,
        files: Sequence[Union[str, Path]],
        *,
        relevance_terms: Optional[Sequence[str]] = None,
    ) -> List[FieldRef]:
        """
        Extract non-assignment field accesses from source files.

        This is intentionally conservative. It keeps fields that can be mapped
        to a visible uORB struct variable, plus unknown fields that match the
        active question/search context.
        """
        refs: List[FieldRef] = []
        dynamic_terms = self._dynamic_relevance_terms(relevance_terms or [])

        for path in self._expand_companion_files(files):
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            var_to_struct = self._extract_struct_variables(text)

            for line_no, line in self._iter_code_lines(text):
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue

                assignment_spans = [match.span() for match in self._FIELD_ASSIGN_PATTERN.finditer(line)]
                for match in self._FIELD_ACCESS_PATTERN.finditer(line):
                    if any(start <= match.start() < end for start, end in assignment_spans):
                        continue

                    var = match.group("var")
                    field_name = self._clean_field_path(match.group("field"))
                    struct = var_to_struct.get(var)
                    if struct is None and not self._matches_dynamic_relevance(var, field_name, dynamic_terms):
                        continue

                    topic = self._topic_from_struct(struct) if struct else None
                    refs.append(
                        FieldRef(
                            variable=var,
                            field=field_name,
                            topic=topic,
                            struct=struct,
                            file=rel_file,
                            line=line_no,
                            evidence=stripped,
                        )
                    )

        return self._dedupe_field_refs(refs)

    def extract_source_assignments_from_source(
        self,
        files: Sequence[Union[str, Path]],
        *,
        expand_companions: bool = True,
        include_pointer_outputs: bool = True,
        include_control_flow: bool = True,
    ) -> List[SourceAssignmentRef]:
        refs: List[SourceAssignmentRef] = []
        paths = self._expand_companion_files(files) if expand_companions else [
            self._resolve_file(file_path) for file_path in files
        ]
        for path in paths:
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            var_to_struct = self._extract_struct_variables(text)
            definitions = self._extract_function_definitions(text, rel_file)
            aliases_per_function = self._extract_reference_aliases_per_function(definitions)
            if include_control_flow:
                control_predicates, unresolved_reach = self._control_predicates_by_line(text)
            else:
                # Candidate screening needs exact targets and callable scope,
                # but reachability is relevant only after a file is admitted.
                control_predicates, unresolved_reach = {}, set()

            code_lines = list(self._iter_code_lines(text))
            for index, (line_no, line) in enumerate(code_lines):
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                if (
                    "=" in stripped
                    and not stripped.endswith((";", "{", "}"))
                    and stripped.count("(") > stripped.count(")")
                ):
                    # RHS continues on later lines (a multi-line call
                    # initializer) — join until parens balance so the
                    # assignment pattern can match the full statement.
                    for _, continuation in code_lines[index + 1 : index + 26]:
                        continuation_code = continuation.split("//", 1)[0].strip()
                        line = line + " " + continuation_code
                        if line.count("(") <= line.count(")"):
                            break
                    stripped = line.strip()
                function_name = self._function_name_for_line(definitions, line_no)
                definition = self._function_definition_for_line(definitions, line_no)
                callable_id = self._callable_id(definition)
                func_aliases = aliases_per_function.get(function_name or "", {})
                predicate_entries = control_predicates.get(line_no, [])
                predicates = [p for p, _ in predicate_entries]
                predicate_lines = [s for _, s in predicate_entries]
                reachability_exact = line_no not in unresolved_reach
                # Constructor-style initialization ``Type name(expr);``
                # inside a function body is an assignment the ``=`` pattern
                # misses (the declaration form loses e.g. quaternion
                # inputs entirely).
                if function_name:
                    ctor = re.match(
                        r"^\s*(?:[A-Za-z_][\w:]*(?:<[^<>]*>)?)\s+"
                        r"(?P<target>[A-Za-z_]\w*)\s*\((?P<expr>[^;={}]+)\)\s*;\s*$",
                        line,
                    )
                    if ctor and not self._FUNCTION_SIGNATURE_PATTERN.search(
                        " ".join(line.split())
                    ):
                        refs.append(
                            SourceAssignmentRef(
                                target=ctor.group("target"),
                                expression=self._normalize_source_expression(
                                    ctor.group("expr")
                                ),
                                function=function_name,
                                callable_id=callable_id,
                                assignment_operator="=",
                                file=rel_file,
                                line=line_no,
                                evidence=stripped,
                                control_predicates=predicates,
                                control_predicate_lines=predicate_lines,
                                reachability_exact=reachability_exact,
                                struct_variables=dict(var_to_struct),
                            )
                        )
                for match in self._SOURCE_ASSIGNMENT_PATTERN.finditer(line):
                    target = self._clean_field_path(match.group("target"))
                    target = self._apply_reference_alias(target, func_aliases)
                    expression = self._normalize_source_expression(match.group("expr"))
                    declaration_kind = (
                        "constexpr"
                        if re.search(r"\bconstexpr\b", line[: match.start("target")])
                        else None
                    )
                    # Skip self-writes that the alias substitution produces.
                    # Two cases this covers:
                    #   1. The alias declaration line itself, which the
                    #      assignment regex matches as ``curr_sp = X.Y``;
                    #      after substitution that collapses to a self-write.
                    #   2. A pointer initialisation like
                    #      ``Type *curr_sp = &X.Y;`` that lives in an outer
                    #      scope but shares the variable name with a
                    #      reference declared in an inner block — the
                    #      profiler's alias extractor is function-scoped
                    #      (not block-scoped), so the alias leaks across.
                    #      Stripping leading ``&*`` from the expression lets
                    #      the self-write check still catch these.
                    if self._clean_field_path(expression.lstrip("&* ")) == target:
                        continue
                    root, field = split_source_field(target)
                    struct = var_to_struct.get(root)
                    target_topic = self._topic_from_struct(struct) if struct else None
                    refs.append(
                        SourceAssignmentRef(
                            target=target,
                            expression=expression,
                            target_topic=target_topic,
                            target_field=field if target_topic else None,
                            function=function_name,
                            callable_id=callable_id,
                            function_parameters=list(definition.get("params", [])) if definition else [],
                            assignment_operator="=",
                            declaration_kind=declaration_kind,
                            file=rel_file,
                            line=line_no,
                            evidence=stripped,
                            control_predicates=predicates,
                            control_predicate_lines=predicate_lines,
                            reachability_exact=reachability_exact,
                            symbol_bindings=self._source_symbol_bindings(
                                " ".join([target, expression, *predicates]),
                                var_to_struct,
                            ),
                            struct_variables=dict(var_to_struct),
                        )
                    )
                for match in self._SOURCE_COMPOUND_ASSIGNMENT_PATTERN.finditer(line):
                    target = self._clean_field_path(match.group("target"))
                    target = self._apply_reference_alias(target, func_aliases)
                    operator = match.group("op")
                    expression = self._compound_assignment_expression(
                        target,
                        operator,
                        self._normalize_source_expression(match.group("expr")),
                    )
                    root, field = split_source_field(target)
                    struct = var_to_struct.get(root)
                    target_topic = self._topic_from_struct(struct) if struct else None
                    refs.append(
                        SourceAssignmentRef(
                            target=target,
                            expression=expression,
                            target_topic=target_topic,
                            target_field=field if target_topic else None,
                            function=function_name,
                            callable_id=callable_id,
                            function_parameters=list(definition.get("params", [])) if definition else [],
                            assignment_operator=operator,
                            file=rel_file,
                            line=line_no,
                            evidence=stripped,
                            control_predicates=predicates,
                            control_predicate_lines=predicate_lines,
                            reachability_exact=reachability_exact,
                            symbol_bindings=self._source_symbol_bindings(
                                " ".join([target, expression, *predicates]),
                                var_to_struct,
                            ),
                            struct_variables=dict(var_to_struct),
                        )
                    )

            refs.extend(self._extract_constant_definitions(text, rel_file))
        if include_pointer_outputs:
            refs.extend(self._extract_pointer_output_call_site_assignments(files))
        return self._dedupe_source_assignment_refs(refs)

    def _extract_pointer_output_call_site_assignments(
        self,
        files: Sequence[Union[str, Path]],
    ) -> List[SourceAssignmentRef]:
        """For functions whose body writes through pointer params, emit the
        body's writes as source_assignments at each call site with the
        call-site arg substituted for the parameter.

        Lets PX4 helpers like ``void mission_item_to_position_setpoint
        (const mission_item_s &item, position_setpoint_s *sp)`` — which
        deliver their result via an out-pointer instead of a return value —
        feed the slicer through ``source_assignments`` instead of being
        rejected as pointer-output mutations.
        """
        pointer_funcs: Dict[str, Dict[str, Any]] = {}
        expanded_files = self._expand_companion_files(files)
        for path in expanded_files:
            text = self._read_text(path)
            if text is None:
                continue
            rel_file = self._rel(path)
            for definition in self._extract_function_definitions(text, rel_file):
                pointer_params = self._function_pointer_params(definition)
                if not pointer_params:
                    continue
                writes = self._pointer_output_writes(str(definition.get("body") or ""), pointer_params)
                if not writes:
                    continue
                pointer_funcs[str(definition.get("name") or "")] = {
                    "pointer_params": pointer_params,
                    "writes": writes,
                }

        if not pointer_funcs:
            return []

        refs: List[SourceAssignmentRef] = []
        function_calls = self.extract_function_calls_from_source(files)
        for call in function_calls:
            info = pointer_funcs.get(call.name) or pointer_funcs.get(call.name.split("::")[-1])
            if info is None:
                continue
            for substituted in self._substitute_pointer_writes(
                info["writes"], info["pointer_params"], call.args
            ):
                refs.append(
                    SourceAssignmentRef(
                        target=substituted["target"],
                        expression=substituted["expression"],
                        target_topic=None,
                        target_field=None,
                        function=None,
                        function_parameters=[],
                        assignment_operator="=",
                        file=call.file,
                        line=call.line,
                        evidence=call.evidence,
                        control_predicates=list(call.control_predicates or []),
                        control_predicate_lines=list(call.control_predicate_lines or []),
                        reachability_exact=call.reachability_exact,
                        symbol_bindings={},
                    )
                )
        return refs

    @staticmethod
    def _function_pointer_params(definition: Dict[str, Any]) -> Dict[str, int]:
        """Map pointer/reference parameter name to positional index."""
        names = definition.get("params") or []
        evidence = str(definition.get("evidence") or "")
        match = re.search(r"\(([^()]*)\)", evidence)
        if not match:
            return {}
        raw_params = match.group(1).split(",")
        out: Dict[str, int] = {}
        for index, raw in enumerate(raw_params):
            stripped = raw.strip()
            if not stripped or stripped == "void":
                continue
            if "*" not in stripped and "&" not in stripped:
                continue
            for name in names:
                if re.search(rf"[*&]\s*{re.escape(name)}\b", stripped):
                    out[name] = index
                    break
        return out

    _POINTER_WRITE_PATTERN = re.compile(
        r"(?P<param>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(?:->|\.)\s*"
        r"(?P<field>[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*"
        r"=(?!=)\s*(?P<expr>[^;]+);"
    )

    def _pointer_output_writes(
        self, body: str, pointer_params: Dict[str, int]
    ) -> List[Dict[str, str]]:
        writes: List[Dict[str, str]] = []
        for match in self._POINTER_WRITE_PATTERN.finditer(body):
            param = match.group("param")
            if param not in pointer_params:
                continue
            writes.append({
                "param": param,
                "field": self._clean_field_path(match.group("field")),
                "expression": self._normalize_source_expression(match.group("expr")),
            })
        return writes

    # Matches a leading C++ type declaration like ``const struct position_setpoint_s *``
    # so we can strip it from an argument text extracted from a function
    # definition line that the call-scanner misidentified as a call.
    _CXX_TYPE_PREFIX_RE = re.compile(
        r"^\s*(?:const\s+|volatile\s+|struct\s+|class\s+|enum\s+|unsigned\s+|signed\s+)*"
        r"[A-Za-z_][A-Za-z0-9_:<>]*"
        r"\s*[*&]\s*"
    )

    @staticmethod
    def _substitute_pointer_writes(
        writes: List[Dict[str, str]],
        pointer_params: Dict[str, int],
        call_args: List[str],
    ) -> List[Dict[str, str]]:
        """Strip C++ type prefixes from ``call_args`` then delegate to the
        DAG-owned :func:`derive_pointer_output_bindings`.

        Kept in the profiler as a thin adapter — the semantic ownership of
        pointer-output substitution lives in the DAG module so a helper
        called from many sites produces identical per-site bindings
        regardless of whether the flatten was driven by profiler
        extraction or by DAG graph-native emission.
        """
        # Lazy import so ``mechanism_source_profiler`` stays free of an
        # analysis-layer import at module load; the flatten path only
        # runs when a real call site is encountered.
        from flight_log_agent.analysis.mechanism_dag import (
            derive_pointer_output_bindings,
        )

        stripped_args: List[str] = []
        for arg in call_args:
            cleaned = str(arg).strip().lstrip("&").strip()
            cleaned = MechanismSourceProfiler._CXX_TYPE_PREFIX_RE.sub("", cleaned).strip()
            stripped_args.append(cleaned)
        return [
            {"target": entry["target"], "expression": entry["expression"]}
            for entry in derive_pointer_output_bindings(writes, pointer_params, stripped_args)
        ]

    def _extract_constant_definitions(
        self,
        text: str,
        rel_file: str,
    ) -> List[SourceAssignmentRef]:
        """Extract enum entries and object-like ``#define`` macros.

        Both forms become :class:`SourceAssignmentRef` records with
        ``target = NAME`` and ``expression = VALUE``, so the slicer and
        the predicate parser can look them up via the same
        ``assignment_resolutions`` path that handles ordinary
        assignments. Without this extraction, PX4-defined bitmask
        constants (e.g. ``STICK_CONFIG_ENABLE_AIRSPEED_SP_MANUAL_BIT``)
        appear as opaque identifiers and predicates referencing them
        stay unresolved.
        """
        cleaned = self._strip_block_comments_preserve_lines(text)
        refs: List[SourceAssignmentRef] = []

        for block in self._ENUM_BLOCK_PATTERN.finditer(cleaned):
            body = block.group("body")
            body_start = block.start("body")
            body_line_offset = cleaned[:body_start].count("\n")
            for entry in self._ENUM_ENTRY_PATTERN.finditer(body):
                name = entry.group("name").strip()
                value = entry.group("value").strip()
                if not name or not value:
                    continue
                line_no = body_line_offset + body[:entry.start()].count("\n") + 1
                refs.append(
                    SourceAssignmentRef(
                        target=name,
                        expression=value,
                        target_topic=None,
                        target_field=None,
                        function=None,
                        function_parameters=[],
                        assignment_operator="=",
                        declaration_kind="enum",
                        file=rel_file,
                        line=line_no,
                        evidence=f"{name} = {value}",
                        control_predicates=[],
                        symbol_bindings={},
                    )
                )

        for match in self._DEFINE_PATTERN.finditer(cleaned):
            name = match.group("name").strip()
            value = match.group("value").strip()
            if not name or not value:
                continue
            line_no = cleaned[:match.start()].count("\n") + 1
            refs.append(
                SourceAssignmentRef(
                    target=name,
                    expression=value,
                    target_topic=None,
                    target_field=None,
                    function=None,
                    function_parameters=[],
                    assignment_operator="=",
                    declaration_kind="define",
                    file=rel_file,
                    line=line_no,
                    evidence=match.group(0).strip(),
                    control_predicates=[],
                    symbol_bindings={},
                )
            )

        return refs

    _ELSE_BODY_PATTERN = re.compile(r"^else(?!\s*if\b)\s*\{")
    _ELSE_BRACELESS_PATTERN = re.compile(r"^else\b(?!\s*if\b)(?!\s*\{)\s*(?P<rest>.*)$")
    _ELSE_IF_PATTERN = re.compile(r"^else\s+if\b")
    _OPAQUE_BLOCK_PATTERN = re.compile(r"\b(?:for|do)\s*[\(\{]")
    _RETURN_STATEMENT_PATTERN = re.compile(r"\breturn\b")

    def _control_predicates_by_line(
        self, text: str
    ) -> Tuple[Dict[int, List[Tuple[str, int]]], Set[int]]:
        """Map each code line to its governing ``(predicate, site_line)``
        pairs — the site is the line of the control statement itself,
        the branch's source identity (two textually identical conditions
        at different sites are different branches).

        The second return value is the set of lines whose reachability is
        NOT exact: lines governed by a control construct this scan does
        not model (``switch``/``case``, loops) or following a
        conditionally nested early return. Their predicates list only
        what was derivable — consumers must mark them unresolved rather
        than present a partial predicate as exact.

        Guard clauses ARE modeled: an arm whose body contains a
        TOP-LEVEL ``return`` pushes the negation of its predicate over
        the remainder of the enclosing block (``if (bad) { return; }``
        gates everything after it with ``!(bad)``). A return nested
        deeper inside the arm returns only conditionally — the enclosing
        block's remainder is then explicitly non-exact, never guessed.
        """
        predicates_by_line: Dict[int, List[Tuple[str, int]]] = {}
        unresolved_lines: Set[int] = set()
        # Active arm entries are mutable records
        # [depth, combined predicate, site line, RAW condition, kind,
        #  has_top_level_return, has_nested_return]. The raw condition —
        # never the combined form — feeds sibling negation, so an
        # else-if chain emits ``!(A) && !(B) && (C)`` (mutually exclusive
        # siblings) instead of negating an already-combined arm. Kind
        # ``opaque`` marks a block whose reachability this scan cannot
        # model (switch, loops): every line it governs is non-exact.
        # Kind ``guard`` is a synthesized post-return negation.
        active: List[list] = []
        # Predicates whose ``{`` hasn't been seen yet — first-in first-out
        # so multiple pending ifs pop in the same order the parser saw them.
        pending: List[Tuple[str, int, str, str]] = []
        # Brace-less arms awaiting their single governed statement.
        braceless_pending: List[Tuple[str, int, str, str]] = []
        # Raw sibling conditions of the arms closed so far in the chain
        # at each depth: a plain ``if`` starts a fresh chain, ``else if``
        # extends it, ``else`` finishes it.
        chain_by_depth: Dict[int, List[str]] = {}
        brace_depth = 0
        lines = list(self._iter_code_lines(text))
        for index, (line_no, line) in enumerate(lines):
            stripped = line.strip()
            leading_closes = len(stripped) - len(stripped.lstrip("}"))
            if leading_closes:
                depth_after = max(brace_depth - leading_closes, 0)
                dropped = [entry for entry in active if entry[0] >= depth_after]
                for depth, _, _, raw, kind, _, _ in dropped:
                    if kind == "if":
                        chain_by_depth[depth] = [raw]
                    elif kind == "else if":
                        chain_by_depth.setdefault(depth, []).append(raw)
                    elif kind == "else":
                        chain_by_depth.pop(depth, None)
                brace_depth = depth_after
                active = [entry for entry in active if entry[0] < brace_depth]
                # Guard clauses: an arm that returned at its top level
                # gates the enclosing block's remainder with its
                # negation; a conditionally nested return makes that
                # remainder explicitly non-exact instead.
                for depth, predicate, site, _, kind, top_ret, nested_ret in dropped:
                    if kind not in {"if", "else if", "else"}:
                        continue
                    if top_ret and predicate:
                        active.append(
                            [max(depth - 1, 0), f"!({predicate})", site, "", "guard", False, False]
                        )
                    elif nested_ret:
                        active.append(
                            [max(depth - 1, 0), "", site, "", "opaque", False, False]
                        )

            def _negation_terms() -> str:
                return " && ".join(
                    f"!({term})" for term in chain_by_depth.get(brace_depth, [])
                )

            def _register_chain(kind: str, raw: str) -> None:
                if kind == "if":
                    chain_by_depth[brace_depth] = [raw]
                elif kind == "else if":
                    chain_by_depth.setdefault(brace_depth, []).append(raw)
                elif kind == "else":
                    chain_by_depth.pop(brace_depth, None)

            line_extras: List[Tuple[str, int]] = []
            is_control_line = False
            remainder_pre = stripped[leading_closes:].lstrip()

            # Allman-style ``{`` on its own line after a brace-less-looking
            # control: the block belongs to it — hand the arm to the
            # normal open processing.
            if braceless_pending and remainder_pre.startswith("{"):
                pending = braceless_pending + pending
                braceless_pending = []

            # Detect ``else`` (plain else) and push the conjunction of
            # every closed sibling's negation. The else's own line is the
            # synthesized branch's site. A brace-less else governs exactly
            # the next statement (or this line's own trailing statement).
            if self._ELSE_BODY_PATTERN.match(remainder_pre):
                negation = _negation_terms()
                if negation:
                    pending.append((negation, line_no, "", "else"))
            else:
                braceless_else = self._ELSE_BRACELESS_PATTERN.match(remainder_pre)
                if braceless_else is not None:
                    negation = _negation_terms()
                    if negation:
                        rest = braceless_else.group("rest").strip()
                        if rest and ";" in rest:
                            line_extras.append((negation, line_no))
                            _register_chain("else", "")
                            if self._RETURN_STATEMENT_PATTERN.search(rest):
                                active.append(
                                    [max(brace_depth - 1, 0), f"!({negation})",
                                     line_no, "", "guard", False, False]
                                )
                        else:
                            is_control_line = True
                            braceless_pending.append((negation, line_no, "", "else"))

            # Reconstruct multi-line if-conditions by joining continuation
            # lines until parens balance. Handles PX4's common:
            #     if (long_a
            #         && long_b) {
            # Without this, the regex's ``[^;\n]*`` condition class stops
            # at the first newline and the predicate is dropped.
            combined = line.split("//", 1)[0]
            paren_balance = combined.count("(") - combined.count(")")
            peek = index + 1
            while paren_balance > 0 and peek < len(lines):
                _, next_line = lines[peek]
                next_code = next_line.split("//", 1)[0].strip()
                combined += " " + next_code
                paren_balance += next_code.count("(") - next_code.count(")")
                peek += 1

            match = self._BRANCH_CONDITION_PATTERN.search(combined)
            branch_kind = " ".join(match.group("kind").split()) if match else ""
            if match and branch_kind in {"if", "else if"}:
                if "{" in combined[match.end():]:
                    condition = match.group("condition").strip()
                    if branch_kind == "else if":
                        negation = _negation_terms()
                        predicate = (
                            f"{negation} && ({condition})" if negation else condition
                        )
                    else:
                        predicate = condition
                    pending.append((predicate, line_no, condition, branch_kind))
                else:
                    # Brace-less arm(s): the control governs exactly its
                    # next statement. Nested brace-less controls on one
                    # line accumulate; innermost registers last, so a
                    # following else binds to the nearest unmatched if.
                    is_control_line = True
                    collected: List[Tuple[str, int, str, str]] = []
                    rest = combined
                    while True:
                        arm = self._BRANCH_CONDITION_PATTERN.search(rest)
                        arm_kind = " ".join(arm.group("kind").split()) if arm else ""
                        if not arm or arm_kind not in {"if", "else if"}:
                            break
                        condition = arm.group("condition").strip()
                        if arm_kind == "else if":
                            negation = _negation_terms()
                            predicate = (
                                f"{negation} && ({condition})"
                                if negation
                                else condition
                            )
                        else:
                            predicate = condition
                        collected.append((predicate, line_no, condition, arm_kind))
                        rest = rest[arm.end():]
                    statement = rest.strip()
                    if statement and ";" in statement:
                        # Same-line governed statement — pending outer
                        # arms attach too (nested brace-less), outermost
                        # registering first so a following else binds to
                        # the nearest unmatched if. A governed return is
                        # a guard: its arm's negation gates the rest of
                        # the enclosing block.
                        returns = bool(
                            self._RETURN_STATEMENT_PATTERN.search(statement)
                        )
                        for predicate, site, raw, kind in [
                            *braceless_pending,
                            *collected,
                        ]:
                            line_extras.append((predicate, site))
                            _register_chain(kind, raw)
                            if returns and predicate:
                                active.append(
                                    [max(brace_depth - 1, 0), f"!({predicate})",
                                     site, "", "guard", False, False]
                                )
                        braceless_pending = []
                    else:
                        braceless_pending.extend(collected)
            elif match and branch_kind in {"while", "switch"} and "{" in combined[match.end():]:
                # Reachability constructs this scan does not model: the
                # governed block is explicitly NON-exact, never silently
                # partial.
                pending.append(("", line_no, "", "opaque"))
            elif self._OPAQUE_BLOCK_PATTERN.search(combined) and "{" in combined:
                pending.append(("", line_no, "", "opaque"))

            # A plain statement consumes any pending brace-less arms —
            # they govern exactly this statement. A governed return is a
            # guard: its arm's negation gates the enclosing block's rest.
            if (
                braceless_pending
                and not is_control_line
                and stripped
                and not stripped.startswith("//")
                and not remainder_pre.startswith("{")
            ):
                returns = bool(self._RETURN_STATEMENT_PATTERN.search(stripped))
                for predicate, site, raw, kind in braceless_pending:
                    line_extras.append((predicate, site))
                    _register_chain(kind, raw)
                    if returns and predicate:
                        active.append(
                            [max(brace_depth - 1, 0), f"!({predicate})",
                             site, "", "guard", False, False]
                        )
                braceless_pending = []

            # Consume opens on THIS line: for each ``{``, pop a pending
            # predicate (if any) and push it as active at the current
            # brace_depth. Deferring the push until we see the ``{`` keeps
            # multi-line if-conditions from being filtered out at their
            # own end-of-line before the body has opened.
            remainder = stripped[leading_closes:]
            opens_here = remainder.count("{")
            pushed_this_line: List[list] = []
            for _ in range(opens_here):
                if pending:
                    predicate, site, raw, kind = pending.pop(0)
                    entry = [brace_depth, predicate, site, raw, kind, False, False]
                    active.append(entry)
                    pushed_this_line.append(entry)
                brace_depth += 1
            brace_depth = max(brace_depth - remainder.count("}"), 0)

            # Early-return bookkeeping for braced arms: a return at the
            # arm's immediate depth is top-level (the arm ALWAYS
            # returns); deeper is conditional. An arm opened-and-closed
            # on this same line owns any return on it.
            if (
                self._RETURN_STATEMENT_PATTERN.search(stripped)
                and not stripped.startswith("//")
            ):
                if pushed_this_line:
                    pushed_this_line[-1][5] = True
                else:
                    for entry in active:
                        if entry[4] in {"if", "else if", "else"}:
                            if brace_depth == entry[0] + 1:
                                entry[5] = True
                            else:
                                entry[6] = True

            # Snapshot AFTER remainder processing so assignments on the
            # same line as the ``{`` see the predicate. Deduped on the
            # predicate text: an else arm and the guard synthesized from
            # its returning sibling carry the same negation.
            snapshot: List[Tuple[str, int]] = []
            seen_predicates: set = set()
            for _, predicate, site, _, kind, _, _ in active:
                if kind == "opaque" or not predicate:
                    continue
                if predicate not in seen_predicates:
                    seen_predicates.add(predicate)
                    snapshot.append((predicate, site))
            for predicate, site in line_extras:
                if predicate not in seen_predicates:
                    seen_predicates.add(predicate)
                    snapshot.append((predicate, site))
            predicates_by_line[line_no] = snapshot
            if any(entry[4] == "opaque" for entry in active):
                unresolved_lines.add(line_no)

            active = [entry for entry in active if entry[0] < brace_depth]
        return predicates_by_line, unresolved_lines

    @staticmethod
    def _compound_assignment_expression(target: str, operator: str, expression: str) -> str:
        binary_operator = operator[:-1]
        if binary_operator:
            return f"{target} {binary_operator} ({expression})"
        return expression

    def extract_function_calls_from_source(
        self,
        files: Sequence[Union[str, Path]],
    ) -> List[FunctionCallRef]:
        expanded_files = self._expand_companion_files(files)
        cache_key = tuple(self._rel(path) for path in expanded_files)
        cached = self._function_call_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        refs: List[FunctionCallRef] = []
        ignored = {
            "if",
            "for",
            "while",
            "switch",
            "return",
            "sizeof",
            "catch",
            "static_cast",
            "reinterpret_cast",
            "const_cast",
            "dynamic_cast",
        }

        member_owners = self._collect_class_member_owners(expanded_files)
        for path in expanded_files:
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            var_to_struct = self._extract_struct_variables(text)
            definitions = self._extract_function_definitions(text, rel_file)
            control_predicates, unresolved_reach = self._control_predicates_by_line(text)
            code_lines = list(self._iter_code_lines(text))
            for index, (line_no, line) in enumerate(code_lines):
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                if self._FUNCTION_SIGNATURE_PATTERN.search(" ".join(stripped.split())):
                    continue
                definition = self._function_definition_for_line(definitions, line_no)
                function_name = str(definition.get("name") or "") if definition else None

                for match in self._FUNCTION_CALL_PATTERN.finditer(line):
                    name = match.group("name")
                    if name in ignored:
                        continue
                    receiver = self._call_receiver(line, match.start())
                    args = self._call_args(line, match.end() - 1)
                    if not args and self._matching_delimiter(
                        line, match.end() - 1, "(", ")"
                    ) is None:
                        # The argument list continues on later lines; join
                        # until it balances so multi-line calls keep args.
                        joined = line
                        for _, continuation in code_lines[index + 1 : index + 26]:
                            joined += " " + continuation.strip()
                            if self._matching_delimiter(
                                joined, match.end() - 1, "(", ")"
                            ) is not None:
                                break
                        args = self._call_args(joined, match.end() - 1)
                    argument_topics = self._argument_topics(args, var_to_struct)
                    argument_owners = self._argument_member_owners(
                        args, function_name, member_owners
                    )
                    predicate_entries = control_predicates.get(line_no, [])
                    predicates = [p for p, _ in predicate_entries]
                    refs.append(
                        FunctionCallRef(
                            name=name,
                            receiver=receiver,
                            args=args,
                            argument_topics=argument_topics,
                            file=rel_file,
                            line=line_no,
                            evidence=stripped,
                            control_predicates=predicates,
                            control_predicate_lines=[s for _, s in predicate_entries],
                            reachability_exact=line_no not in unresolved_reach,
                            symbol_bindings=self._source_symbol_bindings(
                                " ".join([stripped, *predicates]),
                                var_to_struct,
                            ),
                            argument_owners=argument_owners,
                            function=function_name,
                            callable_id=self._callable_id(definition),
                        )
                    )

        deduped = self._dedupe_function_call_refs(refs)
        self._function_call_cache[cache_key] = list(deduped)
        return deduped

    def extract_helper_expressions_from_source(
        self,
        files: Sequence[Union[str, Path]],
        helper_names: Optional[Sequence[str]] = None,
    ) -> List[HelperExpressionRef]:
        helper_name_set = {name for name in (helper_names or []) if name}
        refs: List[HelperExpressionRef] = []
        all_refs: List[HelperExpressionRef] = []
        expanded_files = self._expand_companion_files(files)
        member_to_param = self._collect_param_member_map(expanded_files)
        member_to_struct = self._collect_struct_member_map(expanded_files)

        for path in expanded_files:
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            for definition in self._extract_function_definitions(text, rel_file):
                name = definition["name"]
                short_name = name.split("::")[-1]
                translated = self._translate_helper_body(
                    name=name,
                    file=rel_file,
                    line=definition["line"],
                    params=definition["params"],
                    body=definition["body"],
                    evidence=definition["evidence"],
                    member_to_param=member_to_param,
                    member_to_struct=member_to_struct,
                    return_type=definition.get("return_type"),
                )
                all_refs.append(translated)
                if not helper_name_set or name in helper_name_set or short_name in helper_name_set:
                    refs.append(translated)

        return self._dedupe_helper_expression_refs(self._compose_helper_expressions(refs, all_refs))

    def extract_helper_expressions_recursive(
        self,
        files: Sequence[Union[str, Path]],
        helper_names: Optional[Sequence[str]] = None,
        *,
        max_passes: int = 3,
        max_candidate_files_per_callee: int = 2,
    ) -> List[HelperExpressionRef]:
        """Expand the helper scan to follow ``helper_calls`` across files.

        Closes the cross-file composition gap: the cone helper in
        ``rtl.cpp`` calls ``get_distance_to_next_waypoint`` defined in
        ``lib/geo/geo.cpp``, but a fresh scan of just ``rtl.cpp`` doesn't
        include the callee, so the lowered expression keeps the literal
        call. This method walks each helper's ``helper_calls`` list, finds
        candidate files containing those callees via the existing source
        search, re-extracts, and runs a final compose pass over the union
        so the cone helper inlines the haversine body even when the
        initial caller's scope didn't include the geo library.

        Bounded by ``max_passes`` so a callee chain doesn't run away;
        ``max_candidate_files_per_callee`` caps the per-callee search
        breadth (the source search is approximate — it returns files
        containing the name, not just the definition file).
        """
        initial_files = list(files)
        initial_names = list(helper_names or [])

        visited_files: set[str] = {self._rel(Path(f)) for f in initial_files}
        explored_callees: set[str] = set()

        current_files: list[Union[str, Path]] = initial_files
        current_names: Optional[list[str]] = initial_names if initial_names else None

        all_refs: List[HelperExpressionRef] = []

        for _ in range(max(1, max_passes)):
            pass_refs = self.extract_helper_expressions_from_source(current_files, current_names)
            all_refs.extend(pass_refs)
            # Mark helpers actually FOUND in this pass as explored so we
            # don't re-search files for them. Names in helper_names that
            # were NOT found stay unexplored so subsequent passes can
            # follow ``helper_calls`` to the file that defines them.
            for ref in pass_refs:
                if not ref.name:
                    continue
                explored_callees.add(ref.name)
                explored_callees.add(ref.name.split("::")[-1])

            new_callees: list[str] = []
            for ref in pass_refs:
                for call in ref.helper_calls or []:
                    short = call.split("::")[-1]
                    if not short:
                        continue
                    if short in explored_callees or call in explored_callees:
                        continue
                    if is_safe_math_function_name(call) or is_safe_math_function_name(short):
                        continue
                    explored_callees.add(call)
                    explored_callees.add(short)
                    new_callees.append(short)

            if not new_callees:
                break

            candidate_files: list[Union[str, Path]] = []
            for callee in new_callees:
                hits = self.search_related_source_files([callee], max_files=max_candidate_files_per_callee)
                for hit in hits:
                    if hit.file in visited_files:
                        continue
                    visited_files.add(hit.file)
                    candidate_files.append(hit.file)

            if not candidate_files:
                break

            current_files = candidate_files
            current_names = new_callees

        # Final compose pass so callees discovered in later passes inline
        # into helpers extracted in earlier passes.
        return self._dedupe_helper_expression_refs(
            self._compose_helper_expressions(all_refs, all_refs)
        )

    def extract_branch_conditions_from_source(
        self,
        files: Sequence[Union[str, Path]],
    ) -> List[BranchConditionRef]:
        refs: List[BranchConditionRef] = []

        for path in self._expand_companion_files(files):
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            for line_no, line in self._iter_code_lines(text):
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue

                for match in self._BRANCH_CONDITION_PATTERN.finditer(line):
                    refs.append(
                        BranchConditionRef(
                            kind=" ".join(match.group("kind").split()),
                            condition=match.group("condition").strip(),
                            file=rel_file,
                            line=line_no,
                            evidence=stripped,
                        )
                    )

                for match in self._CASE_CONDITION_PATTERN.finditer(line):
                    refs.append(
                        BranchConditionRef(
                            kind="case",
                            condition=match.group("condition").strip(),
                            file=rel_file,
                            line=line_no,
                            evidence=stripped,
                        )
                    )

        return self._dedupe_branch_condition_refs(refs)

    def extract_parameter_predicates_from_source(
        self,
        files: Sequence[Union[str, Path]],
    ) -> List[ParameterPredicateRef]:
        refs: List[ParameterPredicateRef] = []
        expanded_files = self._expand_companion_files(files)
        member_to_param = self._collect_param_member_map(expanded_files)

        for path in expanded_files:
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            for line_no, line in self._iter_code_lines(text):
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                if self._PARAM_DECL_PATTERN.search(stripped):
                    continue
                if not self._line_mentions_parameter(stripped, member_to_param):
                    continue

                for comparison in self._PARAM_COMPARISON_PATTERN.finditer(stripped):
                    name, member = self._resolve_parameter_operand(
                        comparison.group("left"),
                        member_to_param,
                    )
                    right_name, right_member = self._resolve_parameter_operand(
                        comparison.group("right"),
                        member_to_param,
                    )
                    if name is None and right_name is not None:
                        name = right_name
                        member = right_member

                    refs.append(
                        ParameterPredicateRef(
                            name=name,
                            member=member,
                            predicate=comparison.group(0).strip(),
                            operator=comparison.group("op"),
                            compared_value=comparison.group("right").strip(),
                            file=rel_file,
                            line=line_no,
                            evidence=stripped,
                        )
                    )

                if not refs or refs[-1].line != line_no or refs[-1].file != rel_file:
                    for member, name in member_to_param.items():
                        if member and member in stripped:
                            refs.append(
                                ParameterPredicateRef(
                                    name=name,
                                    member=member,
                                    predicate=stripped,
                                    file=rel_file,
                                    line=line_no,
                                    evidence=stripped,
                                )
                            )
                            break

        return self._dedupe_parameter_predicate_refs(refs)

    def profile_mechanism(
        self,
        queries: Union[str, Sequence[str]],
        max_files: int = 12,
        max_matches_per_file: int = 8,
    ) -> Dict[str, object]:
        """
        Convenience orchestration for mechanism discovery.

        Returns a JSON-serializable dictionary containing related files, uORB IO,
        parameter refs, and assigned fields.
        """
        query_text = "; ".join(self._normalize_queries(queries))
        hits = self.search_related_source_files(
            queries,
            max_files=max_files,
            max_matches_per_file=max_matches_per_file,
        )
        files = [hit.file for hit in hits]

        uorb = self.extract_uorb_io_from_source(files)
        params = self.extract_params_from_source(files)
        fields = self.extract_assigned_fields_from_source(files, relevance_terms=self._normalize_queries(queries))
        read_fields = self.extract_read_fields_from_source(files, relevance_terms=self._normalize_queries(queries))
        source_assignments = self.extract_source_assignments_from_source(files)
        function_calls = self.extract_function_calls_from_source(files)
        helper_expressions = self.extract_helper_expressions_from_source(
            files,
            helper_names=[ref.name for ref in function_calls],
        )
        branch_conditions = self.extract_branch_conditions_from_source(files)
        parameter_predicates = self.extract_parameter_predicates_from_source(files)

        notes: List[str] = []
        if not hits:
            notes.append("No related source files were found for the query.")
        if not params:
            notes.append("No PX4 parameter references were extracted from the selected files.")
        if not uorb["published_topics"] and not uorb["subscribed_topics"]:
            notes.append("No explicit uORB publish/subscribe declarations were extracted from the selected files.")

        profile = MechanismSourceProfile(
            query=query_text,
            source_root=self.source.identity,
            related_files=hits,
            published_topics=uorb["published_topics"],
            subscribed_topics=uorb["subscribed_topics"],
            unknown_direction_topics=uorb["unknown_direction_topics"],
            referenced_parameters=params,
            assigned_fields=fields,
            read_fields=read_fields,
            source_assignments=source_assignments,
            function_calls=function_calls,
            helper_expressions=helper_expressions,
            branch_conditions=branch_conditions,
            parameter_predicates=parameter_predicates,
            notes=notes,
        )
        return profile.model_dump()

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def _normalize_queries(
        self,
        queries: Union[str, Sequence[str]],
        *,
        expand_tokens: bool = True,
    ) -> List[str]:
        if isinstance(queries, str):
            raw = [queries]
        else:
            raw = list(queries)

        normalized: List[str] = []
        for query in raw:
            query = str(query).strip()
            if not query:
                continue
            if self._should_keep_query(query):
                normalized.append(query)

            if expand_tokens:
                # Also search individual strong-looking tokens for recall.
                for token in re.findall(r"[A-Z][A-Z0-9_]{2,}|[A-Za-z_][A-Za-z0-9_]{5,}", query):
                    if self._should_keep_query(token) and token not in normalized:
                        normalized.append(token)

        return normalized

    def _ripgrep_or_python_search(self, query: str) -> List[SourceMatch]:
        cached = self._search_cache.get(query)
        if cached is not None:
            return list(cached)
        try:
            matches = self._ripgrep_search(query)
        except Exception:
            matches = self._python_search(query)
        self._search_cache[query] = list(matches)
        return matches

    def _ripgrep_search(self, query: str) -> List[SourceMatch]:
        local_root = getattr(self.source, "root", None)
        if local_root is not None:
            command = [
                self.rg_path,
                "--line-number",
                "--no-heading",
                "--color=never",
                "--fixed-strings",
                "--ignore-case",
            ]
            for source_glob in self.SOURCE_GLOBS:
                command.extend(["--glob", source_glob])
            for excluded in self.excludes:
                command.extend(["--glob", f"!{excluded.rstrip('/')}/*"])
                command.extend(["--glob", f"!{excluded.rstrip('/')}*/**"])
            command.extend(["--", query, "."])
            completed = subprocess.run(
                command,
                cwd=Path(local_root),
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode not in {0, 1}:
                raise RuntimeError(completed.stderr.strip() or "ripgrep failed")
            matches: List[SourceMatch] = []
            for raw_line in completed.stdout.splitlines():
                file_name, separator, remainder = raw_line.partition(":")
                line_text, line_separator, text = remainder.partition(":")
                if not separator or not line_separator or not line_text.isdigit():
                    continue
                matches.append(
                    SourceMatch(
                        file=file_name.removeprefix("./"),
                        line=int(line_text),
                        text=text.strip(),
                        query=query,
                    )
                )
            return [match for match in matches if not self._is_excluded(match.file)]
        return [
            SourceMatch(file=match.file, line=match.line, text=match.text.strip(), query=query)
            for match in self.source.search(
                query,
                patterns=self.SOURCE_GLOBS,
                ignore_case=True,
                fixed_strings=True,
            )
            if not self._is_excluded(match.file)
        ]

    def _python_search(self, query: str) -> List[SourceMatch]:
        matches: List[SourceMatch] = []
        query_l = query.lower()

        for path in self._iter_source_files():
            text = self._read_text(path)
            if text is None:
                continue
            for line_no, line in self._iter_code_lines(text):
                if query_l in line.lower():
                    matches.append(
                        SourceMatch(
                            file=self._rel(path),
                            line=line_no,
                            text=line.strip(),
                            query=query,
                        )
                    )
        return matches

    def _score_match(self, match: SourceMatch, query: str) -> float:
        score = 1.0
        text_l = match.text.lower()
        query_l = query.lower()
        if query_l in text_l:
            score += 1.0
        if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", query):
            score += 2.0
        if "param" in text_l or "orb_" in text_l or "uorb::" in text_l:
            score += 0.5
        return score

    def _file_path_boost(self, rel_file: str) -> float:
        boost = 0.0
        path_l = rel_file.lower()
        if path_l.startswith("src/lib/") or path_l.startswith("src/modules/"):
            boost += 30.0
        elif path_l.startswith("src/drivers/") or path_l.startswith("src/include/"):
            boost += 15.0
        elif path_l.startswith("src/"):
            boost += 5.0
        if (
            path_l.startswith("test/")
            or "/test" in path_l
            or "_test" in path_l
            or "/unit" in path_l
            # CamelCase test files (FeasibilityCheckerTest.cpp) that the
            # lowercase patterns above miss. Matched on the original case
            # so e.g. "latest.cpp" doesn't false-positive.
            or re.search(r"Tests?\.(?:cpp|hpp|h)$", rel_file)
        ):
            boost -= 100.0
        if any(
            marker in path_l
            for marker in (
                "/catch2/",
                "/cmsis",
                "/libvnc/",
                "/third_party/",
                "/third-party/",
                "/vendor/",
                "/vendors/",
                "/external/",
                "/generated/",
                "/uavcan_drivers/",
            )
        ):
            boost -= 80.0
        if "/examples/" in path_l or path_l.startswith("examples/"):
            boost -= 40.0
        if "/build" in path_l or "/.git" in path_l:
            boost -= 5.0
        return boost

    def _should_keep_query(self, query: str) -> bool:
        query = query.strip()
        if not query:
            return False
        if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", query):
            return True
        if "/" in query or "\\" in query or "." in query:
            return True
        if " " in query:
            return True
        return query.lower() not in self.LOW_VALUE_QUERY_TOKENS

    # ------------------------------------------------------------------
    # Source reading/parsing helpers
    # ------------------------------------------------------------------

    def _iter_source_files(self) -> Iterable[Path]:
        for rel in self.source.list_files(patterns=self.SOURCE_GLOBS):
            if self._is_excluded(rel):
                continue
            yield Path(rel)

    def _is_excluded(self, rel_file: str) -> bool:
        rel_l = rel_file.lower()
        for exclude in self.excludes:
            exclude_l = exclude.lower().rstrip("/")
            if rel_l == exclude_l or rel_l.startswith(exclude_l + "/"):
                return True
        return False

    def _resolve_file(self, file_path: Union[str, Path]) -> Path:
        return Path(str(file_path).replace("\\", "/"))

    def _expand_companion_files(self, files: Sequence[Union[str, Path]]) -> List[Path]:
        """
        Include same-stem .h/.hpp/.cpp companions to catch declarations in
        headers and usage in implementation files. Also include one level of
        local quoted includes, which commonly hold parameter declarations for
        helper implementations.
        """
        expanded: Dict[str, Path] = {}
        for file_path in files:
            path = self._resolve_file(file_path)
            candidates = [path]

            if path.suffix.lower() in {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh", ".hxx"}:
                for suffix in (".h", ".hpp", ".hh", ".hxx", ".cpp", ".cc", ".cxx", ".c"):
                    companion = path.with_suffix(suffix)
                    candidates.append(companion)

            for candidate in candidates:
                if self.source.file_exists(candidate.as_posix()):
                    expanded[candidate.as_posix()] = candidate
                    for include in self._local_include_files(candidate):
                        expanded[include.as_posix()] = include

        return list(expanded.values())

    def _local_include_files(self, path: Path) -> List[Path]:
        text = self._read_text(path)
        if text is None:
            return []
        includes: List[Path] = []
        for line in text.splitlines():
            match = re.match(r'\s*#\s*include\s+"(?P<include>[^"]+)"', line)
            if not match:
                continue
            include_path = path.parent / match.group("include")
            if include_path.is_absolute() or ".." in include_path.parts:
                continue
            if self.source.file_exists(include_path.as_posix()):
                includes.append(include_path)
        return includes

    def _read_text(self, path: Path) -> Optional[str]:
        rel = self._rel(path)
        cached = self._text_cache.get(rel)
        if cached is not None or rel in self._text_cache:
            return cached
        try:
            text = self.source.read_text(rel, errors="ignore")
            if len(text.encode("utf-8")) > self.read_limit_bytes:
                text = None
        except Exception:
            text = None
        self._text_cache[rel] = text
        return text

    def _iter_code_lines(self, text: str) -> Iterable[Tuple[int, str]]:
        # Strip block comments lightly. This is deliberately simple and avoids
        # pretending to be a full C++ parser.
        text = self._strip_block_comments_preserve_lines(text)
        for idx, line in enumerate(text.splitlines(), start=1):
            yield idx, line

    def _extract_struct_variables(self, text: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for _, line in self._iter_code_lines(text):
            for pattern in self._STRUCT_VAR_PATTERNS:
                for match in pattern.finditer(line):
                    struct = match.group("struct")
                    var = match.group("var")
                    mapping[var] = struct
        return mapping

    def _extract_class_definitions(self, text: str) -> List[Dict[str, object]]:
        """Return named class/struct body spans with nested ownership."""
        stripped = self._strip_block_comments_preserve_lines(text)
        definitions: List[Dict[str, object]] = []
        for match in self._CLASS_DECL_PATTERN.finditer(stripped):
            open_brace = stripped.find("{", match.start(), match.end())
            close_brace = self._matching_brace(stripped, open_brace)
            if open_brace < 0 or close_brace is None:
                continue
            definitions.append(
                {
                    "name": match.group("name"),
                    "line": stripped.count("\n", 0, match.start()) + 1,
                    "end_line": stripped.count("\n", 0, close_brace) + 1,
                    "open_index": open_brace,
                    "end_index": close_brace,
                    "bases": self._parse_base_classes(match.group("bases") or ""),
                }
            )
        for definition in definitions:
            containers = [
                candidate
                for candidate in definitions
                if int(candidate["open_index"]) < int(definition["open_index"])
                and int(candidate["end_index"]) > int(definition["end_index"])
            ]
            if containers:
                parent = min(
                    containers,
                    key=lambda candidate: int(candidate["end_index"])
                    - int(candidate["open_index"]),
                )
                definition["name"] = f"{parent['name']}::{definition['name']}"
        return definitions

    @staticmethod
    def _parse_base_classes(raw_bases: str) -> List[str]:
        bases: List[str] = []
        for raw in split_top_level_args(raw_bases):
            cleaned = re.sub(r"\b(?:public|protected|private|virtual)\b", "", raw)
            cleaned = " ".join(cleaned.split()).strip()
            if not cleaned:
                continue
            match = re.search(r"([A-Za-z_][A-Za-z0-9_:]*(?:\s*<.*>)?)\s*$", cleaned)
            if match:
                bases.append("".join(match.group(1).split()))
        return bases

    @staticmethod
    def _member_declarations_from_body(
        body: str,
        *,
        owner: str,
        file: str,
        first_line: int,
    ) -> List[SourceMemberRef]:
        """Extract direct data members from a class body.

        Nested class/function bodies have already been blanked by
        :meth:`_direct_class_body`. The remaining semicolon statements are
        declarations at class scope. Function declarations and type aliases
        are rejected structurally instead of using member-name conventions.
        """
        refs: List[SourceMemberRef] = []
        body = "\n".join(line.split("//", 1)[0] for line in body.splitlines())
        offset = 0
        for statement in body.split(";")[:-1]:
            statement_line = first_line + body.count("\n", 0, offset)
            offset += len(statement) + 1
            cleaned = re.sub(r"\b(?:public|protected|private)\s*:\s*", "", statement)
            cleaned = " ".join(cleaned.split()).strip()
            if not cleaned or cleaned.startswith(
                (
                    "using ",
                    "typedef ",
                    "friend ",
                    "static_assert",
                    "enum ",
                    "class ",
                    "struct ",
                    "#",
                )
            ):
                continue
            # A top-level parenthesis before any initializer denotes a method
            # declaration, macro invocation, or static assertion, not data.
            paren = cleaned.find("(")
            initializer = min(
                [index for index in (cleaned.find("="), cleaned.find("{")) if index >= 0]
                or [len(cleaned)]
            )
            if 0 <= paren < initializer:
                continue
            declaration = cleaned[:initializer].strip()
            first = re.match(
                r"(?P<type>.+?)(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
                r"\s*(?:\[[^\]]*\])?\s*$",
                declaration,
            )
            if not first:
                continue
            type_name = first.group("type").strip().rstrip("*& ")
            name = first.group("name")
            if not type_name or type_name.endswith(("=", ",")):
                continue
            refs.append(
                SourceMemberRef(
                    name=name,
                    owner=owner,
                    type=type_name,
                    file=file,
                    line=statement_line,
                )
            )
        return refs

    @staticmethod
    def _function_parameter_types(raw_params: str) -> List[str]:
        types: List[str] = []
        for raw in split_top_level_args(raw_params):
            param = raw.split("=", 1)[0].strip()
            if not param or param == "void":
                continue
            param = re.sub(r"\[[^\]]*\]", "", param).strip()
            match = re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*$", param)
            if match:
                param = param[: match.start()].strip()
            types.append(" ".join(param.split()))
        return types

    def extract_source_structure_from_source(
        self,
        files: Sequence[Union[str, Path]],
        *,
        expand_companions: bool = True,
    ) -> Dict[str, List[BaseModel]]:
        """Extract source relationships used by deterministic expansion.

        This is a source index, not a mechanism guess: class ownership,
        inheritance, declarations, callable signatures, and include edges are
        all emitted from syntax with stable source sites.
        """
        paths = self._expand_companion_files(files) if expand_companions else [
            self._resolve_file(file_path) for file_path in files
        ]
        classes: List[SourceClassRef] = []
        members: List[SourceMemberRef] = []
        callables: List[SourceCallableRef] = []
        includes: List[SourceIncludeRef] = []
        for path in paths:
            text = self._read_text(path)
            if text is None:
                continue
            rel_file = self._rel(path)
            stripped = self._strip_block_comments_preserve_lines(text)
            definitions = self._extract_class_definitions(stripped)
            for definition in definitions:
                owner = str(definition.get("name") or "")
                line = int(definition.get("line") or 0)
                end_line = int(definition.get("end_line") or line)
                classes.append(
                    SourceClassRef(
                        name=owner,
                        file=rel_file,
                        line=line,
                        end_line=end_line,
                        bases=[str(base) for base in definition.get("bases") or []],
                    )
                )
                start = int(definition.get("open_index") or 0) + 1
                end = int(definition.get("end_index") or start)
                direct_body = self._direct_class_body(stripped[start:end])
                members.extend(
                    self._member_declarations_from_body(
                        direct_body,
                        owner=owner,
                        file=rel_file,
                        first_line=line,
                    )
                )
            for definition in self._extract_function_definitions(stripped, rel_file):
                name = str(definition.get("name") or "")
                owner = self._function_owner(name)
                evidence = str(definition.get("evidence") or "")
                signature_match = re.search(r"\((?P<params>.*)\)\s*(?:const\s*)?\{$", evidence)
                callables.append(
                    SourceCallableRef(
                        name=name,
                        owner=owner,
                        file=rel_file,
                        line=int(definition.get("line") or 0),
                        end_line=int(definition.get("end_line") or 0),
                        callable_id=str(self._callable_id(definition) or ""),
                        parameters=[str(value) for value in definition.get("params") or []],
                        parameter_types=self._function_parameter_types(
                            signature_match.group("params") if signature_match else ""
                        ),
                        return_type=(
                            str(definition.get("return_type"))
                            if definition.get("return_type")
                            else None
                        ),
                    )
                )
            for line_no, line_text in enumerate(stripped.splitlines(), start=1):
                match = re.match(r'\s*#\s*include\s*[<"](?P<path>[^>"]+)[>"]', line_text)
                if not match:
                    continue
                include = match.group("path")
                candidates = [path.parent / include, Path(include)]
                resolved = next(
                    (
                        candidate.as_posix()
                        for candidate in candidates
                        if not candidate.is_absolute()
                        and ".." not in candidate.parts
                        and self.source.file_exists(candidate.as_posix())
                    ),
                    None,
                )
                if resolved:
                    includes.append(
                        SourceIncludeRef(
                            file=rel_file,
                            included_file=resolved,
                            line=line_no,
                        )
                    )
        return {
            "classes": classes,
            "members": members,
            "callables": callables,
            "includes": includes,
        }

    @staticmethod
    def _class_owner_for_line(
        definitions: Sequence[Dict[str, object]], line_no: int
    ) -> Optional[str]:
        candidates = [
            definition
            for definition in definitions
            if int(definition.get("line") or 0) <= line_no
            <= int(definition.get("end_line") or 0)
        ]
        if not candidates:
            return None
        owner = min(
            candidates,
            key=lambda definition: int(definition.get("end_line") or 0)
            - int(definition.get("line") or 0),
        )
        return str(owner.get("name") or "") or None

    @staticmethod
    def _direct_class_body(body: str) -> str:
        """Blank nested bodies while preserving direct member declarations."""
        out: List[str] = []
        depth = 0
        for char in body:
            if char == "{":
                depth += 1
                out.append(" ")
            elif char == "}":
                depth = max(depth - 1, 0)
                out.append(" ")
            elif depth == 0:
                out.append(char)
            else:
                out.append("\n" if char == "\n" else " ")
        return "".join(out)

    def _extract_class_member_owners(self, text: str) -> Dict[str, Set[str]]:
        mapping: Dict[str, Set[str]] = {}
        stripped = self._strip_block_comments_preserve_lines(text)
        for definition in self._extract_class_definitions(stripped):
            start = int(definition.get("open_index") or 0) + 1
            end = int(definition.get("end_index") or start)
            direct_body = self._direct_class_body(stripped[start:end])
            owner = str(definition.get("name") or "")
            for member in self._member_declarations_from_body(
                direct_body,
                owner=owner,
                file="",
                first_line=int(definition.get("line") or 0),
            ):
                mapping.setdefault(member.name, set()).add(owner)
        return mapping

    def _collect_class_member_owners(
        self, files: Sequence[Path]
    ) -> Dict[str, Set[str]]:
        mapping: Dict[str, Set[str]] = {}
        for path in files:
            text = self._read_text(path)
            if text is None:
                continue
            for variable, owners in self._extract_class_member_owners(text).items():
                mapping.setdefault(variable, set()).update(owners)
        return mapping

    @staticmethod
    def _function_owner(function_name: Optional[str]) -> Optional[str]:
        text = str(function_name or "")
        owner, separator, _method = text.rpartition("::")
        return owner if separator else None

    def _argument_member_owners(
        self,
        args: Sequence[str],
        function_name: Optional[str],
        member_owners: Dict[str, Set[str]],
    ) -> Dict[str, str]:
        function_owner = self._function_owner(function_name)
        if not function_owner:
            return {}
        resolved: Dict[str, str] = {}
        for arg in args:
            symbol = self._boundary_argument_symbol(arg)
            if not symbol:
                continue
            root = symbol.replace("->", ".").split(".", 1)[0]
            matching = {
                owner
                for owner in member_owners.get(root, set())
                if owner == function_owner or function_owner.endswith(f"::{owner}")
            }
            if len(matching) == 1:
                resolved[root] = function_owner
        return resolved

    @classmethod
    def _topic_from_orb_reference(cls, expression: str) -> Optional[str]:
        matches = list(cls._ORB_ID_REFERENCE_PATTERN.finditer(str(expression or "")))
        topics = {
            match.group("call") or match.group("scope") for match in matches
        }
        return next(iter(topics)) if len(topics) == 1 else None

    @classmethod
    def _boundary_argument_symbol(cls, expression: str) -> Optional[str]:
        text = str(expression or "").strip()
        text = re.sub(
            r"^\(\s*[A-Za-z_][A-Za-z0-9_:<>\s*&]*\s*\)\s*", "", text
        )
        text = text.lstrip("&* ").strip()
        text = cls._clean_field_path(text)
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:(?:\.|->)[A-Za-z_][A-Za-z0-9_]*)*",
            text,
        ):
            return None
        return text

    def _collect_param_member_map(self, files: Sequence[Path]) -> Dict[str, str]:
        member_to_param: Dict[str, str] = {}
        for path in files:
            text = self._read_text(path)
            if text is None:
                continue
            for _, line in self._iter_code_lines(text):
                for match in self._PARAM_DECL_PATTERN.finditer(line):
                    name = match.group("name")
                    member = match.groupdict().get("member")
                    if member:
                        member_to_param[member] = name
                        member_to_param[member.lstrip("_ ")] = name
        return member_to_param

    def _collect_struct_member_map(self, files: Sequence[Path]) -> Dict[str, str]:
        member_to_struct: Dict[str, str] = {}
        for path in files:
            text = self._read_text(path)
            if text is None:
                continue
            member_to_struct.update(self._extract_struct_variables(text))
        return member_to_struct

    def _line_mentions_parameter(self, line: str, member_to_param: Dict[str, str]) -> bool:
        if self._PX4_PARAM_PATTERN.search(line) or self._PARAM_FIND_PATTERN.search(line):
            return True
        return any(member and member in line for member in member_to_param)

    def _resolve_parameter_operand(
        self,
        operand: str,
        member_to_param: Dict[str, str],
    ) -> Tuple[Optional[str], Optional[str]]:
        operand = operand.strip()
        px4_match = self._PX4_PARAM_PATTERN.search(operand)
        if px4_match:
            return px4_match.group("name"), None

        if operand.endswith(".get()"):
            member_expr = operand[:-6].strip()
            member = member_expr.split(".")[-1]
            return member_to_param.get(member_expr) or member_to_param.get(member), member_expr

        member = operand.split(".")[-1]
        return member_to_param.get(operand) or member_to_param.get(member), operand if operand in member_to_param else None

    def _extract_function_definitions(self, text: str, rel_file: str) -> List[Dict[str, object]]:
        text = self._strip_block_comments_preserve_lines(text)
        definitions: List[Dict[str, object]] = []
        class_definitions = self._extract_class_definitions(text)
        ignored = {"if", "for", "while", "switch", "catch"}
        for open_brace in (match.start() for match in re.finditer(r"\{", text)):
            signature_start = max(
                text.rfind(";", 0, open_brace),
                text.rfind("{", 0, open_brace),
                text.rfind("}", 0, open_brace),
            ) + 1
            raw_signature = text[signature_start:open_brace].strip()
            signature = " ".join(raw_signature.split())
            match = self._FUNCTION_SIGNATURE_PATTERN.search(signature)
            if not match:
                continue
            name = match.group("name")
            if name in ignored:
                continue
            close_brace = self._matching_brace(text, open_brace)
            if close_brace is None:
                continue
            line_start = signature_start + len(text[signature_start:open_brace]) - len(text[signature_start:open_brace].lstrip())
            line_no = text.count("\n", 0, line_start) + 1
            lexical_owner = self._class_owner_for_line(class_definitions, line_no)
            if lexical_owner and "::" not in name:
                name = f"{lexical_owner}::{name}"
            definitions.append(
                {
                    "name": name,
                    "line": line_no,
                    "end_line": text.count("\n", 0, close_brace) + 1,
                    "start_index": match.start(),
                    "end_index": close_brace,
                    "params": self._parse_function_parameters(match.group("params")),
                    "body": text[open_brace + 1:close_brace],
                    "evidence": f"{signature} {{",
                    "file": rel_file,
                    "return_type": self._normalize_return_type(match.group("prefix")),
                }
            )
        return definitions

    # Storage / access qualifiers we drop before the actual return type.
    _RETURN_TYPE_QUALIFIERS = re.compile(
        r"\b(?:inline|static|virtual|explicit|constexpr|const|volatile|"
        r"__attribute__\s*\(\([^)]*\)\)|extern|friend|typename)\b"
    )

    @staticmethod
    def _normalize_return_type(prefix: str) -> Optional[str]:
        """Strip C++ storage qualifiers from a signature prefix to expose
        the raw return type.

        Given ``"static const vehicle_status_s *"`` returns
        ``"vehicle_status_s *"``. Preserves pointer/reference marks and
        template angle brackets so downstream pattern matching (``foo_s *``
        as a topic reference) can recover the topic name unambiguously.
        """
        cleaned = MechanismSourceProfiler._RETURN_TYPE_QUALIFIERS.sub("", prefix or "")
        cleaned = " ".join(cleaned.split())
        return cleaned or None

    @staticmethod
    def _strip_block_comments_preserve_lines(text: str) -> str:
        return re.sub(
            r"/\*.*?\*/",
            lambda match: "\n" * match.group(0).count("\n"),
            text,
            flags=re.DOTALL,
        )

    @staticmethod
    def _function_name_for_line(definitions: List[Dict[str, object]], line_no: int) -> Optional[str]:
        definition = MechanismSourceProfiler._function_definition_for_line(definitions, line_no)
        return str(definition.get("name") or "") if definition else None

    @staticmethod
    def _function_definition_for_line(definitions: List[Dict[str, object]], line_no: int) -> Optional[Dict[str, object]]:
        for definition in definitions:
            start_line = int(definition.get("line") or 0)
            end_line = int(definition.get("end_line") or 0)
            if start_line <= line_no <= end_line:
                return definition
        return None

    @staticmethod
    def _callable_id(definition: Optional[Dict[str, object]]) -> Optional[str]:
        if not definition:
            return None
        return ":".join(
            [
                str(definition.get("file") or ""),
                str(definition.get("line") or 0),
                str(definition.get("name") or ""),
                ",".join(str(p) for p in (definition.get("params") or [])),
            ]
        )

    @staticmethod
    def _matching_brace(text: str, open_brace: int) -> Optional[int]:
        depth = 0
        for index in range(open_brace, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        return None

    @staticmethod
    def _parse_function_parameters(params: str) -> List[str]:
        parsed: List[str] = []
        for raw_param in params.split(","):
            param = raw_param.strip()
            if not param or param == "void":
                continue
            param = param.split("=", 1)[0].strip()
            param = re.sub(r"\[[^\]]*\]", "", param)
            match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", param.replace("*", " ").replace("&", " "))
            if match:
                parsed.append(match.group(1))
        return parsed

    def _call_args(self, line: str, open_paren: int) -> List[str]:
        close_paren = self._matching_delimiter(line, open_paren, "(", ")")
        if close_paren is None:
            return []
        return [
            self._normalize_call_arg(arg)
            for arg in split_top_level_args(line[open_paren + 1:close_paren])
            if arg.strip()
        ]

    @staticmethod
    def _normalize_call_arg(arg: str) -> str:
        arg = arg.strip().lstrip("&")
        arg = MechanismSourceProfiler._clean_field_path(arg)
        return MechanismSourceProfiler._normalize_source_expression(arg)

    @staticmethod
    def _argument_topics(args: List[str], var_to_struct: Dict[str, str]) -> Dict[str, str]:
        topics: Dict[str, str] = {}
        for arg in args:
            root, _ = split_source_field(arg)
            struct = var_to_struct.get(root)
            if struct:
                topics[arg] = MechanismSourceProfiler._topic_from_struct(struct) or ""
        return {arg: topic for arg, topic in topics.items() if topic}

    @staticmethod
    def _normalize_source_expression(expr: str) -> str:
        expr = MechanismSourceProfiler._clean_field_path(expr.strip())
        return MechanismSourceProfiler._normalize_helper_expression(expr)

    def _translate_helper_body(
        self,
        *,
        name: str,
        file: str,
        line: int,
        params: List[str],
        body: str,
        evidence: str,
        member_to_param: Dict[str, str],
        member_to_struct: Dict[str, str],
        return_type: Optional[str] = None,
    ) -> HelperExpressionRef:
        cleaned_body = self._strip_line_comments(body)
        unresolved = self._unsupported_helper_body_reason(cleaned_body)
        statements = self._helper_statements(cleaned_body)
        assignments = self._helper_assignments(cleaned_body)
        branches = self._helper_return_branches(cleaned_body)
        return_expression = self._helper_return_expression(cleaned_body)
        lowered_return_expression: Optional[str] = None
        if unresolved is None:
            try:
                lowered_return_expression = self._lower_helper_statements(statements)
            except _HelperLoweringFailed as exc:
                unresolved = exc.reason
                lowered_return_expression = None
        helper_calls = self._helper_body_calls(cleaned_body)
        symbol_bindings = self._helper_symbol_bindings(cleaned_body, member_to_param, member_to_struct)
        call_resolutions = self._helper_call_resolutions(cleaned_body, member_to_param)

        pointer_params = self._function_pointer_params({"params": params, "evidence": evidence})
        pointer_writes = self._pointer_output_writes(cleaned_body, pointer_params) if pointer_params else []
        # Class-member struct types + locals declared in the body — both
        # get surfaced on the record so the DAG can derive
        # ``var.field → topic.field`` graph-natively.
        struct_variables: Dict[str, str] = dict(member_to_struct or {})
        struct_variables.update(self._extract_struct_variables(cleaned_body))
        if unresolved is None and not return_expression and not lowered_return_expression and not branches:
            if not pointer_writes:
                unresolved = "helper has no return value and no pointer-output writes routable through source_assignments"
        if unresolved is None and len(re.findall(r"\breturn\b", cleaned_body)) > 1 and not branches and not lowered_return_expression:
            unresolved = "helper has multiple return paths that lowering could not combine into a single expression"

        return HelperExpressionRef(
            name=name,
            file=file,
            line=line,
            evidence=evidence,
            callable_id=":".join(
                [file, str(line), name, ",".join(str(param) for param in params)]
            ),
            parameters=params,
            statements=statements if unresolved is None else [],
            assignments=assignments if unresolved is None else {},
            return_expression=return_expression if unresolved is None else None,
            lowered_return_expression=lowered_return_expression if unresolved is None else None,
            branches=branches if unresolved is None else [],
            symbol_bindings=symbol_bindings if unresolved is None else {},
            call_resolutions=call_resolutions,
            helper_calls=helper_calls,
            pointer_output_writes=pointer_writes if unresolved is None else [],
            return_type=return_type,
            struct_variables=struct_variables if unresolved is None else {},
            unresolved_reason=unresolved,
        )

    @staticmethod
    def _strip_line_comments(text: str) -> str:
        return "\n".join(line.split("//", 1)[0] for line in text.splitlines())

    def _helper_statements(self, body: str) -> List[Dict[str, Any]]:
        return self._parse_helper_statement_block(body)

    def _parse_helper_statement_block(self, text: str) -> List[Dict[str, Any]]:
        statements: List[Dict[str, Any]] = []
        index = 0
        while index < len(text):
            index = self._skip_helper_whitespace(text, index)
            if index >= len(text):
                break
            if self._keyword_at(text, index, "if"):
                parsed_if, index = self._parse_helper_if(text, index)
                if parsed_if:
                    statements.append(parsed_if)
                continue
            if self._keyword_at(text, index, "switch"):
                parsed_switch, index = self._parse_helper_switch(text, index)
                if parsed_switch:
                    statements.append(parsed_switch)
                continue
            if self._keyword_at(text, index, "for"):
                parsed_for, index = self._parse_helper_for(text, index)
                if parsed_for:
                    statements.append(parsed_for)
                continue
            if self._keyword_at(text, index, "while"):
                parsed_while, index = self._parse_helper_while(text, index)
                if parsed_while:
                    statements.append(parsed_while)
                continue
            if self._keyword_at(text, index, "do"):
                parsed_do, index = self._parse_helper_do_while(text, index)
                if parsed_do:
                    statements.append(parsed_do)
                continue
            semicolon = self._find_statement_semicolon(text, index)
            if semicolon is None:
                break
            statement = self._parse_helper_simple_statement(text[index:semicolon].strip())
            if statement:
                statements.append(statement)
            index = semicolon + 1
        return statements

    _SWITCH_LABEL_PATTERN = re.compile(r"\b(?P<kind>case\s+(?P<label>[^:]+?)|default)\s*:")

    def _parse_helper_switch(self, text: str, index: int) -> Tuple[Optional[Dict[str, Any]], int]:
        cursor = self._skip_helper_whitespace(text, index + 6)
        if cursor >= len(text) or text[cursor] != "(":
            return None, index + 6
        close_paren = self._matching_delimiter(text, cursor, "(", ")")
        if close_paren is None:
            return None, index + 6
        discriminant = self._normalize_helper_condition(text[cursor + 1:close_paren])
        cursor = self._skip_helper_whitespace(text, close_paren + 1)
        if cursor >= len(text) or text[cursor] != "{":
            return None, cursor
        close_brace = self._matching_delimiter(text, cursor, "{", "}")
        if close_brace is None:
            return None, cursor
        body_text = text[cursor + 1:close_brace]
        cases, default = self._parse_switch_cases(body_text)
        return {
            "kind": "switch",
            "discriminant": discriminant,
            "cases": cases,
            "default": default,
        }, close_brace + 1

    def _parse_switch_cases(
        self, body: str
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        cases: List[Dict[str, Any]] = []
        default: Optional[List[Dict[str, Any]]] = None
        labels = list(self._SWITCH_LABEL_PATTERN.finditer(body))
        if not labels:
            return cases, default
        pending_conditions: List[str] = []
        for i, label in enumerate(labels):
            is_default = label.group("kind").strip().startswith("default")
            next_start = labels[i + 1].start() if i + 1 < len(labels) else len(body)
            section = body[label.end():next_start]
            has_terminator = bool(re.search(r"\b(?:break|return)\b", section))
            stripped = re.sub(r"\bbreak\s*;\s*$", "", section.strip()).strip()
            statements = self._parse_helper_statement_block(stripped)
            if is_default:
                default = statements
                pending_conditions = []
                continue
            label_text = label.group("label").strip() if label.group("label") else ""
            if label_text:
                pending_conditions.append(label_text)
            if has_terminator or stripped:
                cases.append({"conditions": list(pending_conditions), "body": statements})
                pending_conditions = []
        return cases, default

    def _parse_helper_if(self, text: str, index: int) -> Tuple[Optional[Dict[str, Any]], int]:
        cursor = self._skip_helper_whitespace(text, index + 2)
        if cursor >= len(text) or text[cursor] != "(":
            return None, index + 2
        close_paren = self._matching_delimiter(text, cursor, "(", ")")
        if close_paren is None:
            return None, index + 2
        condition = self._normalize_helper_condition(text[cursor + 1:close_paren])
        cursor = self._skip_helper_whitespace(text, close_paren + 1)
        if cursor >= len(text) or text[cursor] != "{":
            return None, cursor
        close_then = self._matching_delimiter(text, cursor, "{", "}")
        if close_then is None:
            return None, cursor
        then_statements = self._parse_helper_statement_block(text[cursor + 1:close_then])

        cursor = self._skip_helper_whitespace(text, close_then + 1)
        else_statements: List[Dict[str, Any]] = []
        if self._keyword_at(text, cursor, "else"):
            cursor = self._skip_helper_whitespace(text, cursor + 4)
            if self._keyword_at(text, cursor, "if"):
                nested_if, cursor = self._parse_helper_if(text, cursor)
                if nested_if:
                    else_statements = [nested_if]
            elif cursor < len(text) and text[cursor] == "{":
                close_else = self._matching_delimiter(text, cursor, "{", "}")
                if close_else is not None:
                    else_statements = self._parse_helper_statement_block(text[cursor + 1:close_else])
                    cursor = close_else + 1

        return {
            "kind": "if",
            "condition": condition,
            "then": then_statements,
            "else": else_statements,
        }, cursor

    @staticmethod
    def _parse_helper_simple_statement(statement: str) -> Optional[Dict[str, Any]]:
        if not statement:
            return None
        inc_dec_match = re.fullmatch(
            r"\s*(?:(?P<pre_op>\+\+|--)\s*(?P<pre_var>[A-Za-z_][A-Za-z0-9_]*)"
            r"|(?P<post_var>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<post_op>\+\+|--))\s*",
            statement,
        )
        if inc_dec_match:
            var = inc_dec_match.group("pre_var") or inc_dec_match.group("post_var")
            op = inc_dec_match.group("pre_op") or inc_dec_match.group("post_op")
            return {
                "kind": "assign",
                "target": var,
                "operator": "+=" if op == "++" else "-=",
                "expression": MechanismSourceProfiler._compound_assignment_expression(
                    var, "+=" if op == "++" else "-=", "1"
                ),
            }
        return_match = re.match(r"\breturn\s+(?P<expr>.+)$", statement, flags=re.DOTALL)
        if return_match:
            return {
                "kind": "return",
                "expression": MechanismSourceProfiler._normalize_helper_expression(return_match.group("expr")),
            }
        declaration_match = re.match(
            r"(?:const\s+)?(?:auto|float|double|int|bool|uint\d+_t|int\d+_t|[A-Za-z_][A-Za-z0-9_:<>]*)"
            r"\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<expr>.+)$",
            statement,
            flags=re.DOTALL,
        )
        if declaration_match:
            return {
                "kind": "declare",
                "target": declaration_match.group("name"),
                "expression": MechanismSourceProfiler._normalize_helper_expression(declaration_match.group("expr")),
            }
        compound_assignment_match = re.match(
            r"(?P<target>[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*)*)"
            r"\s*(?P<op>\+=|-=|\*=|/=|%=|\|=|&=|\^=)\s*(?P<expr>.+)$",
            statement,
            flags=re.DOTALL,
        )
        if compound_assignment_match:
            target = MechanismSourceProfiler._clean_field_path(compound_assignment_match.group("target"))
            expression = MechanismSourceProfiler._compound_assignment_expression(
                target,
                compound_assignment_match.group("op"),
                MechanismSourceProfiler._normalize_helper_expression(compound_assignment_match.group("expr")),
            )
            return {
                "kind": "assign",
                "target": target,
                "expression": expression,
                "operator": compound_assignment_match.group("op"),
            }
        assignment_match = re.match(
            r"(?P<target>[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*)*)"
            r"\s*=\s*(?P<expr>.+)$",
            statement,
            flags=re.DOTALL,
        )
        if assignment_match:
            return {
                "kind": "assign",
                "target": MechanismSourceProfiler._clean_field_path(assignment_match.group("target")),
                "expression": MechanismSourceProfiler._normalize_helper_expression(assignment_match.group("expr")),
            }
        return None

    @staticmethod
    def _skip_helper_whitespace(text: str, index: int) -> int:
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    @staticmethod
    def _keyword_at(text: str, index: int, keyword: str) -> bool:
        if index < 0 or index + len(keyword) > len(text) or text[index:index + len(keyword)] != keyword:
            return False
        before_ok = index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_")
        after_index = index + len(keyword)
        after_ok = after_index == len(text) or not (text[after_index].isalnum() or text[after_index] == "_")
        return before_ok and after_ok

    @staticmethod
    def _matching_delimiter(text: str, open_index: int, open_char: str, close_char: str) -> Optional[int]:
        depth = 0
        for index in range(open_index, len(text)):
            char = text[index]
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return index
        return None

    @staticmethod
    def _find_statement_semicolon(text: str, start: int) -> Optional[int]:
        paren_depth = 0
        brace_depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(paren_depth - 1, 0)
            elif char == "{":
                brace_depth += 1
            elif char == "}":
                if brace_depth == 0:
                    return None
                brace_depth -= 1
            elif char == ";" and paren_depth == 0 and brace_depth == 0:
                return index
        return None

    def _lower_helper_statements(self, statements: List[Dict[str, Any]]) -> Optional[str]:
        env: Dict[str, str] = {}
        return self._lower_helper_statement_block(statements, env)

    def _lower_helper_statement_block(
        self,
        statements: List[Dict[str, Any]],
        env: Dict[str, str],
    ) -> Optional[str]:
        for index, statement in enumerate(statements):
            kind = statement.get("kind")
            if kind in {"declare", "assign"}:
                target = str(statement.get("target") or "")
                expression = self._substitute_helper_locals(str(statement.get("expression") or ""), env)
                if target:
                    self._guarded_env_write(env, target, expression)
            elif kind == "return":
                return self._substitute_helper_locals(str(statement.get("expression") or ""), env)
            elif kind == "if":
                condition = self._substitute_helper_locals(str(statement.get("condition") or ""), env)
                then_env = dict(env)
                else_env = dict(env)
                then_return = self._lower_helper_statement_block(
                    list(statement.get("then") or []),
                    then_env,
                )
                else_return = self._lower_helper_statement_block(
                    list(statement.get("else") or []),
                    else_env,
                )
                if then_return is not None and else_return is not None:
                    return f"({then_return} if {condition} else {else_return})"
                if then_return is not None:
                    continuation = self._lower_helper_statement_block(
                        statements[index + 1:],
                        else_env,
                    )
                    if continuation is not None:
                        return f"({then_return} if {condition} else {continuation})"
                if else_return is not None:
                    continuation = self._lower_helper_statement_block(
                        statements[index + 1:],
                        then_env,
                    )
                    if continuation is not None:
                        return f"({continuation} if {condition} else {else_return})"
                for target in sorted(set(then_env) | set(else_env)):
                    before = env.get(target, target)
                    then_expr = then_env.get(target, before)
                    else_expr = else_env.get(target, before)
                    if then_expr != before or else_expr != before:
                        self._guarded_env_write(
                            env,
                            target,
                            f"({then_expr} if {condition} else {else_expr})",
                        )
            elif kind == "switch":
                lowered = self._lower_switch_statement(statement, statements[index + 1:], env)
                if lowered is not None:
                    return lowered
            elif kind == "for":
                early = self._lower_for_statement(statement, env)
                if early is not None:
                    return early
            elif kind == "for_range":
                header = str(statement.get("header") or "").strip()
                collection = header.split(":", 1)[1].strip() if ":" in header else header
                raise _HelperLoweringFailed(
                    f"helper body uses range-based for over '{collection}' which cannot be enumerated statically"
                )
            elif kind == "while":
                early = self._lower_while_statement(statement, env)
                if early is not None:
                    return early
            elif kind == "do_while":
                early = self._lower_do_while_statement(statement, env)
                if early is not None:
                    return early
        return None

    _WHILE_RUNAWAY = 65536
    # Hard ceiling on any single lowered expression. The if-merge embeds the
    # running value into BOTH ternary branches, so state accumulated across
    # unrolled loop iterations grows ~2^N (mission.cpp read_mission_item's
    # DO_JUMP loop reached hundreds of MB before the process died). Lowering
    # ABORTS via _HelperLoweringFailed — the helper stays opaque with a
    # reason — rather than continuing with a truncated/capped result, which
    # would be semantically wrong for loops that advance per-iteration state.
    _LOWERED_EXPRESSION_LIMIT = 32768

    def _guarded_env_write(self, env: Dict[str, str], target: str, expression: str) -> None:
        if len(expression) > self._LOWERED_EXPRESSION_LIMIT:
            raise _HelperLoweringFailed(
                f"lowered expression for '{target}' exceeded "
                f"{self._LOWERED_EXPRESSION_LIMIT} characters; accumulated "
                "per-iteration state cannot be flattened statically"
            )
        env[target] = expression

    def _lower_while_statement(
        self,
        statement: Dict[str, Any],
        env: Dict[str, str],
    ) -> Optional[str]:
        condition_text = str(statement.get("condition") or "").strip()
        body = list(statement.get("body") or [])
        if not condition_text:
            raise _HelperLoweringFailed("while loop condition is empty")
        for _ in range(self._WHILE_RUNAWAY):
            cond_value = self._evaluate_condition(condition_text, env, "while loop condition")
            if not cond_value:
                return None
            result = self._lower_helper_statement_block(body, env)
            if result is not None:
                return result
        raise _HelperLoweringFailed(
            f"while loop with condition '{condition_text}' did not terminate within {self._WHILE_RUNAWAY} iterations"
        )

    def _lower_do_while_statement(
        self,
        statement: Dict[str, Any],
        env: Dict[str, str],
    ) -> Optional[str]:
        condition_text = str(statement.get("condition") or "").strip()
        body = list(statement.get("body") or [])
        if not condition_text:
            raise _HelperLoweringFailed("do-while loop condition is empty")
        for _ in range(self._WHILE_RUNAWAY):
            result = self._lower_helper_statement_block(body, env)
            if result is not None:
                return result
            cond_value = self._evaluate_condition(condition_text, env, "do-while loop condition")
            if not cond_value:
                return None
        raise _HelperLoweringFailed(
            f"do-while loop with condition '{condition_text}' did not terminate within {self._WHILE_RUNAWAY} iterations"
        )

    def _evaluate_condition(self, condition_text: str, env: Dict[str, str], context: str) -> bool:
        from flight_log_agent.analysis.safe_eval import eval_const_expression

        substituted = self._substitute_helper_locals(condition_text, env)
        value = eval_const_expression(substituted)
        if value is None:
            raise _HelperLoweringFailed(
                f"{context} '{condition_text}' is not statically resolvable"
            )
        return bool(value)

    def _lower_for_statement(
        self,
        statement: Dict[str, Any],
        env: Dict[str, str],
    ) -> Optional[str]:
        from flight_log_agent.analysis.safe_eval import eval_const_expression

        init = statement.get("init") or {}
        condition_text = str(statement.get("condition") or "").strip()
        increment = statement.get("increment") or {}
        init_text = str(statement.get("init_text") or "").strip()
        inc_text = str(statement.get("increment_text") or "").strip()
        body = list(statement.get("body") or [])

        iter_var = init.get("target")
        init_value_text = init.get("expression")
        if not iter_var:
            raise _HelperLoweringFailed(
                f"for loop init '{init_text}' did not parse as a declaration"
            )
        if init_value_text is None:
            raise _HelperLoweringFailed(
                f"for loop init '{init_text}' has no initializer expression"
            )
        if not condition_text:
            raise _HelperLoweringFailed("for loop condition is empty")
        increment_target = increment.get("target")
        if increment_target != iter_var:
            # Real source doesn't infinite-loop; treat the loop as a no-op and let the outer block continue.
            return None
        step_text = str(increment.get("expression") or "1")
        step_value = eval_const_expression(self._substitute_helper_locals(step_text, env))
        if step_value is None:
            raise _HelperLoweringFailed(
                f"for loop increment '{inc_text}' is not statically resolvable"
            )
        step = int(step_value) * (-1 if increment.get("operator") == "-=" else 1)
        if step == 0:
            return None

        start_value = eval_const_expression(self._substitute_helper_locals(str(init_value_text), env))
        if start_value is None:
            raise _HelperLoweringFailed(
                f"for loop init expression '{init_value_text}' is not statically resolvable"
            )
        start = int(start_value)

        cond_match = re.fullmatch(
            rf"\s*{re.escape(iter_var)}\s*(?P<op><|<=|>|>=|!=)\s*(?P<bound>.+?)\s*",
            self._substitute_helper_locals(condition_text, env),
        )
        if not cond_match:
            raise _HelperLoweringFailed(
                f"for loop condition '{condition_text}' does not have form '{iter_var} op literal'"
            )
        op = cond_match.group("op")
        bound_expr = cond_match.group("bound").strip()
        bound_value = eval_const_expression(bound_expr)
        if bound_value is None:
            raise _HelperLoweringFailed(
                f"for loop bound '{bound_expr}' is not statically resolvable"
            )
        bound = int(bound_value)

        # Iteration count is fully determined by literal init / cond / inc;
        # walk every iteration the source actually requests.
        values: List[int] = []
        i = start
        while True:
            if op == "<" and not (i < bound):
                break
            if op == "<=" and not (i <= bound):
                break
            if op == ">" and not (i > bound):
                break
            if op == ">=" and not (i >= bound):
                break
            if op == "!=" and not (i != bound):
                break
            values.append(i)
            i += step

        for value in values:
            env[iter_var] = str(value)
            result = self._lower_helper_statement_block(body, env)
            if result is not None:
                env.pop(iter_var, None)
                return result
        env.pop(iter_var, None)
        return None

    def _lower_switch_statement(
        self,
        statement: Dict[str, Any],
        continuation_statements: List[Dict[str, Any]],
        env: Dict[str, str],
    ) -> Optional[str]:
        discriminant = self._substitute_helper_locals(
            str(statement.get("discriminant") or ""), env
        )
        cases = list(statement.get("cases") or [])
        default_statements = list(statement.get("default") or [])
        if not cases:
            return None
        case_results: List[Tuple[List[str], Optional[str]]] = []
        for case in cases:
            case_env = dict(env)
            case_return = self._lower_helper_statement_block(
                list(case.get("body") or []), case_env
            )
            case_results.append((list(case.get("conditions") or []), case_return))
        default_env = dict(env)
        default_return = (
            self._lower_helper_statement_block(default_statements, default_env)
            if default_statements
            else None
        )
        chain = default_return
        if chain is None and default_statements:
            return None
        if chain is None:
            chain = self._lower_helper_statement_block(
                continuation_statements, dict(env)
            )
        if chain is None:
            return None
        for conditions, case_return in reversed(case_results):
            if case_return is None or not conditions:
                return None
            condition_expr = self._switch_condition_expression(discriminant, conditions)
            chain = f"({case_return} if {condition_expr} else {chain})"
        return chain

    @staticmethod
    def _switch_condition_expression(discriminant: str, conditions: List[str]) -> str:
        if len(conditions) == 1:
            return f"{discriminant} == {conditions[0]}"
        return " or ".join(f"({discriminant} == {c})" for c in conditions)

    _FOR_INCREMENT_PATTERN = re.compile(
        r"\s*(?:(?P<pre_op>\+\+|--)\s*(?P<pre_var>[A-Za-z_][A-Za-z0-9_]*)"
        r"|(?P<post_var>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<post_op>\+\+|--))\s*"
    )

    def _parse_helper_for(self, text: str, index: int) -> Tuple[Optional[Dict[str, Any]], int]:
        cursor = self._skip_helper_whitespace(text, index + 3)
        if cursor >= len(text) or text[cursor] != "(":
            return None, index + 3
        close_paren = self._matching_delimiter(text, cursor, "(", ")")
        if close_paren is None:
            return None, index + 3
        header = text[cursor + 1:close_paren]
        if ":" in header and ";" not in header:
            cursor = self._skip_helper_whitespace(text, close_paren + 1)
            if cursor < len(text) and text[cursor] == "{":
                close_brace = self._matching_delimiter(text, cursor, "{", "}")
                if close_brace is not None:
                    return {"kind": "for_range", "header": header.strip(), "body": []}, close_brace + 1
            return {"kind": "for_range", "header": header.strip(), "body": []}, close_paren + 1
        parts = header.split(";")
        if len(parts) != 3:
            return None, close_paren + 1
        init_text, cond_text, inc_text = (p.strip() for p in parts)
        init = self._parse_helper_simple_statement(init_text) if init_text else None
        increment = self._parse_helper_increment(inc_text)
        cursor = self._skip_helper_whitespace(text, close_paren + 1)
        if cursor >= len(text) or text[cursor] != "{":
            return None, cursor
        close_brace = self._matching_delimiter(text, cursor, "{", "}")
        if close_brace is None:
            return None, cursor
        body = self._parse_helper_statement_block(text[cursor + 1:close_brace])
        return {
            "kind": "for",
            "init": init,
            "init_text": init_text,
            "condition": cond_text,
            "increment": increment,
            "increment_text": inc_text,
            "body": body,
        }, close_brace + 1

    def _parse_helper_while(self, text: str, index: int) -> Tuple[Optional[Dict[str, Any]], int]:
        cursor = self._skip_helper_whitespace(text, index + 5)
        if cursor >= len(text) or text[cursor] != "(":
            return None, index + 5
        close_paren = self._matching_delimiter(text, cursor, "(", ")")
        if close_paren is None:
            return None, index + 5
        condition = self._normalize_helper_condition(text[cursor + 1:close_paren])
        cursor = self._skip_helper_whitespace(text, close_paren + 1)
        if cursor >= len(text) or text[cursor] != "{":
            return None, cursor
        close_brace = self._matching_delimiter(text, cursor, "{", "}")
        if close_brace is None:
            return None, cursor
        body = self._parse_helper_statement_block(text[cursor + 1:close_brace])
        return {"kind": "while", "condition": condition, "body": body}, close_brace + 1

    def _parse_helper_do_while(self, text: str, index: int) -> Tuple[Optional[Dict[str, Any]], int]:
        cursor = self._skip_helper_whitespace(text, index + 2)
        if cursor >= len(text) or text[cursor] != "{":
            return None, cursor
        close_brace = self._matching_delimiter(text, cursor, "{", "}")
        if close_brace is None:
            return None, cursor
        body = self._parse_helper_statement_block(text[cursor + 1:close_brace])
        cursor = self._skip_helper_whitespace(text, close_brace + 1)
        if not self._keyword_at(text, cursor, "while"):
            return None, cursor
        cursor = self._skip_helper_whitespace(text, cursor + 5)
        if cursor >= len(text) or text[cursor] != "(":
            return None, cursor
        close_paren = self._matching_delimiter(text, cursor, "(", ")")
        if close_paren is None:
            return None, cursor
        condition = self._normalize_helper_condition(text[cursor + 1:close_paren])
        cursor = close_paren + 1
        semicolon = self._find_statement_semicolon(text, cursor)
        end = (semicolon + 1) if semicolon is not None else cursor
        return {"kind": "do_while", "condition": condition, "body": body}, end

    def _parse_helper_increment(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        if not text:
            return None
        match = self._FOR_INCREMENT_PATTERN.fullmatch(text)
        if match:
            var = match.group("pre_var") or match.group("post_var")
            op = match.group("pre_op") or match.group("post_op")
            return {
                "target": var,
                "operator": "+=" if op == "++" else "-=",
                "expression": "1",
            }
        compound_match = re.fullmatch(
            r"\s*(?P<target>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op>\+=|-=)\s*(?P<step>.+?)\s*",
            text,
        )
        if compound_match:
            return {
                "target": compound_match.group("target"),
                "operator": compound_match.group("op"),
                "expression": self._normalize_helper_expression(compound_match.group("step")),
            }
        return self._parse_helper_simple_statement(text + ";")

    def _compose_helper_expressions(
        self,
        refs: List[HelperExpressionRef],
        all_refs: List[HelperExpressionRef],
    ) -> List[HelperExpressionRef]:
        helper_index = self._composable_helper_index(all_refs)
        for ref in refs:
            stack = {ref.name}
            if ref.return_expression:
                ref.return_expression = self._apply_parameter_symbol_bindings(
                    ref.return_expression,
                    ref.symbol_bindings,
                )
                ref.return_expression = self._compose_helper_expression(ref.return_expression, helper_index, stack)
            if ref.lowered_return_expression:
                ref.lowered_return_expression = self._apply_parameter_symbol_bindings(
                    ref.lowered_return_expression,
                    ref.symbol_bindings,
                )
                ref.lowered_return_expression = self._compose_helper_expression(
                    ref.lowered_return_expression,
                    helper_index,
                    stack,
                )
            ref.call_resolutions = self._mark_composed_helper_calls(
                ref.call_resolutions,
                helper_index,
                current_name=ref.name,
            )
        return refs

    def _composable_helper_index(self, refs: List[HelperExpressionRef]) -> Dict[str, HelperExpressionRef]:
        full_names: Dict[str, HelperExpressionRef] = {}
        short_names: Dict[str, List[HelperExpressionRef]] = {}
        for ref in refs:
            if not self._helper_composable_expression(ref):
                continue
            full_names[ref.name] = ref
            short_names.setdefault(ref.name.split("::")[-1], []).append(ref)

        out = dict(full_names)
        for short_name, matches in short_names.items():
            if len(matches) == 1:
                out[short_name] = matches[0]
        return out

    def _helper_composable_expression(self, ref: HelperExpressionRef) -> Optional[str]:
        if ref.unresolved_reason:
            return None
        expression = ref.lowered_return_expression or ref.return_expression
        if not expression:
            return None
        return self._apply_parameter_symbol_bindings(expression, ref.symbol_bindings)

    def _compose_helper_expression(
        self,
        expression: str,
        helper_index: Dict[str, HelperExpressionRef],
        stack: set[str],
    ) -> str:
        out: List[str] = []
        index = 0
        for match in self._FUNCTION_CALL_PATTERN.finditer(expression):
            name = match.group("name")
            short_name = name.split("::")[-1]
            if is_safe_math_function_name(name):
                continue
            callee = helper_index.get(name) or helper_index.get(short_name)
            if callee is None or callee.name in stack:
                continue
            open_paren = match.end() - 1
            close_paren = self._matching_delimiter(expression, open_paren, "(", ")")
            if close_paren is None:
                continue

            callee_expression = self._helper_composable_expression(callee)
            if not callee_expression:
                continue
            args = [
                self._normalize_call_arg(arg)
                for arg in split_top_level_args(expression[open_paren + 1:close_paren])
            ]
            inlined = substitute_expression_symbols(callee_expression, callee.parameters, args)
            inlined = self._compose_helper_expression(inlined, helper_index, {*stack, callee.name})
            replacement_start = self._helper_call_replacement_start(expression, index, match.start())
            out.append(expression[index:replacement_start])
            out.append(f"({inlined})")
            index = close_paren + 1
        out.append(expression[index:])
        return "".join(out)

    @staticmethod
    def _helper_call_replacement_start(expression: str, start: int, call_start: int) -> int:
        prefix = expression[start:call_start]
        receiver = re.search(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\.$", prefix)
        if receiver:
            return start + receiver.start()
        return call_start

    @staticmethod
    def _apply_parameter_symbol_bindings(expression: str, bindings: Dict[str, str]) -> str:
        parameter_bindings = {
            source: target
            for source, target in bindings.items()
            if re.match(r"^[A-Z][A-Z0-9_]*$", target or "")
        }
        return substitute_expression_symbols(
            expression,
            list(parameter_bindings.keys()),
            list(parameter_bindings.values()),
        )

    def _mark_composed_helper_calls(
        self,
        resolutions: List[Dict[str, Any]],
        helper_index: Dict[str, HelperExpressionRef],
        *,
        current_name: str,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for resolution in resolutions:
            call = str(resolution.get("call") or "")
            name = call[:-2] if call.endswith("()") else call
            short_name = re.split(r"::|\.", name)[-1]
            callee = helper_index.get(name) or helper_index.get(short_name)
            if resolution.get("kind") in {"source_helper_candidate", "unresolved_runtime_call"} and (
                callee is not None and callee.name != current_name
            ):
                expression = self._helper_composable_expression(callee)
                updated = dict(resolution)
                updated["kind"] = "translated_pure_helper"
                if expression:
                    updated["expression"] = expression
                out.append(updated)
            else:
                out.append(resolution)
        return out

    @staticmethod
    def _substitute_helper_locals(expression: str, env: Dict[str, str]) -> str:
        substituted = expression
        for name, value in sorted(env.items(), key=lambda item: len(item[0]), reverse=True):
            substituted = re.sub(
                rf"(?<![A-Za-z0-9_\.]){re.escape(name)}(?![A-Za-z0-9_\.])",
                f"({value})",
                substituted,
            )
        return substituted

    def _unsupported_helper_body_reason(self, body: str) -> Optional[str]:
        if re.search(r"\bgoto\b", body):
            return "helper body uses goto"
        return None

    def _helper_assignments(self, body: str) -> Dict[str, str]:
        assignments: Dict[str, str] = {}
        pattern = re.compile(
            r"(?:^|;|\n)\s*(?:const\s+)?(?:auto|float|double|int|bool|uint\d+_t|int\d+_t|[A-Za-z_][A-Za-z0-9_:<>]*)"
            r"\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<expr>.*?);",
            re.DOTALL,
        )
        for match in pattern.finditer(body):
            assignments[match.group("name")] = self._normalize_helper_expression(match.group("expr"))
        return assignments

    def _helper_return_expression(self, body: str) -> Optional[str]:
        matches = re.findall(r"\breturn\s+(?P<expr>.*?)\s*;", body, flags=re.DOTALL)
        if len(matches) != 1:
            return None
        return self._normalize_helper_expression(matches[0])

    def _helper_return_branches(self, body: str) -> List[Dict[str, str]]:
        branches: List[Dict[str, str]] = []
        pattern = re.compile(
            r"\bif\s*\((?P<condition>[^{};]+)\)\s*\{\s*return\s+(?P<true_expr>[^;]+);\s*\}"
            r"(?:\s*else\s*\{\s*return\s+(?P<false_expr>[^;]+);\s*\})?",
            re.DOTALL,
        )
        for match in pattern.finditer(body):
            branches.append(
                {
                    "condition": self._normalize_helper_expression(match.group("condition")),
                    "expression": self._normalize_helper_expression(match.group("true_expr")),
                }
            )
            false_expr = match.group("false_expr")
            if false_expr:
                branches.append(
                    {
                        "condition": f"!({self._normalize_helper_expression(match.group('condition'))})",
                        "expression": self._normalize_helper_expression(false_expr),
                    }
                )
        return branches

    def _helper_body_calls(self, body: str) -> List[str]:
        ignored = {"if", "return", "static_cast", "const_cast", "reinterpret_cast", "dynamic_cast"}
        names: List[str] = []
        for match in self._FUNCTION_CALL_PATTERN.finditer(body):
            name = match.group("name")
            short_name = name.split("::")[-1]
            if short_name in ignored:
                continue
            names.append(name)
        return list(dict.fromkeys(names))

    def _helper_symbol_bindings(
        self,
        body: str,
        member_to_param: Dict[str, str],
        member_to_struct: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        bindings: Dict[str, str] = {}
        for match in self._PARAM_GET_MEMBER_PATTERN.finditer(body):
            member_expr = re.sub(r"\s+", "", match.group("member"))
            member_name = member_expr.split(".")[-1]
            param_name = member_to_param.get(member_expr) or member_to_param.get(member_name)
            if param_name:
                bindings[f"{member_expr}.get()"] = param_name

        var_to_struct = dict(member_to_struct or {})
        var_to_struct.update(self._extract_struct_variables(body))
        for match in self._FIELD_ACCESS_PATTERN.finditer(body):
            var = match.group("var")
            field_name = self._clean_field_path(match.group("field"))
            struct = var_to_struct.get(var)
            topic = self._topic_from_struct(struct) if struct else None
            if topic:
                bindings[f"{var}.{field_name}"] = f"{topic}.{field_name}"
        return bindings

    def _source_symbol_bindings(
        self,
        expression: str,
        var_to_struct: Dict[str, str],
    ) -> Dict[str, str]:
        bindings: Dict[str, str] = {}
        for match in self._FIELD_ACCESS_PATTERN.finditer(expression):
            var = match.group("var")
            field_name = self._clean_field_path(match.group("field"))
            struct = var_to_struct.get(var)
            topic = self._topic_from_struct(struct) if struct else None
            if topic:
                bindings[f"{var}.{field_name}"] = f"{topic}.{field_name}"
        return bindings

    def _helper_call_resolutions(self, body: str, member_to_param: Dict[str, str]) -> List[Dict[str, Any]]:
        resolutions: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()
        member_call_pattern = re.compile(
            r"(?P<receiver>[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*)*)"
            r"\s*(?P<access>\.|->)\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
        )
        for match in member_call_pattern.finditer(body):
            receiver = self._clean_field_path(match.group("receiver"))
            name = match.group("name")
            call = f"{receiver}.{name}"
            param_name = member_to_param.get(receiver) or member_to_param.get(receiver.split(".")[-1])
            if name == "get" and param_name:
                resolution = {
                    "call": f"{call}()",
                    "kind": "parameter_accessor",
                    "parameter": param_name,
                }
            else:
                resolution = {
                    "call": f"{call}()",
                    "kind": "unresolved_runtime_call",
                }
            key = (resolution["call"], resolution["kind"])
            if key not in seen:
                seen.add(key)
                resolutions.append(resolution)

        for name in self._helper_body_calls(body):
            short_name = name.split("::")[-1]
            if short_name == "get":
                continue
            canonical_name = canonical_math_function_name(name)
            if is_safe_math_function_name(name):
                resolution = {"call": f"{name}()", "kind": "math_function", "canonical_name": canonical_name}
            else:
                resolution = {"call": f"{name}()", "kind": "source_helper_candidate"}
            key = (resolution["call"], resolution["kind"])
            if key not in seen:
                seen.add(key)
                resolutions.append(resolution)
        return resolutions

    @staticmethod
    def _normalize_helper_expression(expr: str) -> str:
        expr = MechanismSourceProfiler._clean_field_path(expr.strip())
        cast_match = re.match(r"\bstatic_cast\s*<[^>]+>\s*\(", expr)
        if cast_match and expr.endswith(")"):
            expr = expr[cast_match.end():-1].strip()
        expr = re.sub(r"\bstatic_cast\s*<[^>]+>\s*\(([^()]+)\)", r"\1", expr)
        expr = re.sub(r"\b([0-9]+(?:\.[0-9]+)?)f\b", r"\1", expr)
        expr = normalize_expression_function_names(expr)
        expr = " ".join(expr.split())
        expr = re.sub(r"\(\s+", "(", expr)
        expr = re.sub(r"\s+\)", ")", expr)
        return expr

    @staticmethod
    def _normalize_helper_condition(expr: str) -> str:
        expr = MechanismSourceProfiler._normalize_helper_expression(expr)
        expr = expr.replace("&&", " and ").replace("||", " or ")
        expr = re.sub(r"!(?!=)", "not ", expr)
        return " ".join(expr.split())

    @staticmethod
    def _call_receiver(line: str, call_start: int) -> Optional[str]:
        prefix = line[:call_start].rstrip()
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\.|->)\s*$", prefix)
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _topic_from_struct(struct: Optional[str]) -> Optional[str]:
        if not struct:
            return None
        return struct[:-2] if struct.endswith("_s") else struct

    @staticmethod
    def _struct_from_topic(topic: str) -> str:
        return f"{topic}_s"

    @staticmethod
    def _clean_field_path(field_text: str) -> str:
        cleaned = re.sub(r"\s*(?:\.|->)\s*", ".", field_text.strip())
        return cleaned[5:] if cleaned.startswith("this.") else cleaned

    @classmethod
    def _extract_reference_aliases_per_function(
        cls,
        definitions: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        """Per-function map ``{function_name: {var: aliased_path}}``.

        Only aliases whose RHS resolves to a field projection on another
        struct (``container.field`` / ``container->field`` / chained getters
        ending in ``->field``) are kept. The dereferenced-getter pattern
        ``Type &name = *getter();`` is intentionally *excluded* — the
        existing struct-var path handles those cleanly via the struct type
        and the alias would only obscure the resolution.
        """
        aliases_per_function: Dict[str, Dict[str, str]] = {}
        for definition in definitions:
            function_name = str(definition.get("name") or "")
            body = str(definition.get("body") or "")
            if not function_name or not body:
                continue
            func_aliases: Dict[str, str] = {}
            for match in cls._REFERENCE_ALIAS_PATTERN.finditer(body):
                name = match.group("name")
                raw_expr = match.group("expr").strip()
                if not cls._alias_rhs_is_field_projection(raw_expr):
                    continue
                func_aliases[name] = cls._clean_field_path(raw_expr.lstrip("*&"))
            if func_aliases:
                aliases_per_function[function_name] = func_aliases
        return aliases_per_function

    @classmethod
    def _is_alias_declaration_line(cls, line: str) -> bool:
        return bool(cls._REFERENCE_ALIAS_PATTERN.match(line.strip()))

    @classmethod
    def _alias_rhs_is_field_projection(cls, rhs: str) -> bool:
        """Whether ``rhs`` ends in a member access without a trailing call.

        Used to filter alias declarations: only ``Type &name = <expr>.field``
        or ``Type &name = ...->field`` style declarations get aliased. A
        plain ``Type &name = *getter()`` is left alone so the struct-var
        path continues to bind ``name.X`` via the struct type.
        """
        return bool(cls._ALIAS_TRAILING_MEMBER_ACCESS_RE.search(rhs))

    @staticmethod
    def _apply_reference_alias(path: str, aliases: Dict[str, str]) -> str:
        """If ``path``'s leading identifier is in ``aliases``, substitute it.

        Returns ``path`` unchanged when no alias applies. Substitution is
        textual on the leading dotted identifier; the rest of the path is
        preserved verbatim so e.g. ``curr_sp.lat`` becomes
        ``<alias>.lat`` and ``curr_sp.acceptance_radius`` becomes
        ``<alias>.acceptance_radius``.
        """
        if not aliases or not path:
            return path
        head, sep, tail = path.partition(".")
        if head in aliases:
            replacement = aliases[head]
            return f"{replacement}.{tail}" if tail else replacement
        return path

    def _rel(self, path: Path) -> str:
        return path.as_posix()

    @staticmethod
    def _looks_like_param_member(member: str) -> bool:
        member_l = member.lower()
        return (
            member_l.startswith("_param")
            or member_l.startswith("param")
            or member_l.startswith("params")
            or "_param_" in member_l
        )

    def _dynamic_relevance_terms(self, raw_terms: Sequence[str]) -> set[str]:
        terms: set[str] = set()
        for raw in raw_terms:
            text = str(raw).strip().lower()
            if not text:
                continue
            normalized = re.sub(r"[^a-z0-9_./]+", " ", text)
            for token in normalized.split():
                if not self._is_relevance_token(token):
                    continue
                terms.add(token)
                if "." in token:
                    terms.update(part for part in token.split(".") if self._is_relevance_token(part))
                if "/" in token:
                    terms.update(part for part in token.split("/") if self._is_relevance_token(part))
        return terms

    def _matches_dynamic_relevance(self, var: str, field_name: str, terms: set[str]) -> bool:
        if not terms:
            return False
        candidates = {
            part
            for text in (var, field_name, f"{var}.{field_name}")
            for part in re.split(r"[^A-Za-z0-9_]+", text.lower())
            if part
        }
        candidates.add(f"{var}.{field_name}".lower())
        return any(
            term == candidate or (len(term) >= 4 and (term in candidate or candidate in term))
            for term in terms
            for candidate in candidates
        )

    def _is_relevance_token(self, token: str) -> bool:
        if len(token) < 3:
            return False
        if token in self.LOW_VALUE_QUERY_TOKENS:
            return False
        if token.isdigit():
            return False
        return True

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe_topic_refs(refs: Sequence[TopicRef]) -> List[TopicRef]:
        seen = set()
        out: List[TopicRef] = []
        for ref in refs:
            key = (ref.topic, ref.direction, ref.file, ref.line, ref.api, ref.variable)
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out

    @staticmethod
    def _dedupe_param_refs(refs: Sequence[ParameterRef]) -> List[ParameterRef]:
        seen = set()
        out: List[ParameterRef] = []
        for ref in refs:
            key = (ref.name, ref.member, ref.file, ref.line, ref.access_pattern)
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out

    @staticmethod
    def _dedupe_field_refs(refs: Sequence[FieldRef]) -> List[FieldRef]:
        seen = set()
        out: List[FieldRef] = []
        for ref in refs:
            key = (ref.topic, ref.struct, ref.variable, ref.field, ref.file, ref.line, ref.assignment_operator)
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out

    @staticmethod
    def _dedupe_source_assignment_refs(refs: Sequence[SourceAssignmentRef]) -> List[SourceAssignmentRef]:
        seen = set()
        out: List[SourceAssignmentRef] = []
        for ref in refs:
            key = (ref.target, ref.expression, ref.function, ref.file, ref.line)
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out

    @staticmethod
    def _dedupe_function_call_refs(refs: Sequence[FunctionCallRef]) -> List[FunctionCallRef]:
        seen = set()
        out: List[FunctionCallRef] = []
        for ref in refs:
            key = (ref.name, ref.receiver, ref.file, ref.line, tuple(ref.control_predicates))
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out

    @staticmethod
    def _dedupe_helper_expression_refs(refs: Sequence[HelperExpressionRef]) -> List[HelperExpressionRef]:
        seen = set()
        out: List[HelperExpressionRef] = []
        for ref in refs:
            key = (ref.name, ref.file, ref.line)
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out

    @staticmethod
    def _dedupe_branch_condition_refs(refs: Sequence[BranchConditionRef]) -> List[BranchConditionRef]:
        seen = set()
        out: List[BranchConditionRef] = []
        for ref in refs:
            key = (ref.kind, ref.condition, ref.file, ref.line)
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out

    @staticmethod
    def _dedupe_parameter_predicate_refs(refs: Sequence[ParameterPredicateRef]) -> List[ParameterPredicateRef]:
        seen = set()
        out: List[ParameterPredicateRef] = []
        for ref in refs:
            key = (ref.name, ref.member, ref.predicate, ref.file, ref.line)
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out


def split_source_field(value: str) -> Tuple[str, str]:
    cleaned = MechanismSourceProfiler._clean_field_path(value)
    if "." not in cleaned:
        return cleaned, ""
    root, field = cleaned.split(".", 1)
    return root, field


def split_top_level_args(args: str) -> List[str]:
    out: List[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(args):
        if char in "([{<":
            depth += 1
        elif char in ")]}>":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            out.append(args[start:index].strip())
            start = index + 1
    tail = args[start:].strip()
    if tail:
        out.append(tail)
    return out


def substitute_expression_symbols(expression: str, names: Sequence[str], values: Sequence[str]) -> str:
    substituted = expression
    for name, value in sorted(zip(names, values), key=lambda item: len(item[0]), reverse=True):
        substituted = re.sub(
            rf"(?<![A-Za-z0-9_\.]){re.escape(name)}(?![A-Za-z0-9_])",
            str(value),
            substituted,
        )
    return substituted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile PX4 source files for mechanism resolving.")
    parser.add_argument("source_path", help="Path to PX4-Autopilot source tree")
    parser.add_argument("query", nargs="+", help="Search query terms")
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--max-matches-per-file", type=int, default=8)
    parser.add_argument("--out", help="Optional output JSON path")
    args = parser.parse_args()

    profiler = MechanismSourceProfiler(args.source_path)
    profile = profiler.profile_mechanism(
        args.query,
        max_files=args.max_files,
        max_matches_per_file=args.max_matches_per_file,
    )

    payload = json.dumps(profile, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
