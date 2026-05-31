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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from flight_log_agent.expression_math import (
    canonical_math_function_name,
    is_safe_math_function_name,
    normalize_expression_function_names,
)


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


class TopicRef(BaseModel):
    topic: str
    direction: str  # "publish", "subscribe", or "unknown"
    file: str
    line: int
    evidence: str
    struct: Optional[str] = None
    variable: Optional[str] = None
    api: Optional[str] = None


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


class HelperExpressionRef(BaseModel):
    name: str
    file: str
    line: int
    evidence: str
    parameters: List[str] = Field(default_factory=list)
    statements: List[Dict[str, Any]] = Field(default_factory=list)
    assignments: Dict[str, str] = Field(default_factory=dict)
    return_expression: Optional[str] = None
    lowered_return_expression: Optional[str] = None
    branches: List[Dict[str, str]] = Field(default_factory=list)
    symbol_bindings: Dict[str, str] = Field(default_factory=dict)
    call_resolutions: List[Dict[str, Any]] = Field(default_factory=list)
    helper_calls: List[str] = Field(default_factory=list)
    unresolved_reason: Optional[str] = None


class BranchConditionRef(BaseModel):
    kind: str
    condition: str
    file: str
    line: int
    evidence: str


class ParameterPredicateRef(BaseModel):
    name: Optional[str]
    predicate: str
    file: str
    line: int
    evidence: str
    member: Optional[str] = None
    operator: Optional[str] = None
    compared_value: Optional[str] = None


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
    function_calls: List[FunctionCallRef] = Field(default_factory=list)
    helper_expressions: List[HelperExpressionRef] = Field(default_factory=list)
    branch_conditions: List[BranchConditionRef] = Field(default_factory=list)
    parameter_predicates: List[ParameterPredicateRef] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


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
    _FUNCTION_SIGNATURE_PATTERN = re.compile(
        r"(?P<prefix>[A-Za-z_][A-Za-z0-9_:<>,~*&\s]*?)\s+"
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
        source_path: Union[str, Path],
        rg_path: str = "rg",
        read_limit_bytes: int = 2_000_000,
        excludes: Optional[Sequence[str]] = None,
    ) -> None:
        self.source_path = Path(source_path).expanduser().resolve()
        self.rg_path = rg_path
        self.read_limit_bytes = read_limit_bytes
        self.excludes = tuple(excludes) if excludes is not None else self.DEFAULT_EXCLUDES

        if not self.source_path.exists():
            raise FileNotFoundError(f"PX4 source path does not exist: {self.source_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_related_source_files(
        self,
        queries: Union[str, Sequence[str]],
        max_files: int = 12,
        max_matches_per_file: int = 8,
    ) -> List[SourceFileHit]:
        """
        Search the PX4 source tree for files related to one or more query terms.

        Args:
            queries: Single query string or list of query strings. Use concrete
                terms when possible, e.g. "FW_TKO_PITCH_MIN", "takeoff pitch",
                "position_setpoint_triplet".
            max_files: Maximum number of ranked source files to return.
            max_matches_per_file: Number of evidence lines to keep per file.

        Returns:
            Ranked source file hits with evidence lines.
        """
        query_list = self._normalize_queries(queries)
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
        return ranked[:max_files]

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
        member_to_param = self._collect_param_member_map(expanded_files)

        for path in expanded_files:
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            for line_no, line in self._iter_code_lines(text):
                for direction, api, pattern in self._UORB_DECL_PATTERNS:
                    for match in pattern.finditer(line):
                        struct = match.group("struct")
                        topic = self._topic_from_struct(struct)
                        refs.append(
                            TopicRef(
                                topic=topic,
                                struct=struct,
                                variable=match.groupdict().get("var"),
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
    ) -> List[FieldRef]:
        """
        Extract assigned fields from source files.

        The method maps fields to a uORB topic when it can infer the variable's
        struct type. For example:

            vehicle_attitude_setpoint_s attitude_sp{};
            attitude_sp.pitch_body = pitch_sp;

        becomes:

            topic=vehicle_attitude_setpoint, field=pitch_body

        It also keeps unknown fields when the variable is not mapped, because
        those can still be useful branch/logic evidence for a resolver.
        """
        refs: List[FieldRef] = []

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
                    # struct fields and likely PX4 setpoint/status variable names.
                    struct = var_to_struct.get(var)
                    if struct is None and not self._looks_like_relevant_assignment(var, field_name):
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
    ) -> List[FieldRef]:
        """
        Extract non-assignment field accesses from source files.

        This is intentionally conservative. It keeps fields that can be mapped
        to a visible uORB struct variable, plus likely setpoint/status accesses
        that may need a resolver decision.
        """
        refs: List[FieldRef] = []

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
                    if struct is None and not self._looks_like_relevant_assignment(var, field_name):
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

    def extract_function_calls_from_source(
        self,
        files: Sequence[Union[str, Path]],
    ) -> List[FunctionCallRef]:
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

        for path in self._expand_companion_files(files):
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            for line_no, line in self._iter_code_lines(text):
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue

                for match in self._FUNCTION_CALL_PATTERN.finditer(line):
                    name = match.group("name")
                    if name in ignored:
                        continue
                    receiver = self._call_receiver(line, match.start())
                    refs.append(
                        FunctionCallRef(
                            name=name,
                            receiver=receiver,
                            file=rel_file,
                            line=line_no,
                            evidence=stripped,
                        )
                    )

        return self._dedupe_function_call_refs(refs)

    def extract_helper_expressions_from_source(
        self,
        files: Sequence[Union[str, Path]],
        helper_names: Optional[Sequence[str]] = None,
    ) -> List[HelperExpressionRef]:
        helper_name_set = {name for name in (helper_names or []) if name}
        refs: List[HelperExpressionRef] = []
        expanded_files = self._expand_companion_files(files)
        member_to_param = self._collect_param_member_map(expanded_files)

        for path in expanded_files:
            text = self._read_text(path)
            if text is None:
                continue

            rel_file = self._rel(path)
            for definition in self._extract_function_definitions(text, rel_file):
                name = definition["name"]
                short_name = name.split("::")[-1]
                if helper_name_set and name not in helper_name_set and short_name not in helper_name_set:
                    continue
                translated = self._translate_helper_body(
                    name=name,
                    file=rel_file,
                    line=definition["line"],
                    params=definition["params"],
                    body=definition["body"],
                    evidence=definition["evidence"],
                    member_to_param=member_to_param,
                )
                refs.append(translated)

        return self._dedupe_helper_expression_refs(refs)

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
        fields = self.extract_assigned_fields_from_source(files)
        read_fields = self.extract_read_fields_from_source(files)
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
            source_root=str(self.source_path),
            related_files=hits,
            published_topics=uorb["published_topics"],
            subscribed_topics=uorb["subscribed_topics"],
            unknown_direction_topics=uorb["unknown_direction_topics"],
            referenced_parameters=params,
            assigned_fields=fields,
            read_fields=read_fields,
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

    def _normalize_queries(self, queries: Union[str, Sequence[str]]) -> List[str]:
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

            # Also search individual strong-looking tokens for recall.
            for token in re.findall(r"[A-Z][A-Z0-9_]{2,}|[A-Za-z_][A-Za-z0-9_]{5,}", query):
                if self._should_keep_query(token) and token not in normalized:
                    normalized.append(token)

        return normalized

    def _ripgrep_or_python_search(self, query: str) -> List[SourceMatch]:
        try:
            return self._ripgrep_search(query)
        except Exception:
            return self._python_search(query)

    def _ripgrep_search(self, query: str) -> List[SourceMatch]:
        cmd = [
            self.rg_path,
            "--json",
            "--line-number",
            "--ignore-case",
            "--fixed-strings",
        ]

        for glob in self.SOURCE_GLOBS:
            cmd += ["-g", glob]

        for exclude in self.excludes:
            cmd += ["-g", f"!{exclude}/**"]

        cmd += [query, str(self.source_path)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )

        matches: List[SourceMatch] = []
        for raw_line in result.stdout.splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            if event.get("type") != "match":
                continue

            data = event.get("data", {})
            path_text = data.get("path", {}).get("text")
            line_number = data.get("line_number")
            line_text = data.get("lines", {}).get("text", "").rstrip("\n")
            if not path_text or line_number is None:
                continue

            matches.append(
                SourceMatch(
                    file=self._rel(Path(path_text)),
                    line=int(line_number),
                    text=line_text.strip(),
                    query=query,
                )
            )

        return matches

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
        for path in self.source_path.rglob("*"):
            if not path.is_file():
                continue
            if not any(path.match(f"**/{glob}") for glob in self.SOURCE_GLOBS):
                continue
            rel = self._rel(path)
            if self._is_excluded(rel):
                continue
            yield path

    def _is_excluded(self, rel_file: str) -> bool:
        rel_l = rel_file.lower()
        for exclude in self.excludes:
            exclude_l = exclude.lower().rstrip("/")
            if rel_l == exclude_l or rel_l.startswith(exclude_l + "/"):
                return True
        return False

    def _resolve_file(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path)
        if path.is_absolute():
            return path
        return self.source_path / path

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
                if candidate.exists() and candidate.is_file():
                    expanded[str(candidate.resolve())] = candidate.resolve()
                    for include in self._local_include_files(candidate):
                        expanded[str(include.resolve())] = include.resolve()

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
            include_path = (path.parent / match.group("include")).resolve()
            try:
                include_path.relative_to(self.source_path)
            except ValueError:
                continue
            if include_path.exists() and include_path.is_file():
                includes.append(include_path)
        return includes

    def _read_text(self, path: Path) -> Optional[str]:
        try:
            if path.stat().st_size > self.read_limit_bytes:
                return None
            return path.read_text(errors="ignore")
        except Exception:
            return None

    def _iter_code_lines(self, text: str) -> Iterable[Tuple[int, str]]:
        # Strip block comments lightly. This is deliberately simple and avoids
        # pretending to be a full C++ parser.
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
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
            definitions.append(
                {
                    "name": name,
                    "line": line_no,
                    "params": self._parse_function_parameters(match.group("params")),
                    "body": text[open_brace + 1:close_brace],
                    "evidence": f"{signature} {{",
                    "file": rel_file,
                }
            )
        return definitions

    @staticmethod
    def _strip_block_comments_preserve_lines(text: str) -> str:
        return re.sub(
            r"/\*.*?\*/",
            lambda match: "\n" * match.group(0).count("\n"),
            text,
            flags=re.DOTALL,
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
    ) -> HelperExpressionRef:
        cleaned_body = self._strip_line_comments(body)
        unresolved = self._unsupported_helper_body_reason(cleaned_body)
        statements = self._helper_statements(cleaned_body)
        assignments = self._helper_assignments(cleaned_body)
        branches = self._helper_return_branches(cleaned_body)
        return_expression = self._helper_return_expression(cleaned_body)
        lowered_return_expression = self._lower_helper_statements(statements) if unresolved is None else None
        helper_calls = self._helper_body_calls(cleaned_body)
        symbol_bindings = self._helper_symbol_bindings(cleaned_body, member_to_param)
        call_resolutions = self._helper_call_resolutions(cleaned_body, member_to_param)

        if unresolved is None and not return_expression and not lowered_return_expression and not branches:
            unresolved = "helper body has no simple return expression"
        if unresolved is None and len(re.findall(r"\breturn\b", cleaned_body)) > 1 and not branches and not lowered_return_expression:
            unresolved = "helper body has multiple returns without translatable branch structure"

        return HelperExpressionRef(
            name=name,
            file=file,
            line=line,
            evidence=evidence,
            parameters=params,
            statements=statements if unresolved is None else [],
            assignments=assignments if unresolved is None else {},
            return_expression=return_expression if unresolved is None else None,
            lowered_return_expression=lowered_return_expression if unresolved is None else None,
            branches=branches if unresolved is None else [],
            symbol_bindings=symbol_bindings if unresolved is None else {},
            call_resolutions=call_resolutions,
            helper_calls=helper_calls,
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
            semicolon = self._find_statement_semicolon(text, index)
            if semicolon is None:
                break
            statement = self._parse_helper_simple_statement(text[index:semicolon].strip())
            if statement:
                statements.append(statement)
            index = semicolon + 1
        return statements

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
        for statement in statements:
            kind = statement.get("kind")
            if kind in {"declare", "assign"}:
                target = str(statement.get("target") or "")
                expression = self._substitute_helper_locals(str(statement.get("expression") or ""), env)
                if target:
                    env[target] = expression
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
                for target in sorted(set(then_env) | set(else_env)):
                    before = env.get(target, target)
                    then_expr = then_env.get(target, before)
                    else_expr = else_env.get(target, before)
                    if then_expr != before or else_expr != before:
                        env[target] = f"({then_expr} if {condition} else {else_expr})"
        return None

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

    @staticmethod
    def _unsupported_helper_body_reason(body: str) -> Optional[str]:
        unsupported_patterns = [
            (r"\b(for|while|switch|case|goto)\b", "helper body uses unsupported control flow"),
            (r"\breturn\s*;", "helper body returns void"),
            (r"\*[A-Za-z_][A-Za-z0-9_]*\s*=", "helper body mutates pointer output"),
            (r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.|->)[A-Za-z_][A-Za-z0-9_]*\s*=", "helper body mutates object state"),
            (r"\+\+|--", "helper body mutates state"),
        ]
        for pattern, reason in unsupported_patterns:
            if re.search(pattern, body):
                return reason
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

    def _helper_symbol_bindings(self, body: str, member_to_param: Dict[str, str]) -> Dict[str, str]:
        bindings: Dict[str, str] = {}
        for match in self._PARAM_GET_MEMBER_PATTERN.finditer(body):
            member_expr = re.sub(r"\s+", "", match.group("member"))
            member_name = member_expr.split(".")[-1]
            param_name = member_to_param.get(member_expr) or member_to_param.get(member_name)
            if param_name:
                bindings[f"{member_expr}.get()"] = param_name

        var_to_struct = self._extract_struct_variables(body)
        for match in self._FIELD_ACCESS_PATTERN.finditer(body):
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
        expr = expr.strip()
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
        return re.sub(r"\s*(?:\.|->)\s*", ".", field_text.strip())

    def _rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.source_path))
        except Exception:
            return str(path)

    @staticmethod
    def _looks_like_param_member(member: str) -> bool:
        member_l = member.lower()
        return (
            member_l.startswith("_param")
            or member_l.startswith("param")
            or member_l.startswith("params")
            or "_param_" in member_l
        )

    @staticmethod
    def _looks_like_relevant_assignment(var: str, field_name: str) -> bool:
        var_l = var.lower()
        field_l = field_name.lower()
        variable_hint = any(
            token in var_l
            for token in (
                "sp",
                "setpoint",
                "status",
                "vehicle",
                "mission",
                "pos",
                "att",
                "airspeed",
                "tecs",
                "npfg",
                "control",
            )
        )
        field_hint = any(
            token in field_l
            for token in (
                "timestamp",
                "lat",
                "lon",
                "alt",
                "roll",
                "pitch",
                "yaw",
                "thrust",
                "airspeed",
                "velocity",
                "valid",
                "type",
                "nav_state",
                "loiter",
                "radius",
                "current",
                "previous",
                "next",
            )
        )
        return variable_hint and field_hint

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
    def _dedupe_function_call_refs(refs: Sequence[FunctionCallRef]) -> List[FunctionCallRef]:
        seen = set()
        out: List[FunctionCallRef] = []
        for ref in refs:
            key = (ref.name, ref.receiver, ref.file, ref.line)
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
