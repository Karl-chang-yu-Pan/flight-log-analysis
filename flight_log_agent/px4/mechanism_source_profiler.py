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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field


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
        r"\bParam(?:Bool|Int|Float|ExtBool|ExtInt|ExtFloat|Custom)?\s*<\s*px4::params::(?P<name>[A-Z][A-Z0-9_]*)\s*>\s*(?P<member>[A-Za-z_][A-Za-z0-9_]*)?"
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

                hit.score += self._score_match(match, query)

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
        for path in self._expand_companion_files(files):
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
            normalized.append(query)

            # Also search individual strong-looking tokens for recall.
            for token in re.findall(r"[A-Z][A-Z0-9_]{2,}|[A-Za-z_][A-Za-z0-9_]{5,}", query):
                if token not in normalized:
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
            boost += 4.0
        elif path_l.startswith("src/drivers/") or path_l.startswith("src/include/"):
            boost += 2.0
        elif path_l.startswith("src/"):
            boost += 1.0
        if "/test" in path_l or "_test" in path_l or "/unit" in path_l:
            boost -= 2.0
        if "/build" in path_l or "/.git" in path_l:
            boost -= 5.0
        return boost

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
        headers and usage in implementation files.
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

        return list(expanded.values())

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
