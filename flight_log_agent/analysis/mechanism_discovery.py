"""Deterministic mechanism discovery — the profiler→DAG production seam.

Stage 1 of the discovery rework (#73): adapt the profiler's per-file facts
(``SourceFileFacts``) into the inputs :func:`build_mechanism_dag` expects.
Persistent fact-cache use is intentionally suspended until DAG construction
semantics are accepted; discovery currently extracts every requested file
fresh.

The binding mapping was validated end-to-end against real PX4 v1.14.3
source (airspeed / NPFG / weathervane / terrain mechanisms) before being
promoted here from the test shim.

The discovery fixed point retrieves candidates, validates exact source
entities, loads their facts, rebuilds the DAG, and repeats from the typed
frontier. No LLM code belongs in this file.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from flight_log_agent.analysis.source_expansion import (
    SourceExpansionResolver,
    SourceStructureIndex,
    SourceSymbolIdentity,
    UnresolvedSourceReference,
    reference_receiver_is_source_boundary,
)
from flight_log_agent.analysis.mechanism_dag import (
    MechanismDAG,
    _DAGBuilder,
    build_mechanism_dag,
    evaluate_feasibility,
)
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler
from flight_log_agent.px4.source_facts_cache import (
    SourceFileFacts,
    extract_facts_for_file,
)
from flight_log_agent.symbols import (
    exact_symbol,
    parse_signal_reference,
    source_storage_produces_reference,
    symbol_produces_reference,
)
from flight_log_agent.utils import dedupe_keep_order


def _as_dict(ref: Any) -> dict[str, Any]:
    if isinstance(ref, dict):
        return ref
    if hasattr(ref, "model_dump"):
        return ref.model_dump(exclude_none=True)
    return dict(vars(ref))


def binding_from_assignment(assignment: Any) -> dict[str, Any]:
    """Convert one profiler ``SourceAssignmentRef`` into a DAG binding dict.

    The DAG builder reads ``target_symbol`` / ``source_symbol`` /
    ``assignment_path`` rather than the profiler's flat ``target`` /
    ``expression`` / ``file`` / ``line``. Message transfers are separate
    operation bindings; an assignment is never relabelled as a logged value
    merely because the same object is published later.
    """
    ref = _as_dict(assignment)
    expression_ref = dict(ref.get("expression_ref") or {})
    topic = ref.get("target_topic")
    field_name = ref.get("target_field")
    return {
        "target_symbol": str(ref.get("target") or ""),
        "source_symbol": str(ref.get("expression") or ""),
        "function": str(ref.get("function") or ""),
        "callable_id": str(ref.get("callable_id") or ""),
        "function_owner": str(ref.get("owner") or ""),
        "target_identity": dict(ref.get("target_identity") or {}),
        "function_parameters": list(ref.get("function_parameters") or []),
        "declaration_kind": str(ref.get("declaration_kind") or ""),
        "constant_scopes": [
            str(value) for value in (ref.get("constant_scopes") or []) if value
        ],
        "assignment_operator": str(ref.get("assignment_operator") or "="),
        "source_site_id": str(ref.get("source_site_id") or ""),
        "expression_ref": expression_ref,
        "reference_identities": {
            exact_symbol(str(symbol)): dict(identity)
            for symbol, identity in (
                expression_ref.get("input_identities") or {}
            ).items()
            if exact_symbol(str(symbol)) and isinstance(identity, dict)
        },
        "control_expression_refs": [
            dict(value)
            for value in (ref.get("control_expression_refs") or [])
        ],
        "assignment_path": [
            {
                "file": str(ref.get("file") or ""),
                "line": int(ref.get("line") or 0),
                "expression": str(ref.get("expression") or ""),
            }
        ],
        # A struct type proves declaration compatibility, not publication.
        "declared_signal": f"{topic}.{field_name}" if topic and field_name else "",
        "logged_signal": "",
        "control_predicates": list(ref.get("control_predicates") or []),
        "control_predicate_lines": list(ref.get("control_predicate_lines") or []),
        "control_predicate_site_ids": list(
            ref.get("control_predicate_site_ids") or []
        ),
        "reachability_exact": bool(ref.get("reachability_exact", True)),
        "struct_variables": dict(ref.get("struct_variables") or {}),
    }


@dataclass
class DAGInputs:
    """Aggregated, deduplicated ``build_mechanism_dag`` inputs.

    One instance per discovery round, built from the union of every
    loaded file's Layer 1 facts.
    """

    bindings: list[dict[str, Any]] = field(default_factory=list)
    helper_expressions: list[dict[str, Any]] = field(default_factory=list)
    parameter_predicates: list[dict[str, Any]] = field(default_factory=list)
    parameter_bindings: list[dict[str, Any]] = field(default_factory=list)
    parameter_names: set[str] = field(default_factory=set)
    call_statements: list[dict[str, Any]] = field(default_factory=list)
    boundary_bindings: list[dict[str, Any]] = field(default_factory=list)
    structure: SourceStructureIndex = field(default_factory=SourceStructureIndex)


def dag_inputs_from_facts(facts: Iterable[Any]) -> DAGInputs:
    """Aggregate per-file facts into deduplicated DAG-builder inputs.

    Layer 1 entries are per-file, so duplicates only arise when the same
    file is passed twice or two files declare identical facts (headers);
    both are deduplicated on source identity. Parameter members retain their
    declaring owner instead of being collapsed into a process-wide name map.
    """
    inputs = DAGInputs()
    seen_bindings: set[tuple[Any, ...]] = set()
    seen_helpers: set[tuple[Any, ...]] = set()
    seen_predicates: set[tuple[Any, ...]] = set()
    seen_calls: set[tuple[Any, ...]] = set()
    seen_parameters: set[tuple[Any, ...]] = set()

    entries = [_as_dict(facts_entry) for facts_entry in facts]
    inputs.structure = SourceStructureIndex.from_facts(entries)
    calls_by_source_site = {
        str(call.get("source_site_id") or ""): call
        for entry in entries
        for raw_call in entry.get("function_calls") or []
        if (call := _as_dict(raw_call)).get("source_site_id")
    }

    class_bases: dict[str, list[str]] = {}
    callable_owners: dict[str, str] = {}
    for entry in entries:
        for raw_class in entry.get("classes") or []:
            class_ref = _as_dict(raw_class)
            name = str(class_ref.get("name") or "")
            if name:
                class_bases[name] = [str(base) for base in class_ref.get("bases") or []]
        for raw_callable in entry.get("callables") or []:
            callable_ref = _as_dict(raw_callable)
            callable_id = str(callable_ref.get("callable_id") or "")
            owner = str(callable_ref.get("owner") or "")
            if callable_id and owner:
                callable_owners[callable_id] = owner

    def owner_lineage(owner: str) -> set[str]:
        lineage: set[str] = set()
        pending = [owner] if owner else []
        while pending:
            current = pending.pop(0)
            if not current or current in lineage:
                continue
            lineage.add(current)
            pending.extend(class_bases.get(current, []))
        return lineage

    topic_refs_by_variable: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        for direction_key in ("subscribed_topics", "published_topics"):
            direction = "subscribe" if direction_key == "subscribed_topics" else "publish"
            for raw_ref in entry.get(direction_key) or []:
                ref = _as_dict(raw_ref)
                variable = str(ref.get("variable") or "")
                topic = str(ref.get("topic") or "")
                if not variable or not topic:
                    continue
                transfer_call = calls_by_source_site.get(
                    str(ref.get("source_site_id") or ""), {}
                ) if ref.get("transfer") else {}
                topic_refs_by_variable.setdefault(variable, []).append(
                    {
                        "source_symbol": variable,
                        "topic": topic,
                        "instance": ref.get("instance"),
                        "direction": direction,
                        "file": str(ref.get("file") or entry.get("file") or ""),
                        "line": int(ref.get("line") or 0),
                        "function": str(ref.get("function") or ""),
                        "callable_id": str(ref.get("callable_id") or ""),
                        "source_owner": str(ref.get("variable_owner") or ""),
                        "source_identity": dict(
                            ref.get("variable_identity") or {}
                        ),
                        "endpoint_kind": str(ref.get("endpoint_kind") or ""),
                        "source_site_id": str(ref.get("source_site_id") or ""),
                        "transfer": bool(ref.get("transfer", False)),
                        "provenance": str(ref.get("api") or direction_key),
                        "control_predicates": [
                            str(value)
                            for value in (
                                transfer_call.get("control_predicates")
                                or ref.get("control_predicates")
                                or []
                            )
                        ],
                        "control_predicate_lines": [
                            int(value)
                            for value in (
                                transfer_call.get("control_predicate_lines")
                                or ref.get("control_predicate_lines")
                                or []
                            )
                        ],
                        "control_predicate_site_ids": [
                            str(value)
                            for value in (
                                transfer_call.get("control_predicate_site_ids")
                                or ref.get("control_predicate_site_ids")
                                or []
                            )
                        ],
                        "control_expression_refs": [
                            dict(value)
                            for value in (
                                transfer_call.get("control_expression_refs")
                                or ref.get("control_expression_refs")
                                or []
                            )
                        ],
                        "reachability_exact": bool(
                            transfer_call.get(
                                "reachability_exact",
                                ref.get("reachability_exact", True),
                            )
                        ),
                    }
                )

    def scoped_boundaries(
        variable: str,
        direction: str,
        source_file: str,
        callable_id: str,
        caller_owner: str,
        source_identity: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        receiver = variable.replace("->", ".")
        if receiver.startswith("this."):
            receiver = receiver[5:]
        lineage = owner_lineage(caller_owner)
        candidates = [
            item
            for lookup in ({receiver, "this"} if not receiver else {receiver})
            for item in topic_refs_by_variable.get(lookup, [])
            if (
                item.get("direction") == direction
                and item.get("topic")
                and not item.get("transfer")
            )
        ]
        scoped: list[dict[str, Any]] = []
        for item in candidates:
            endpoint_identity = dict(item.get("source_identity") or {})
            if source_identity and endpoint_identity:
                reference = SourceSymbolIdentity.model_validate(source_identity)
                producer = SourceSymbolIdentity.model_validate(endpoint_identity)
                if "unknown" in {reference.kind, producer.kind}:
                    continue
                if (
                    reference.declaration_proven
                    and producer.declaration_proven
                    and not inputs.structure.storage_compatible(
                        reference, producer
                    )
                ):
                    continue
            endpoint_kind = str(item.get("endpoint_kind") or "")
            item_owner = str(item.get("source_owner") or "")
            item_callable = str(item.get("callable_id") or "")
            item_file = str(item.get("file") or "")
            if not endpoint_kind:
                endpoint_kind = "member" if item_owner else "local" if item_callable else "global"
            if endpoint_kind == "base":
                if caller_owner and item_owner in lineage and receiver in {"", "this"}:
                    scoped.append(item)
            elif endpoint_kind == "member":
                if caller_owner and item_owner in lineage:
                    scoped.append(item)
            elif endpoint_kind == "local":
                if callable_id and item_callable == callable_id:
                    scoped.append(item)
            elif endpoint_kind == "global" and source_file and item_file == source_file:
                scoped.append(item)
        return scoped

    inputs.boundary_bindings.extend(
        item
        for refs in topic_refs_by_variable.values()
        for item in refs
    )

    for entry in entries:
        for raw_call in entry.get("function_calls") or []:
            call = _as_dict(raw_call)
            name = str(call.get("name") or "").rsplit("::", 1)[-1]
            receiver = str(call.get("receiver") or "")
            args = [str(arg) for arg in (call.get("args") or [])]
            if not args:
                continue
            direction = "subscribe" if name in {"copy", "update"} else (
                "publish" if name == "publish" else ""
            )
            if not direction:
                continue
            file = str(call.get("file") or entry.get("file") or "")
            callable_id = str(call.get("callable_id") or "")
            function = str(call.get("function") or "")
            caller_owner = callable_owners.get(callable_id, "")
            if not caller_owner and "::" in function:
                caller_owner = function.rpartition("::")[0]
            endpoints = scoped_boundaries(
                receiver or "this",
                direction,
                file,
                callable_id,
                caller_owner,
                source_identity=dict(call.get("receiver_identity") or {}),
            )
            if not endpoints:
                continue
            source_symbol = args[0].lstrip("&*").strip()
            if not source_symbol:
                continue
            root = source_symbol.replace("->", ".").split(".", 1)[0]
            source_owner = str(
                (call.get("argument_owners") or {}).get(root) or ""
            )
            argument_identity: dict[str, Any] = {}
            argument_refs = list(call.get("argument_expressions") or [])
            if argument_refs:
                identities = dict(
                    (argument_refs[0] or {}).get("input_identities") or {}
                )
                argument_identity = dict(
                    identities.get(exact_symbol(source_symbol))
                    or identities.get(exact_symbol(root))
                    or {}
                )
                if argument_identity:
                    source_owner = (
                        str(
                            argument_identity.get("declaring_class")
                            or argument_identity.get("class_owner")
                            or source_owner
                        )
                        if argument_identity.get("kind") == "member"
                        else ""
                    )
            for endpoint in endpoints:
                inputs.boundary_bindings.append(
                    {
                        "source_symbol": source_symbol,
                        "topic": str(endpoint.get("topic") or ""),
                        "instance": endpoint.get("instance"),
                        "direction": direction,
                        "file": file,
                        "function": function,
                        "callable_id": callable_id,
                        "source_owner": source_owner,
                        "source_identity": argument_identity,
                        "source_site_id": str(call.get("source_site_id") or ""),
                        "line": int(call.get("line") or 0),
                        "transfer": True,
                        "provenance": f"{receiver}.{name}",
                        "control_predicates": [
                            *(
                                str(value)
                                for value in endpoint.get("control_predicates") or []
                            ),
                            *(
                                str(value)
                                for value in call.get("control_predicates") or []
                            ),
                        ],
                        "control_predicate_lines": [
                            *(
                                int(value)
                                for value in endpoint.get("control_predicate_lines") or []
                            ),
                            *(
                                int(value)
                                for value in call.get("control_predicate_lines") or []
                            ),
                        ],
                        "control_predicate_site_ids": [
                            *(
                                str(value)
                                for value in endpoint.get("control_predicate_site_ids") or []
                            ),
                            *(
                                str(value)
                                for value in call.get("control_predicate_site_ids") or []
                            ),
                        ],
                        "control_expression_refs": [
                            *(
                                dict(value)
                                for value in endpoint.get("control_expression_refs") or []
                            ),
                            *(
                                dict(value)
                                for value in call.get("control_expression_refs") or []
                            ),
                        ],
                        "reachability_exact": bool(
                            endpoint.get("reachability_exact", True)
                        ) and bool(call.get("reachability_exact", True)),
                    }
                )

    def boundary_operation(item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Lower one source-proven message transfer to an ordinary binding."""
        if not item.get("transfer"):
            return None
        topic = str(item.get("topic") or "")
        payload = str(item.get("source_symbol") or "")
        direction = str(item.get("direction") or "")
        if not topic or not payload or direction not in {"subscribe", "publish"}:
            return None
        instance = item.get("instance")
        topic_identity = f"{topic}[{instance}]" if instance is not None else topic
        payload_identity = dict(item.get("source_identity") or {})
        if not payload_identity:
            callable_id = str(item.get("callable_id") or "")
            callable_record = inputs.structure.callables_by_id.get(callable_id) or {}
            derived_identity = inputs.structure.symbol_identity(
                payload,
                file=str(item.get("file") or ""),
                callable_id=callable_id,
                function_name=str(
                    callable_record.get("name") or item.get("function") or ""
                ),
                function_parameters=[
                    str(value)
                    for value in (callable_record.get("parameters") or [])
                ],
                class_owner_hint=str(item.get("source_owner") or ""),
            )
            if derived_identity.kind != "unknown":
                payload_identity = derived_identity.model_dump()
        if direction == "subscribe":
            target, source = payload, topic_identity
            target_identity = payload_identity
            input_identities: dict[str, Any] = {}
        else:
            target, source = topic_identity, payload
            target_identity = {}
            input_identities = {source: payload_identity} if payload_identity else {}
        source_site_id = str(item.get("source_site_id") or "")
        file = str(item.get("file") or "")
        line = int(item.get("line") or 0)
        return {
            "target_symbol": target,
            "source_symbol": source,
            "function": str(item.get("function") or ""),
            "callable_id": str(item.get("callable_id") or ""),
            "function_owner": str(item.get("source_owner") or ""),
            "target_identity": target_identity,
            "assignment_operator": "=",
            "source_site_id": source_site_id,
            "expression_ref": {
                "text": source,
                "lowered_text": source,
                "input_symbols": [source],
                "input_identities": input_identities,
                "call_results": [],
                "direct_storage": source,
                "exact": True,
            },
            "reference_identities": input_identities,
            "assignment_path": [
                {"file": file, "line": line, "expression": source}
            ],
            "declared_signal": topic_identity,
            "logged_signal": "",
            "control_predicates": list(item.get("control_predicates") or []),
            "control_predicate_lines": list(
                item.get("control_predicate_lines") or []
            ),
            "control_predicate_site_ids": list(
                item.get("control_predicate_site_ids") or []
            ),
            "control_expression_refs": list(
                item.get("control_expression_refs") or []
            ),
            "reachability_exact": bool(item.get("reachability_exact", True)),
            "synthetic_boundary_transfer": True,
            "boundary_direction": direction,
            "external_source_signal": direction == "subscribe",
            "external_target_signal": direction == "publish",
            "provenance": str(item.get("provenance") or direction),
        }

    seen_transfers: set[tuple[Any, ...]] = set()
    for item in inputs.boundary_bindings:
        operation = boundary_operation(item)
        if operation is None:
            continue
        transfer_key = (
            str(operation.get("source_site_id") or ""),
            str(operation.get("target_symbol") or ""),
            str(operation.get("source_symbol") or ""),
            str(operation.get("boundary_direction") or ""),
        )
        if transfer_key in seen_transfers:
            continue
        seen_transfers.add(transfer_key)
        inputs.bindings.append(operation)

    endpoint_declarations = {
        str((item.get("source_identity") or {}).get("declaration_id") or "")
        for item in inputs.boundary_bindings
        if not item.get("transfer")
    }
    endpoint_declarations.discard("")

    for entry in entries:

        for assignment in entry.get("source_assignments") or []:
            ref = _as_dict(assignment)
            target_declaration = str(
                (ref.get("target_identity") or {}).get("declaration_id") or ""
            )
            if target_declaration in endpoint_declarations:
                # Endpoint construction configures the external transfer; it
                # is not a runtime value writer for the endpoint object.
                continue
            key = (
                str(ref.get("source_site_id") or ""),
                str(ref.get("callable_id") or ""),
                str(ref.get("file") or ""),
                int(ref.get("line") or 0),
                str(ref.get("target") or ""),
                str(ref.get("expression") or ""),
            )
            if key in seen_bindings:
                continue
            seen_bindings.add(key)
            binding = binding_from_assignment(ref)
            inputs.bindings.append(binding)

        for helper in entry.get("helper_expressions") or []:
            helper_dict = _as_dict(helper)
            key = (
                str(helper_dict.get("source_site_id") or ""),
                str(helper_dict.get("callable_id") or ""),
                str(helper_dict.get("name") or ""),
                str(helper_dict.get("file") or ""),
                int(helper_dict.get("line") or 0),
            )
            if key in seen_helpers:
                continue
            seen_helpers.add(key)
            inputs.helper_expressions.append(helper_dict)

        for predicate in entry.get("parameter_predicates") or []:
            predicate_dict = _as_dict(predicate)
            key = (
                str(predicate_dict.get("source_site_id") or ""),
                str(predicate_dict.get("callable_id") or ""),
                str(predicate_dict.get("predicate") or ""),
                str(predicate_dict.get("file") or ""),
                int(predicate_dict.get("line") or 0),
            )
            if key in seen_predicates:
                continue
            seen_predicates.add(key)
            inputs.parameter_predicates.append(predicate_dict)

        for call in entry.get("function_calls") or []:
            call_dict = _as_dict(call)
            key = (
                str(call_dict.get("source_site_id") or ""),
                str(call_dict.get("callable_id") or ""),
                str(call_dict.get("name") or ""),
                str(call_dict.get("receiver") or ""),
                str(call_dict.get("file") or ""),
                int(call_dict.get("line") or 0),
                tuple(str(arg) for arg in call_dict.get("args") or []),
            )
            if key in seen_calls:
                continue
            seen_calls.add(key)
            inputs.call_statements.append(call_dict)

        for parameter in entry.get("referenced_parameters") or []:
            parameter_dict = _as_dict(parameter)
            name = parameter_dict.get("name")
            member = parameter_dict.get("member")
            if name:
                inputs.parameter_names.add(str(name))
                if member:
                    key = (
                        str(parameter_dict.get("source_site_id") or ""),
                        str(parameter_dict.get("owner") or ""),
                        str(member),
                        str(name),
                        str(parameter_dict.get("file") or ""),
                    )
                    if key not in seen_parameters:
                        seen_parameters.add(key)
                        inputs.parameter_bindings.append(parameter_dict)

    inputs.bindings = inputs.structure.enrich_bindings(inputs.bindings)
    inputs.call_statements = inputs.structure.enrich_calls(inputs.call_statements)
    return inputs


def load_facts(
    profiler: MechanismSourceProfiler,
    cache_root: Union[str, Path],
    files: Sequence[str],
    source_hash: str,
    *,
    source_root: Optional[Union[str, Path]] = None,
    git_path: str = "git",
) -> list[SourceFileFacts]:
    """Extract fresh per-file facts for ``files`` without disk-cache use.

    Layer 1 cache helpers remain available in ``source_facts_cache`` for later
    reactivation, but constructor development must not consume or create stale
    facts. ``cache_root``, ``source_root``, and ``git_path`` remain in this
    stable API so cache use can be restored after DAG correctness is accepted.
    Duplicate paths load once and order is preserved.
    """
    _ = (cache_root, source_root, git_path)
    facts: list[SourceFileFacts] = []
    seen: set[str] = set()
    for file_path in files:
        normalized = str(file_path)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        facts.append(extract_facts_for_file(profiler, normalized, source_hash))
    return facts

# ---------------------------------------------------------------------------
# Source survey (Phase 0: propose terminals from source, not from memory)
# ---------------------------------------------------------------------------


_PARAMETER_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_SIGNAL_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9_]*(?:\[\d+\])?\.[A-Za-z_][A-Za-z0-9_.\[\]]*")
_IDENTIFIER_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")


@dataclass
class SurveyAnchors:
    """Exact identifiers the question names, verified against the tree.

    Nothing here is a name-similarity guess: a parameter anchor exists
    only if the source declares that parameter, a signal anchor only if
    the flight actually recorded it, a callable anchor only if the tree
    defines a function with that name.
    """

    parameters: set[str] = field(default_factory=set)
    parameter_members: set[str] = field(default_factory=set)
    signals: set[str] = field(default_factory=set)
    callables: set[str] = field(default_factory=set)

    def __bool__(self) -> bool:
        return bool(self.parameters or self.signals or self.callables)


@dataclass
class SurveyedTarget:
    """One symbol written in one source file, reached from an anchor."""

    symbol: str
    source_target_id: str = ""
    callable_id: str = ""
    identity: dict[str, Any] = field(default_factory=dict)
    function: str = ""
    line: int = 0
    expression: str = ""
    writes: int = 0
    published_signal: str = ""
    # 0 = the write reads an anchor directly; 1 = one dataflow hop away.
    distance: int = 0


@dataclass
class SurveyedFile:
    file: str
    targets: list[SurveyedTarget] = field(default_factory=list)


@dataclass
class SourceSurvey:
    """Deterministic menu of terminals that EXIST in the pinned source.

    Search + two extractors (assignments, parameters) — no fact bundle,
    no cache. Two jobs:

    * remove identifier RECALL from terminal selection — a symbol absent
      from the tree cannot be proposed, so a member or module path
      remembered from another release never reaches discovery;
    * keep the menu ACCURATE — targets are those a question anchor
      actually reaches through source dataflow, not every symbol a file
      happens to write. A whole file's write list is a dump, and
      truncating a dump discards evidence at random.
    """

    files: list[SurveyedFile] = field(default_factory=list)
    anchors: SurveyAnchors = field(default_factory=SurveyAnchors)

    def symbols(self) -> set[str]:
        return {target.symbol for entry in self.files for target in entry.targets}

    def files_for(self, symbol: str) -> list[str]:
        return [
            entry.file
            for entry in self.files
            if any(target.symbol == symbol for target in entry.targets)
        ]

    def targets_for(
        self,
        symbol: str,
        *,
        source_target_id: str = "",
        file: Optional[str] = None,
    ) -> list[tuple[str, SurveyedTarget]]:
        """Return exact surveyed storage candidates for a model proposal."""
        return [
            (entry.file, target)
            for entry in self.files
            for target in entry.targets
            if target.symbol == symbol
            and (not source_target_id or target.source_target_id == source_target_id)
            and (not file or entry.file == file)
        ]

    def as_payload(self) -> dict[str, Any]:
        return {
            "anchors": {
                "parameters": sorted(self.anchors.parameters),
                "signals": sorted(self.anchors.signals),
                "callables": sorted(self.anchors.callables),
            },
            "files": [
                {
                    "file": entry.file,
                    "write_targets": [
                        {
                            "symbol": target.symbol,
                            "source_target_id": target.source_target_id,
                            "callable_id": target.callable_id,
                            "identity": target.identity,
                            "function": target.function,
                            "line": target.line,
                            "expression": target.expression,
                            "writes": target.writes,
                            "reaches_anchor_in_hops": target.distance,
                            **(
                                {"published_signal": target.published_signal}
                                if target.published_signal
                                else {}
                            ),
                        }
                        for target in entry.targets
                    ],
                }
                for entry in self.files
            ],
        }


def derive_survey_anchors(
    texts: Sequence[str],
    *,
    declared_parameters: dict[str, set[str]],
    observed_signals: Iterable[str] = (),
    known_callables: Iterable[str] = (),
) -> SurveyAnchors:
    """Anchors the question names AND the tree confirms.

    ``declared_parameters`` maps a canonical parameter name to the source
    members that read it (the ``DEFINE_PARAMETERS`` map) — the parameter
    a question names resolves to the exact member its writes reference.
    """
    blob = " ".join(str(t) for t in texts if t)
    observed = {str(s) for s in observed_signals if s}
    callables = {str(c) for c in known_callables if c}

    anchors = SurveyAnchors()
    for token in _PARAMETER_TOKEN_RE.findall(blob):
        members = declared_parameters.get(token)
        if members is not None:
            anchors.parameters.add(token)
            anchors.parameter_members.update(members)
    for token in _SIGNAL_TOKEN_RE.findall(blob):
        if token in observed:
            anchors.signals.add(token)
    for token in _IDENTIFIER_TOKEN_RE.findall(blob):
        if token in callables:
            anchors.callables.add(token)
    return anchors


def _bare_callable(name: Any) -> str:
    return str(name or "").rsplit("::", 1)[-1].strip()


def survey_source_files(
    profiler: MechanismSourceProfiler,
    files: Sequence[str],
    *,
    texts: Sequence[str] = (),
    observed_signals: Iterable[str] = (),
    max_hops: int = 1,
    source_hash: str = "",
) -> SourceSurvey:
    """Write targets that a question anchor reaches through source dataflow.

    Facts are extracted through the configured source backend so the survey
    uses the same callable/member identities as DAG construction.

    Selection is structural, in dataflow hops from an anchor:

    * hop 0 — the write READS an anchor (a parameter's declared member,
      an observed signal, or any write inside an anchored callable):
      these are the decision sites where the questioned quantity is
      computed;
    * hop 1..``max_hops`` — writes one dataflow step away: producers of
      what an anchored write reads, and consumers that read what it
      wrote.

    Writes that resolve to a published topic field are kept whenever they
    are reached, since a published placement is how the questioned
    quantity appears in the log. With no anchor at all the survey reports
    every write target of the ranked files — honest breadth rather than a
    silent cut.
    """
    file_list = dedupe_keep_order([str(f) for f in files if f])
    if not file_list:
        return SourceSurvey()

    resolver = SourceExpansionResolver(profiler, source_hash)
    fact_files = dedupe_keep_order(
        [
            *file_list,
            *(
                companion
                for file_path in file_list
                for companion in resolver.companion_files(file_path)
            ),
        ]
    )
    facts = [
        extract_facts_for_file(profiler, file_path, source_hash)
        for file_path in fact_files
    ]
    inputs = dag_inputs_from_facts(facts)
    assignments = list(inputs.bindings)
    if not assignments:
        return SourceSurvey()

    declared_parameters: dict[str, set[str]] = {}
    for facts_entry in facts:
        for parameter in facts_entry.referenced_parameters:
            ref = _as_dict(parameter)
            name = str(ref.get("name") or "")
            if not name:
                continue
            members = declared_parameters.setdefault(name, set())
            member = str(ref.get("member") or "")
            if member:
                members.add(member)

    known_callables = {
        _bare_callable(a.get("function")) for a in assignments if a.get("function")
    }
    anchors = derive_survey_anchors(
        texts,
        declared_parameters=declared_parameters,
        observed_signals=observed_signals,
        known_callables=known_callables,
    )

    def reads_anchor(assignment: dict[str, Any]) -> bool:
        expression = str(
            assignment.get("source_symbol") or assignment.get("expression") or ""
        )
        if any(member in expression for member in anchors.parameter_members):
            return True
        if any(name in expression for name in anchors.parameters):
            return True
        if any(signal in expression for signal in anchors.signals):
            return True
        return _bare_callable(assignment.get("function")) in anchors.callables

    selected: dict[int, int] = {}  # index -> hop distance
    if anchors:
        for index, assignment in enumerate(assignments):
            if reads_anchor(assignment):
                selected[index] = 0
        for hop in range(1, max_hops + 1):
            frontier = {
                index for index, distance in selected.items() if distance == hop - 1
            }
            if not frontier:
                break
            written = {
                SourceSymbolIdentity.model_validate(
                    assignments[i].get("target_identity") or {}
                ).storage_key()
                for i in frontier
                if assignments[i].get("target_identity")
            }
            read: set[tuple[str, ...]] = set()
            for i in frontier:
                for raw_identity in (
                    assignments[i].get("reference_identities") or {}
                ).values():
                    read.add(
                        SourceSymbolIdentity.model_validate(raw_identity).storage_key()
                    )
            for index, assignment in enumerate(assignments):
                if index in selected:
                    continue
                raw_target_identity = assignment.get("target_identity") or {}
                target_identity = (
                    SourceSymbolIdentity.model_validate(raw_target_identity)
                    if raw_target_identity
                    else None
                )
                reference_keys = {
                    SourceSymbolIdentity.model_validate(raw).storage_key()
                    for raw in (
                        assignment.get("reference_identities") or {}
                    ).values()
                }
                produces_input = bool(
                    target_identity and target_identity.storage_key() in read
                )
                consumes_output = bool(reference_keys & written)
                if produces_input or consumes_output:
                    selected[index] = hop
    else:
        selected = {index: 0 for index in range(len(assignments))}

    per_file: dict[str, dict[tuple[str, ...], SurveyedTarget]] = {}
    for index, distance in sorted(selected.items(), key=lambda kv: (kv[1], kv[0])):
        assignment = assignments[index]
        path = assignment.get("assignment_path") or []
        site = path[0] if path else {}
        file = str((site or {}).get("file") or "")
        symbol = str(
            assignment.get("target_symbol") or assignment.get("target") or ""
        ).strip()
        if not file or not symbol:
            continue
        raw_identity = assignment.get("target_identity") or {}
        identity = (
            SourceSymbolIdentity.model_validate(raw_identity)
            if raw_identity
            else None
        )
        storage_key = (
            identity.storage_key()
            if identity is not None
            else ("external_signal", exact_symbol(symbol))
        )
        source_target_id = hashlib.sha256(
            repr(storage_key).encode("utf-8")
        ).hexdigest()[:24]
        targets = per_file.setdefault(file, {})
        target = targets.get(storage_key)
        if target is None:
            target = SurveyedTarget(
                symbol=symbol,
                source_target_id=source_target_id,
                callable_id=str(assignment.get("callable_id") or ""),
                identity=identity.model_dump() if identity is not None else {},
                function=str(assignment.get("function") or ""),
                line=int((site or {}).get("line") or 0),
                expression=str(
                    assignment.get("source_symbol")
                    or assignment.get("expression")
                    or ""
                )[:120],
                distance=distance,
            )
            targets[storage_key] = target
        target.writes += 1
        target.distance = min(target.distance, distance)
        published = (
            str(assignment.get("target_symbol") or "")
            if assignment.get("synthetic_boundary_transfer")
            and assignment.get("boundary_direction") == "publish"
            else ""
        )
        if published and not target.published_signal:
            target.published_signal = published

    ordered_files = dedupe_keep_order([*file_list, *per_file])
    return SourceSurvey(
        files=[
            SurveyedFile(
                file=file,
                targets=sorted(
                    per_file[file].values(),
                    key=lambda t: (t.distance, -t.writes, t.symbol),
                ),
            )
            for file in ordered_files
            if per_file.get(file)
        ],
        anchors=anchors,
    )


def survey_for_queries(
    profiler: MechanismSourceProfiler,
    queries: Sequence[str],
    *,
    texts: Sequence[str] = (),
    observed_signals: Iterable[str] = (),
    max_files: int = 8,
    source_hash: str = "",
) -> tuple[SourceSurvey, list[str]]:
    """Rank source files for ``queries`` and survey their anchored targets.

    Returns the survey and the ranked file list, so the caller can hand
    the same files to discovery instead of repeating the search.
    """
    query_list = [str(q) for q in queries if str(q).strip()]
    if not query_list:
        return SourceSurvey(), []
    hits = profiler.search_related_source_files(query_list, max_files=max_files)
    files = [hit.file for hit in hits if hit.score > 0]
    survey = survey_source_files(
        profiler,
        files,
        texts=[*texts, *query_list],
        observed_signals=observed_signals,
        source_hash=source_hash,
    )
    return survey, files


# ---------------------------------------------------------------------------
# Terminal validation (Phase 0: trust the question and terminals)
# ---------------------------------------------------------------------------


def split_terminal_qualifier(terminal: str) -> tuple[str, str]:
    """Split ``Class::member`` into ``(qualifier, bare member)``.

    The qualifier is SCOPE, not spelling noise: validation uses it to
    pick the write file whose class family matches, instead of
    discarding it. The bare member is what writers key on at the
    assignment site. Member paths and indices are preserved."""
    text = str(terminal or "").strip()
    qualifier, sep, bare = text.rpartition("::")
    return (qualifier.strip() if sep else "", bare.strip())


def canonicalize_terminal(terminal: str) -> str:
    """The sliceable terminal form: the bare member as written at the
    assignment site. The class qualifier is handled separately as scope
    evidence by :func:`validate_terminal` — see
    :func:`split_terminal_qualifier`."""
    return split_terminal_qualifier(terminal)[1]


@dataclass
class TerminalValidation:
    """Deterministic verdict on one candidate terminal, produced BEFORE
    any DAG is built from it.

    ``status``:

    * ``valid`` — the terminal has write targets in the loaded facts
      (scoped to ``resolved_file`` when one could be determined).
    * ``absent`` — no write target exists in the loaded facts. Observing a
      field in the ULog does not substitute for its source publish operation.
    * ``absent_in_scope`` — write targets exist, but none in the
      declared terminal file's family or module.
    * ``ambiguous`` — write targets span several unrelated locations
      and no terminal file was declared to pick one; slicing would fuse
      foreign modules, so nothing is built.
    """

    terminal: str
    status: str
    logged: bool = False
    write_files: list[str] = field(default_factory=list)
    resolved_file: Optional[str] = None
    resolved_identity: Optional[dict[str, Any]] = None
    reason: Optional[str] = None


def validate_terminal(
    terminal: str,
    bindings: Iterable[dict[str, Any]],
    logged_signals: Iterable[str],
    terminal_file: Optional[str] = None,
    source_structure: Optional[SourceStructureIndex] = None,
    terminal_identity: Optional[SourceSymbolIdentity | dict[str, Any]] = None,
) -> TerminalValidation:
    """Validate a candidate terminal against actual write targets and
    the observed catalogue — never by prompt trust or name shape.

    Matching mirrors the DAG builder's exact written-target lookup, so a
    terminal validated here is one the slicer can act on. Declaration
    identity and an explicit terminal file provide scope; spelling and file
    proximity never choose among unrelated writers. A ``Class::member``
    qualifier selects only a writer whose extracted ownership matches.
    """
    qualifier, canonical = split_terminal_qualifier(terminal)
    structure = source_structure or SourceStructureIndex()
    requested_identity = (
        terminal_identity
        if isinstance(terminal_identity, SourceSymbolIdentity)
        else SourceSymbolIdentity.model_validate(terminal_identity)
        if terminal_identity
        else None
    )
    norm = exact_symbol(canonical)
    if not norm:
        return TerminalValidation(
            terminal=canonical, status="absent", reason="empty terminal"
        )

    logged = norm in {exact_symbol(str(s)) for s in logged_signals if s}

    matches: list[dict[str, Any]] = []
    for binding in bindings:
        target = exact_symbol(
            str(binding.get("target_symbol") or binding.get("target") or "")
        )
        if source_storage_produces_reference(target, norm):
            matches.append(binding)

    all_matches = list(matches)
    writes_per_file: dict[str, int] = {}
    for binding in all_matches:
        file = _DAGBuilder._binding_first_file(binding)
        if file:
            writes_per_file[file] = writes_per_file.get(file, 0) + 1
    write_files = sorted(writes_per_file)

    if not all_matches:
        return TerminalValidation(
            terminal=canonical,
            status="absent",
            logged=logged,
            reason=(
                "observed logged output has no source writer in loaded facts"
                if logged
                else "no write target in loaded facts"
            ),
        )

    def best_file(files: Iterable[str]) -> Optional[str]:
        ranked = sorted(set(files), key=lambda f: (-writes_per_file.get(f, 0), f))
        return ranked[0] if ranked else None

    def binding_identity(binding: dict[str, Any]) -> Optional[SourceSymbolIdentity]:
        raw = binding.get("target_identity")
        if not raw:
            return None
        try:
            return SourceSymbolIdentity.model_validate(raw)
        except (TypeError, ValueError):
            return None

    if requested_identity is not None:
        matches = [
            binding
            for binding in matches
            if (producer := binding_identity(binding)) is not None
            and structure.storage_compatible(requested_identity, producer)
        ]
        if not matches:
            return TerminalValidation(
                terminal=canonical,
                status="absent_in_scope",
                logged=logged,
                write_files=write_files,
                resolved_identity=requested_identity.model_dump(),
                reason="no write target matches the surveyed source storage identity",
            )

    if qualifier and not terminal_file:
        lineage = set(structure.lineage(qualifier))
        class_files = []
        for binding in matches:
            identity = binding.get("target_identity") or {}
            owners = {
                str(identity.get("class_owner") or ""),
                str(identity.get("declaring_class") or ""),
                str(binding.get("function_owner") or ""),
            }
            if owners & lineage:
                file = _DAGBuilder._binding_first_file(binding)
                if file:
                    class_files.append(file)
        if class_files:
            class_file_set = set(class_files)
            matches = [
                binding
                for binding in matches
                if _DAGBuilder._binding_first_file(binding) in class_file_set
            ]
            terminal_file = best_file(class_files)

    if terminal_file:
        def source_unit_related(candidate_file: str) -> bool:
            if candidate_file == terminal_file:
                return True
            return (
                terminal_file in structure.includes.get(candidate_file, set())
                or candidate_file
                in structure.includes.get(terminal_file, set())
            )

        scoped_matches = [
            binding
            for binding in matches
            if source_unit_related(_DAGBuilder._binding_first_file(binding))
        ]
        if not scoped_matches:
            return TerminalValidation(
                terminal=canonical,
                status="absent_in_scope",
                logged=logged,
                write_files=write_files,
                resolved_identity=(
                    requested_identity.model_dump() if requested_identity else None
                ),
                reason=(
                    f"no write target in declared file {terminal_file};"
                    f" written in: {', '.join(write_files[:4])}"
                ),
            )
        # A source identity already supplies the storage scope. Keep every
        # compatible member writer across methods/files; the file is only
        # proof that the proposed survey site exists. Without an identity,
        # the file remains the only safe narrowing evidence.
        if requested_identity is None:
            matches = scoped_matches

    identities: dict[tuple[str, ...], SourceSymbolIdentity] = {}
    for binding in matches:
        identity = binding_identity(binding)
        if identity is not None:
            identities.setdefault(identity.storage_key(), identity)

    if len(identities) == 1 or (
        not identities
        and len({_DAGBuilder._binding_first_file(binding) for binding in matches}) == 1
    ):
        resolved_identity = next(iter(identities.values()), requested_identity)
        return TerminalValidation(
            terminal=canonical,
            status="valid",
            logged=logged,
            write_files=write_files,
            resolved_file=(
                best_file(_DAGBuilder._binding_first_file(binding) for binding in matches)
                if terminal_file
                else None
            ),
            resolved_identity=(
                resolved_identity.model_dump() if resolved_identity is not None else None
            ),
        )
    return TerminalValidation(
        terminal=canonical,
        status="ambiguous",
        logged=logged,
        write_files=write_files,
        reason=(
            f"write targets span {len(identities) or len(write_files)} scoped identities with no terminal"
            f" file declared: {', '.join(write_files[:4])}"
        ),
    )


# ---------------------------------------------------------------------------
# Deterministic discovery fixpoint
# ---------------------------------------------------------------------------


def _definition_queries(symbol: str) -> list[str]:
    """Search queries likely to hit the file that *defines* ``symbol``.

    Unresolved entries can be dotted accessor chains
    (``_scale_check_groundspeed.isAllFinite``); the writable entity is the
    root, so query on an assignment-shaped root pattern. Short roots omit
    the broad assignment query because it carries no useful structural
    selectivity; typed callable resolution handles accessor tails.
    """
    root = symbol.split(".", 1)[0].split("->", 1)[0].strip().strip("&*")
    queries: list[str] = []
    if len(root) >= 4:
        queries.append(f"{root} =")
    if "." in symbol or "->" in symbol:
        # Accessor gap (``_obj.getThing``): the definition lives under the
        # METHOD name, not the receiver — ``root =`` greps the wrong thing.
        tail = symbol.replace("->", ".").rsplit(".", 1)[-1].strip().strip("()")
        if len(tail) >= 4:
            queries.append(f"{tail}(")
    return queries


def _gap_definition_files(
    profiler: MechanismSourceProfiler,
    symbols: Iterable[str],
    *,
    max_files_per_gap: int = 2,
    max_matched_files: int = 10,
    max_total: int = 16,
) -> list[str]:
    """Return files containing exact assignment definitions for ``symbols``.

    The numeric arguments remain for API compatibility but no longer limit
    correctness. Search examines every hit and admits a file only after the
    profiler extracts an exact write target from that same file.
    """
    _ = (max_files_per_gap, max_matched_files, max_total)
    files: list[str] = []
    for symbol in sorted({str(s) for s in symbols if s}):
        canonical = exact_symbol(symbol)
        for query in _definition_queries(symbol):
            hits = profiler.search_related_source_files(
                [query], max_files=None, expand_query_tokens=False
            )
            for hit in hits:
                assignments = profiler.extract_source_assignments_from_source(
                    [hit.file]
                )
                if any(
                    assignment.file == hit.file
                    and symbol_produces_reference(
                        exact_symbol(assignment.target), canonical
                    )
                    for assignment in assignments
                ):
                    files.append(hit.file)
    return dedupe_keep_order(files)


def make_helper_body_provider(
    profiler: MechanismSourceProfiler,
    fetched_files: list[str],
    *,
    structure: Optional[SourceStructureIndex] = None,
    resolver: Optional[SourceExpansionResolver] = None,
) -> Callable[..., Any]:
    """Load exact callable definitions without admitting ranked hit files.

    Search returns every candidate. Only files from which the profiler
    extracts a matching helper definition are retained; receiver ownership
    and arity further narrow the result when the DAG supplies call context.
    """
    source_structure = structure or SourceStructureIndex()
    source_resolver = resolver or SourceExpansionResolver(profiler, "provider")

    def provider(
        helper_name: str,
        reference: Optional[UnresolvedSourceReference] = None,
    ) -> Any:
        call_reference = reference or UnresolvedSourceReference(
            symbol=helper_name,
            kind="callable",
        )
        resolved: list[Any] = []
        for candidate in source_resolver.resolve(
            call_reference, source_structure
        ):
            exact = [
                helper
                for helper in candidate.facts.helper_expressions
                if helper.file == candidate.file
                and helper.name.rsplit("::", 1)[-1] == helper_name
                and (
                    not candidate.matched_identity
                    or helper.callable_id == candidate.matched_identity
                )
            ]
            if not exact:
                continue
            if candidate.file not in fetched_files:
                fetched_files.append(candidate.file)
            resolved.extend(exact)
        return resolved

    return provider


@dataclass
class DiscoveryRound:
    """Per-round trace of the fixpoint, kept for judge/debug consumption."""

    index: int
    new_files: list[str]
    unresolved_symbols: list[str]
    vertices: int
    edges: int


@dataclass
class DiscoveryResult:
    dag: Optional[MechanismDAG]
    inputs: DAGInputs
    files_loaded: list[str]
    rounds: list[DiscoveryRound]
    terminal_validation: Optional[TerminalValidation] = None


def _reachable_source_sites(dag: MechanismDAG) -> set[tuple[str, int]]:
    """Return source sites on backward paths from surviving terminal writes."""
    vertices = {vertex.id: vertex for vertex in dag.vertices}
    terminal_ids = {
        vertex.id
        for vertex in dag.vertices
        if vertex.kind == "operation"
        and bool((vertex.metadata or {}).get("is_terminal"))
    }
    incoming: dict[str, list[str]] = {}
    for edge in dag.edges:
        incoming.setdefault(edge.target_id, []).append(edge.source_id)
    reachable = set(terminal_ids)
    pending = list(terminal_ids)
    while pending:
        current = pending.pop()
        for predecessor in incoming.get(current, ()):
            if predecessor not in reachable:
                reachable.add(predecessor)
                pending.append(predecessor)
    return {
        (str(vertex.file or ""), int(vertex.line))
        for vertex_id, vertex in vertices.items()
        if vertex_id in reachable and vertex.file and vertex.line is not None
    }


def _reachable_frontier_references(
    references: Sequence[UnresolvedSourceReference],
    full_dag: MechanismDAG,
    feasible_dag: MechanismDAG,
) -> list[UnresolvedSourceReference]:
    """Drop only references proven to originate solely on removed paths."""
    full_sites = _reachable_source_sites(full_dag)
    feasible_sites = _reachable_source_sites(feasible_dag)
    kept: list[UnresolvedSourceReference] = []
    for reference in references:
        site = (
            str(reference.file or ""),
            int(reference.line) if reference.line is not None else 0,
        )
        if site[0] and site[1] and site in full_sites and site not in feasible_sites:
            continue
        kept.append(reference)
    return kept


def discover_mechanism_dag(
    profiler: MechanismSourceProfiler,
    cache_root: Union[str, Path],
    seeds: Sequence[str],
    terminal: str,
    source_hash: str,
    *,
    source_root: Optional[Union[str, Path]] = None,
    terminal_file: Optional[str] = None,
    terminal_identity: Optional[SourceSymbolIdentity | dict[str, Any]] = None,
    max_rounds: int = 3,
    max_files_per_round: int = 8,
    max_files_total: int = 24,
    inventory: Optional[dict[str, Any]] = None,
    schema_signals: Optional[Iterable[str]] = None,
    logged_signals: Optional[Iterable[str]] = None,
    parameter_values: Optional[dict[str, Any]] = None,
    enum_registry: Optional[dict[str, dict[str, Any]]] = None,
    preranked_files: Optional[Sequence[str]] = None,
    round_annotator: Optional[Callable[[MechanismDAG], MechanismDAG]] = None,
) -> DiscoveryResult:
    """Build a DAG by exact, provenance-checked fixed-point expansion.

    Search results are candidates only. A file joins the source index after
    it proves an exact terminal, symbol, callable, class, or companion
    relationship. The graph itself admits only entities reached by the DAG's
    backward walk. Expansion has no file, round, or gap budget; it terminates
    when every structured frontier item has been resolved, rejected, or
    visited without discovering a new source entity.

    Every terminal — seeder-proposed, judge-proposed, or replayed from
    the Layer 4 cache — passes :func:`validate_terminal` against the
    loaded exact-writer facts before anything is built from it. An absent,
    ambiguous, or out-of-scope terminal stops with the structured reason in
    ``terminal_validation``; building would invent or fuse source identity.

    ``max_rounds``, ``max_files_per_round``, and ``max_files_total`` remain in
    the API for compatibility but are intentionally ignored. They previously
    made source-search order affect correctness.
    """
    _ = (cache_root, source_root, max_rounds, max_files_per_round, max_files_total)
    terminal_as_given = str(terminal or "").strip()
    terminal = canonicalize_terminal(terminal)
    requested_terminal_identity = (
        terminal_identity
        if isinstance(terminal_identity, SourceSymbolIdentity)
        else SourceSymbolIdentity.model_validate(terminal_identity)
        if terminal_identity
        else None
    )
    if logged_signals is not None:
        # Materialized once: consumed by per-round validation AND the
        # builder, so a one-shot iterable must not exhaust in between.
        logged_signals = {str(s) for s in logged_signals}

    if enum_registry is None:
        # Schema-derived message enums, flattened PER MESSAGE (the scope
        # the reference itself carries) — loaded once per discovery.
        from flight_log_agent.px4.msg_schema import load_px4_declared_constant_registry

        try:
            enum_registry = load_px4_declared_constant_registry(profiler.source)
        except Exception:
            enum_registry = {}

    resolver = SourceExpansionResolver(profiler, source_hash)
    loaded: list[str] = []
    facts_by_file: dict[str, SourceFileFacts] = {}
    rounds: list[DiscoveryRound] = []
    dag: Optional[MechanismDAG] = None
    inputs = DAGInputs()
    validation: Optional[TerminalValidation] = None
    visited: set[tuple[Any, ...]] = set()

    def bindings_write_terminal(bindings: Iterable[dict[str, Any]]) -> bool:
        canonical = exact_symbol(terminal)
        return any(
            source_storage_produces_reference(
                exact_symbol(
                    str(binding.get("target_symbol") or binding.get("target") or "")
                ),
                canonical,
            )
            for binding in bindings
        )

    def candidate_bundle(file_path: str) -> tuple[list[str], DAGInputs]:
        """Load one candidate and its source companions for proof only."""
        files = dedupe_keep_order(
            [file_path, *resolver.companion_files(file_path)]
        )
        return files, dag_inputs_from_facts(
            resolver.facts_for(candidate) for candidate in files
        )

    def admit_terminal_candidate(file_path: str) -> list[str]:
        files, candidate_inputs = candidate_bundle(file_path)
        return files if bindings_write_terminal(candidate_inputs.bindings) else []

    explicit_files = [str(terminal_file)] if terminal_file else []
    explicit_companions = [
        companion
        for file_path in explicit_files
        for companion in resolver.companion_files(file_path)
    ]
    terminal_queries = _definition_queries(terminal)
    parsed_terminal = parse_signal_reference(exact_symbol(terminal))
    if parsed_terminal is not None:
        topic = parsed_terminal[0]
        terminal_queries.extend(
            [f"ORB_ID({topic})", f"ORB_ID::{topic}"]
        )
    if not terminal_queries:
        terminal_queries.append(terminal)
    candidate_files = dedupe_keep_order(
        [
            *explicit_files,
            *explicit_companions,
            *(str(file_path) for file_path in (preranked_files or ()) if file_path),
        ]
    )
    # Seeder queries remain useful for the source survey, but they are not
    # graph provenance and therefore cannot admit round-zero files.
    _ = seeds
    pending: list[str] = []
    explicit_set = set(explicit_files) | set(explicit_companions)
    for file_path in candidate_files:
        admitted = admit_terminal_candidate(file_path)
        if file_path in explicit_set:
            pending.append(file_path)
            pending.extend(resolver.companion_files(file_path))
        elif admitted:
            pending.extend(admitted)
    if not terminal_file or (
        requested_terminal_identity is not None
        and requested_terminal_identity.kind in {"member", "global"}
    ):
        # A ranked survey is not an exhaustive declaration index. Unscoped
        # terminals must inspect every exact writer so ambiguity cannot depend
        # on ranking order. A declared terminal file already supplies scope.
        for hit in profiler.search_related_source_files(
            terminal_queries,
            max_files=None,
            expand_query_tokens=False,
        ):
            pending.extend(admit_terminal_candidate(hit.file))
    pending = dedupe_keep_order(pending)

    index = 0
    first_pass = True
    while first_pass or pending:
        first_pass = False
        new_files = [file_path for file_path in pending if file_path not in facts_by_file]
        pending = []
        for file_path in new_files:
            facts = resolver.facts_for(file_path)
            facts_by_file[file_path] = facts
            loaded.append(file_path)

        inputs = dag_inputs_from_facts(facts_by_file.values())

        # Gate the build on terminal validation. A verdict scoped to a
        # resolved file is locked — later rounds load foreign files whose
        # same-named writers cannot hijack a scoped slice. An UNSCOPED
        # valid verdict is re-checked every round: writers arriving from
        # gap searches can reveal the terminal as genuinely ambiguous.
        locked = (
            validation is not None
            and validation.status == "valid"
            and bool(validation.resolved_file or terminal_file)
        )
        if not locked:
            validation = validate_terminal(
                terminal_as_given,
                inputs.bindings,
                logged_signals or (),
                terminal_file,
                source_structure=inputs.structure,
                terminal_identity=requested_terminal_identity,
            )
        if validation.status != "valid":
            dag = None
            rounds.append(
                DiscoveryRound(
                    index=index,
                    new_files=list(new_files),
                    unresolved_symbols=[],
                    vertices=0,
                    edges=0,
                )
            )
            break

        fetched_files: list[str] = []
        provider = make_helper_body_provider(
            profiler,
            fetched_files,
            structure=inputs.structure,
            resolver=resolver,
        )
        dag = build_mechanism_dag(
            inputs.bindings,
            terminal,
            inventory=inventory,
            schema_signals=schema_signals,
            logged_signals=logged_signals,
            helper_expressions=inputs.helper_expressions,
            helper_body_provider=provider,
            parameter_predicates=inputs.parameter_predicates,
            parameter_values=parameter_values,
            parameter_names=inputs.parameter_names,
            parameter_bindings=inputs.parameter_bindings,
            terminal_file=validation.resolved_file or terminal_file,
            terminal_identity=validation.resolved_identity,
            call_statements=inputs.call_statements,
            boundary_bindings=inputs.boundary_bindings,
            enum_registry=enum_registry,
            source_structure=inputs.structure,
        )

        unresolved = set(dag.unresolved_symbols)
        rounds.append(
            DiscoveryRound(
                index=index,
                new_files=list(new_files),
                unresolved_symbols=sorted(unresolved),
                vertices=len(dag.vertices),
                edges=len(dag.edges),
            )
        )

        frontier_dag = (
            round_annotator(dag)
            if round_annotator is not None
            else evaluate_feasibility(
                dag,
                parameter_values=parameter_values,
                prune_dead=True,
            )
        )
        references = _reachable_frontier_references(
            list(dag.unresolved_references), dag, frontier_dag
        )
        known_classes = set(inputs.structure.direct_bases)
        for owner, bases in inputs.structure.direct_bases.items():
            owner_record = next(
                (
                    item
                    for facts in facts_by_file.values()
                    for item in facts.classes
                    if item.name == owner
                ),
                None,
            )
            for base in bases:
                if base not in known_classes:
                    references.append(
                        UnresolvedSourceReference(
                            symbol=base,
                            kind="class",
                            file=owner_record.file if owner_record else "",
                            line=owner_record.line if owner_record else None,
                            class_owner=owner,
                        )
                    )

        next_files: list[str] = []
        for reference in references:
            if reference_receiver_is_source_boundary(
                reference,
                inputs.boundary_bindings,
                inputs.structure,
            ):
                continue
            key = resolver.resolution_key(reference, inputs.structure)
            if key in visited:
                continue
            visited.add(key)
            for candidate in resolver.resolve(reference, inputs.structure):
                next_files.append(candidate.file)
                next_files.extend(resolver.companion_files(candidate.file))
        for file_path in fetched_files:
            next_files.append(file_path)
            next_files.extend(resolver.companion_files(file_path))
        pending = [
            file_path
            for file_path in dedupe_keep_order(next_files)
            if file_path not in facts_by_file
        ]
        if not pending:
            break
        index += 1

    return DiscoveryResult(
        dag=dag,
        inputs=inputs,
        files_loaded=loaded,
        rounds=rounds,
        terminal_validation=validation,
    )
