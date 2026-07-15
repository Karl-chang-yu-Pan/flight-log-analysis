"""Tree-sitter C/C++ source-fact extraction for mechanism DAG discovery.

The legacy profiler remains available while this backend is validated against
real logs.  This module owns structural parsing only: expression normalization
and deterministic helper lowering are shared with the existing profiler so the
two backends differ at the syntax boundary, not in downstream DAG semantics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from tree_sitter import Language, Node, Parser, Tree
import tree_sitter_cpp

from flight_log_agent.expression_math import (
    canonical_math_function_name,
    is_safe_math_function_name,
)
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
    _HelperLoweringFailed,
    split_source_field,
)


_CPP_LANGUAGE = Language(tree_sitter_cpp.language())


@dataclass(frozen=True)
class _ControlTerm:
    expression: str
    line: int
    site_id: str


@dataclass
class _Callable:
    name: str
    owner: Optional[str]
    file: str
    line: int
    end_line: int
    callable_id: str
    parameters: list[str]
    parameter_types: list[str]
    return_type: Optional[str]
    node: Node
    body: Node
    evidence: str
    parent_callable_id: Optional[str] = None
    captures: list[str] = field(default_factory=list)
    capture_arguments: list[str] = field(default_factory=list)
    is_lambda: bool = False
    capture_exact: bool = True


@dataclass
class _ParsedUnit:
    file: str
    source: bytes
    tree: Tree
    classes: list[SourceClassRef] = field(default_factory=list)
    members: list[SourceMemberRef] = field(default_factory=list)
    callables: list[_Callable] = field(default_factory=list)
    includes: list[SourceIncludeRef] = field(default_factory=list)
    class_ranges: list[tuple[int, int, str]] = field(default_factory=list)
    namespace_ranges: list[tuple[int, int, str]] = field(default_factory=list)

    def text(self, node: Optional[Node]) -> str:
        if node is None:
            return ""
        return self.source[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def line(self, node: Node) -> int:
        return int(node.start_point.row) + 1

    def site_id(self, node: Node) -> str:
        return f"{self.file}:{node.start_byte}:{node.end_byte}:{node.type}"

    def evidence(self, node: Node) -> str:
        return " ".join(self.text(node).split())

    def class_owner(self, node: Node) -> Optional[str]:
        candidates = [
            item
            for item in self.class_ranges
            if item[0] <= node.start_byte and node.end_byte <= item[1]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1] - item[0])[2]

    def namespace_owner(self, node: Node) -> Optional[str]:
        candidates = [
            item
            for item in self.namespace_ranges
            if item[0] <= node.start_byte and node.end_byte <= item[1]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1] - item[0])[2]


@dataclass
class _SourceContext:
    units: list[_ParsedUnit]
    classes: dict[str, SourceClassRef] = field(default_factory=dict)
    members: dict[tuple[str, str], SourceMemberRef] = field(default_factory=dict)
    callables: dict[str, _Callable] = field(default_factory=dict)
    parameter_members: dict[tuple[str, str], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for unit in self.units:
            for item in unit.classes:
                self.classes[item.name] = item
            for item in unit.members:
                self.members[(item.owner, item.name)] = item
            for item in unit.callables:
                self.callables[item.callable_id] = item

    def lineage(self, owner: Optional[str]) -> list[str]:
        if not owner:
            return []
        ordered: list[str] = []
        pending = [owner]
        seen: set[str] = set()
        while pending:
            current = pending.pop(0)
            if not current or current in seen:
                continue
            seen.add(current)
            ordered.append(current)
            item = self.classes.get(current)
            if item is not None:
                pending.extend(item.bases)
        return ordered

    def declaring_member(self, owner: Optional[str], name: str) -> Optional[SourceMemberRef]:
        for candidate in self.lineage(owner):
            item = self.members.get((candidate, name))
            if item is not None:
                return item
        return None

    def parameter_for_member(self, owner: Optional[str], name: str) -> Optional[str]:
        for candidate in self.lineage(owner):
            value = self.parameter_members.get((candidate, name))
            if value:
                return value
        return None


@dataclass
class _ExtractionState:
    unit: _ParsedUnit
    context: _SourceContext
    callable: _Callable
    struct_variables: dict[str, str]
    assignments: list[SourceAssignmentRef] = field(default_factory=list)
    calls: list[FunctionCallRef] = field(default_factory=list)
    branches: list[BranchConditionRef] = field(default_factory=list)
    return_paths: list[dict[str, Any]] = field(default_factory=list)
    seen_assignments: set[tuple[int, int]] = field(default_factory=set)
    seen_calls: set[tuple[int, int]] = field(default_factory=set)
    lambda_bindings: dict[str, _Callable] = field(default_factory=dict)
    storage_aliases: dict[str, str] = field(default_factory=dict)
    alias_capable_symbols: set[str] = field(default_factory=set)
    reference_alias_symbols: set[str] = field(default_factory=set)


def _walk(node: Node) -> Iterator[Node]:
    yield node
    for child in node.named_children:
        yield from _walk(child)


def _walk_operations(node: Node) -> Iterator[Node]:
    """Walk one callable without descending into nested lambda callables."""
    yield node
    for child in node.named_children:
        if child.type == "lambda_expression":
            continue
        yield from _walk_operations(child)


def _last_identifier(node: Optional[Node]) -> Optional[Node]:
    if node is None:
        return None
    identifiers = [
        child
        for child in _walk(node)
        if child.type
        in {
            "identifier",
            "field_identifier",
            "type_identifier",
            "namespace_identifier",
        }
    ]
    return identifiers[-1] if identifiers else None


def _field_or_self(node: Node, field_name: str) -> Node:
    child = node.child_by_field_name(field_name)
    return child if child is not None else node


def _declaration_declarators(node: Node) -> list[Node]:
    direct = list(node.children_by_field_name("declarator"))
    if direct:
        return [_field_or_self(child, "declarator") for child in direct]
    return [
        _field_or_self(child, "declarator")
        for child in node.named_children
        if child.type == "init_declarator"
    ]


def _strip_initializer_delimiters(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and (text[0], text[-1]) in {
        ("(", ")"),
        ("{", "}"),
        ("[", "]"),
    }:
        return text[1:-1].strip()
    return text


def _and(left: str, right: str) -> str:
    if left == "false" or right == "false":
        return "false"
    if left == "true":
        return right
    if right == "true":
        return left
    if left == right:
        return left
    return f"({left}) && ({right})"


def _or(left: str, right: str) -> str:
    if left == "true" or right == "true":
        return "true"
    if left == "false":
        return right
    if right == "false":
        return left
    if left == right:
        return left
    return f"({left}) || ({right})"


def _not(value: str) -> str:
    if value == "true":
        return "false"
    if value == "false":
        return "true"
    return f"!({value})"


class TreeSitterSourceExtractor:
    """Extract one backend-complete ``SourceFileFacts`` payload from C++ ASTs."""

    def __init__(self, profiler: MechanismSourceProfiler) -> None:
        self.profiler = profiler
        self.parser = Parser(_CPP_LANGUAGE)
        self._units: dict[str, Optional[_ParsedUnit]] = {}
        self._fully_structured_units: set[str] = set()
        self._paths_by_file: dict[str, list[Path]] = {}

    def _source_family_paths(self, file_path: str) -> list[Path]:
        cached = self._paths_by_file.get(file_path)
        if cached is not None:
            return list(cached)
        primary_path = self.profiler._resolve_file(file_path)
        paths = self.profiler._expand_companion_files([primary_path])
        if all(self.profiler._rel(path) != file_path for path in paths):
            paths.insert(0, primary_path)
        self._paths_by_file[file_path] = list(paths)
        return paths

    def extract(self, file_path: str, source_hash: str):
        # Imported lazily to keep the legacy profiler usable when the optional
        # parser dependency has not been installed yet.
        from flight_log_agent.px4.source_facts_cache import SourceFileFacts

        paths = self._source_family_paths(file_path)
        units = [unit for path in paths if (unit := self._parse(path)) is not None]
        primary = next((unit for unit in units if unit.file == file_path), None)
        if primary is None:
            return SourceFileFacts(
                file=file_path,
                source_hash=source_hash,
                parser_backend="tree_sitter",
                parse_diagnostics={"error": "source file could not be read"},
            )

        self._resolve_class_bases(units)
        context = _SourceContext(units)
        self._index_parameter_members(context)
        states = [self._extract_callable(primary, context, item) for item in primary.callables]
        global_assignments = self._extract_global_constants(primary)
        topics = self._extract_topics(primary, context, states)
        params = self._extract_parameters(primary, context)
        assignments = [
            *global_assignments,
            *(item for state in states for item in state.assignments),
        ]
        calls = [item for state in states for item in state.calls]
        helpers = [self._helper_from_state(state) for state in states]
        branches = [item for state in states for item in state.branches]
        parameter_predicates = self._parameter_predicates(
            primary, context, params
        )
        assigned_fields, read_fields = self._field_refs(
            primary, context, states
        )
        diagnostics = self._diagnostics(primary)

        return SourceFileFacts(
            file=file_path,
            source_hash=source_hash,
            parser_backend="tree_sitter",
            parse_diagnostics=diagnostics,
            published_topics=self._dedupe(topics["publish"]),
            subscribed_topics=self._dedupe(topics["subscribe"]),
            unknown_direction_topics=self._dedupe(topics["unknown"]),
            referenced_parameters=self._dedupe(params),
            assigned_fields=self._dedupe(assigned_fields),
            read_fields=self._dedupe(read_fields),
            source_assignments=self._dedupe(assignments),
            function_calls=self._dedupe(calls),
            helper_expressions=self._dedupe(helpers),
            branch_conditions=self._dedupe(branches),
            parameter_predicates=self._dedupe(parameter_predicates),
            classes=primary.classes,
            members=primary.members,
            callables=[
                SourceCallableRef(
                    name=item.name,
                    owner=item.owner,
                    file=item.file,
                    line=item.line,
                    end_line=item.end_line,
                    callable_id=item.callable_id,
                    parameters=item.parameters,
                    parameter_types=item.parameter_types,
                    return_type=item.return_type,
                )
                for item in primary.callables
            ],
            includes=primary.includes,
        )

    def extract_admission(
        self,
        file_path: str,
        source_hash: str,
        *,
        kind: str,
        symbol: str,
    ):
        """Extract only facts needed to admit one expansion candidate.

        This uses the same parsed AST and source-fact constructors as full
        extraction. Callable/class admission needs structure only. Symbol and
        constant admission extracts assignments only from callables whose AST
        text contains the requested identifier, plus source-level constants.
        No candidate count or source-search budget is applied.
        """
        from flight_log_agent.px4.source_facts_cache import SourceFileFacts

        paths = self._source_family_paths(file_path)
        units = [
            unit
            for path in paths
            if (unit := self._parse(path, full_structure=False)) is not None
        ]
        primary = next((unit for unit in units if unit.file == file_path), None)
        if primary is None:
            return SourceFileFacts(
                file=file_path,
                source_hash=source_hash,
                parser_backend="tree_sitter:admission",
                parse_diagnostics={"error": "source file could not be read"},
            )

        self._resolve_class_bases(units)
        assignments: list[SourceAssignmentRef] = []
        if kind in {"symbol", "constant"}:
            requested = str(symbol or "").replace("->", ".")
            root_match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", requested)
            requested_root = root_match.group(0) if root_match else ""
            for callable_item in primary.callables:
                body_text = primary.text(callable_item.body)
                if requested_root and not re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(requested_root)}(?![A-Za-z0-9_])",
                    body_text,
                ):
                    continue
                assignments.extend(
                    self._admission_assignments(primary, callable_item)
                )
            assignments.extend(self._extract_global_constants(primary))

        return SourceFileFacts(
            file=file_path,
            source_hash=source_hash,
            parser_backend="tree_sitter:admission",
            # Detailed diagnostics walk the entire AST and are part of full
            # extraction. Admission only needs to fail closed on a parser
            # error; exact candidates are fully extracted immediately after.
            parse_diagnostics={
                "has_error": bool(primary.tree.root_node.has_error)
            },
            source_assignments=self._dedupe(assignments),
            classes=primary.classes,
            members=primary.members,
            callables=[
                SourceCallableRef(
                    name=item.name,
                    owner=item.owner,
                    file=item.file,
                    line=item.line,
                    end_line=item.end_line,
                    callable_id=item.callable_id,
                    parameters=item.parameters,
                    parameter_types=item.parameter_types,
                    return_type=item.return_type,
                )
                for item in primary.callables
            ],
            includes=primary.includes,
        )

    def _admission_assignments(
        self, unit: _ParsedUnit, callable_item: _Callable
    ) -> list[SourceAssignmentRef]:
        """Collect raw AST write targets without full callable extraction."""
        assignments: list[SourceAssignmentRef] = []
        seen: set[tuple[int, int]] = set()
        for node in _walk_operations(callable_item.body):
            target_node: Optional[Node] = None
            value_node: Optional[Node] = None
            operator = "="
            if node.type == "assignment_expression":
                target_node = node.child_by_field_name("left")
                value_node = node.child_by_field_name("right")
                if target_node is not None and value_node is not None:
                    operator = unit.source[
                        target_node.end_byte : value_node.start_byte
                    ].decode("utf-8", errors="replace").strip()
            elif node.type == "init_declarator":
                declarator = node.child_by_field_name("declarator")
                target_node = _last_identifier(declarator)
                value_node = node.child_by_field_name("value")
            if target_node is None or value_node is None:
                continue
            site = (node.start_byte, node.end_byte)
            if site in seen:
                continue
            seen.add(site)
            target = self._canonical_symbol(unit.text(target_node))
            if not target:
                continue
            expression = self.profiler._normalize_source_expression(
                _strip_initializer_delimiters(unit.text(value_node))
            )
            assignments.append(
                SourceAssignmentRef(
                    target=target,
                    expression=expression,
                    assignment_operator=operator,
                    file=unit.file,
                    line=unit.line(node),
                    evidence=unit.evidence(node),
                    function=callable_item.name,
                    callable_id=callable_item.callable_id,
                    function_parameters=list(callable_item.parameters),
                    source_site_id=unit.site_id(node),
                )
            )
        return assignments

    def _parse(
        self, path: Path, *, full_structure: bool = True
    ) -> Optional[_ParsedUnit]:
        file = self.profiler._rel(path)
        if file in self._units:
            unit = self._units[file]
            if unit is None:
                return None
        else:
            text = self.profiler._read_text(path)
            if text is None:
                self._units[file] = None
                return None
            source = text.encode("utf-8")
            unit = _ParsedUnit(
                file=file, source=source, tree=self.parser.parse(source)
            )
            self._units[file] = unit
            self._extract_structure(
                unit, path, include_lambdas=full_structure
            )
            if full_structure:
                self._fully_structured_units.add(file)
            return unit
        if full_structure and file not in self._fully_structured_units:
            unit.classes.clear()
            unit.members.clear()
            unit.callables.clear()
            unit.includes.clear()
            unit.class_ranges.clear()
            unit.namespace_ranges.clear()
            self._extract_structure(unit, path, include_lambdas=True)
            self._fully_structured_units.add(file)
        return unit

    def _extract_structure(
        self,
        unit: _ParsedUnit,
        path: Path,
        *,
        include_lambdas: bool,
    ) -> None:
        self._extract_classes(unit, unit.tree.root_node, None, ())
        existing_members = {(item.owner, item.name) for item in unit.members}
        for name, member, owner, template in self._parameter_declarations(unit):
            if not owner or not member or (owner, member) in existing_members:
                continue
            unit.members.append(
                SourceMemberRef(
                    name=member,
                    owner=owner,
                    type=unit.text(template),
                    file=unit.file,
                    line=unit.line(template),
                )
            )
            existing_members.add((owner, member))
        self._extract_callables(unit, include_lambdas=include_lambdas)
        if not include_lambdas:
            return
        for node in _walk(unit.tree.root_node):
            if node.type != "preproc_include":
                continue
            path_node = node.child_by_field_name("path")
            include = unit.text(path_node).strip('<>"')
            candidates = [path.parent / include, Path(include)]
            resolved = next(
                (
                    candidate.as_posix()
                    for candidate in candidates
                    if not candidate.is_absolute()
                    and ".." not in candidate.parts
                    and self.profiler.source.file_exists(candidate.as_posix())
                ),
                None,
            )
            if resolved:
                unit.includes.append(
                    SourceIncludeRef(
                        file=unit.file,
                        included_file=resolved,
                        line=unit.line(node),
                    )
                )

    @staticmethod
    def _resolve_class_bases(units: Sequence[_ParsedUnit]) -> None:
        """Resolve base names against source-declared lexical scopes."""
        declared = {
            item.name
            for unit in units
            for item in unit.classes
            if item.name
        }
        for unit in units:
            resolved_classes: list[SourceClassRef] = []
            for item in unit.classes:
                owner_parts = item.name.split("::")[:-1]
                bases: list[str] = []
                for raw_base in item.bases:
                    base = raw_base.strip().lstrip(":")
                    base = re.sub(r"^virtual\s+", "", base)
                    base_name = base.split("<", 1)[0].strip()
                    candidates = [
                        "::".join([*owner_parts[:depth], base_name])
                        for depth in range(len(owner_parts), -1, -1)
                        if base_name
                    ]
                    matches = [candidate for candidate in candidates if candidate in declared]
                    resolved = matches[0] if matches else base
                    if resolved and resolved not in bases:
                        bases.append(resolved)
                resolved_classes.append(item.model_copy(update={"bases": bases}))
            unit.classes = resolved_classes

    def _extract_classes(
        self,
        unit: _ParsedUnit,
        node: Node,
        lexical_owner: Optional[str],
        namespace_scope: tuple[str, ...],
    ) -> None:
        if node.type == "namespace_definition":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                components = (f"(anonymous@{unit.file})",)
            else:
                components = tuple(
                    part
                    for part in unit.text(name_node).replace(" ", "").split("::")
                    if part
                )
            nested_scope = (*namespace_scope, *components)
            body = node.child_by_field_name("body")
            if body is not None and nested_scope:
                unit.namespace_ranges.append(
                    (body.start_byte, body.end_byte, "::".join(nested_scope))
                )
                for child in body.named_children:
                    self._extract_classes(
                        unit, child, lexical_owner, nested_scope
                    )
            return

        owner = lexical_owner
        if node.type in {"class_specifier", "struct_specifier"}:
            name_node = node.child_by_field_name("name")
            short_name = unit.text(name_node).strip()
            if short_name:
                namespace = "::".join(namespace_scope)
                owner = (
                    f"{lexical_owner}::{short_name}"
                    if lexical_owner
                    else f"{namespace}::{short_name}"
                    if namespace
                    else short_name
                )
                bases: list[str] = []
                base_clause = next(
                    (child for child in node.named_children if child.type == "base_class_clause"),
                    None,
                )
                if base_clause is not None:
                    for child in base_clause.named_children:
                        if child.type == "access_specifier":
                            continue
                        value = "".join(unit.text(child).split())
                        if value and value not in bases:
                            bases.append(value)
                ref = SourceClassRef(
                    name=owner,
                    file=unit.file,
                    line=unit.line(node),
                    end_line=int(node.end_point.row) + 1,
                    bases=bases,
                )
                unit.classes.append(ref)
                unit.class_ranges.append((node.start_byte, node.end_byte, owner))
                body = node.child_by_field_name("body")
                if body is not None:
                    for child in body.named_children:
                        if child.type == "field_declaration":
                            unit.members.extend(
                                self._members_from_declaration(unit, child, owner)
                            )
        for child in node.named_children:
            self._extract_classes(unit, child, owner, namespace_scope)

    def _members_from_declaration(
        self, unit: _ParsedUnit, node: Node, owner: str
    ) -> list[SourceMemberRef]:
        if any(child.type == "function_declarator" for child in _walk(node)):
            return []
        type_node = node.child_by_field_name("type")
        type_text = unit.text(type_node).strip() or None
        refs: list[SourceMemberRef] = []
        declarators = _declaration_declarators(node)
        for declarator in declarators:
            name_node = _last_identifier(declarator)
            name = unit.text(name_node).strip()
            if not name:
                continue
            refs.append(
                SourceMemberRef(
                    name=name,
                    owner=owner,
                    type=type_text,
                    file=unit.file,
                    line=unit.line(declarator),
                )
            )
        return refs

    def _extract_callables(
        self, unit: _ParsedUnit, *, include_lambdas: bool = True
    ) -> None:
        # Materialize before traversing each declarator. py-tree-sitter's
        # child cursors are not safe to traverse re-entrantly from a suspended
        # recursive generator on some grammar/backend combinations.
        definitions = [
            node
            for node in _walk(unit.tree.root_node)
            if node.type == "function_definition"
        ]
        for node in definitions:
            body = node.child_by_field_name("body")
            declarator = node.child_by_field_name("declarator")
            function_declarator = next(
                (
                    child
                    for child in _walk(declarator) if child.type == "function_declarator"
                ),
                None,
            )
            if body is None or function_declarator is None:
                continue
            name_node = function_declarator.child_by_field_name("declarator")
            raw_name = unit.text(name_node).strip()
            lexical_owner = unit.class_owner(node)
            namespace_owner = unit.namespace_owner(node)
            qualified_owner, separator, short_name = raw_name.rpartition("::")
            if separator:
                owner = qualified_owner.strip(":")
                if namespace_owner and not owner.startswith(f"{namespace_owner}::"):
                    owner = f"{namespace_owner}::{owner}"
                name = f"{owner}::{short_name}"
            elif lexical_owner:
                owner = lexical_owner
                name = f"{owner}::{raw_name}"
            else:
                owner = None
                name = (
                    f"{namespace_owner}::{raw_name}"
                    if namespace_owner
                    else raw_name
                )
            parameters_node = function_declarator.child_by_field_name("parameters")
            parameters: list[str] = []
            parameter_types: list[str] = []
            if parameters_node is not None:
                for param in parameters_node.named_children:
                    if param.type not in {
                        "parameter_declaration",
                        "optional_parameter_declaration",
                    }:
                        continue
                    declarator_node = param.child_by_field_name("declarator")
                    identifier = _last_identifier(declarator_node)
                    parameter_name = unit.text(identifier).strip()
                    if parameter_name:
                        parameters.append(parameter_name)
                    type_node = param.child_by_field_name("type")
                    type_text = unit.text(type_node).strip()
                    if declarator_node is not None:
                        declarator_text = unit.text(declarator_node)
                        suffix = declarator_text[: max(0, declarator_text.rfind(parameter_name))]
                        type_text = " ".join(f"{type_text} {suffix}".split())
                    parameter_types.append(type_text)
            return_type_node = node.child_by_field_name("type")
            return_type = unit.text(return_type_node).strip() or None
            line = unit.line(node)
            callable_id = ":".join(
                [unit.file, str(line), name, ",".join(parameters)]
            )
            evidence = " ".join(
                unit.source[node.start_byte : body.start_byte + 1]
                .decode("utf-8", errors="replace")
                .split()
            )
            unit.callables.append(
                _Callable(
                    name=name,
                    owner=owner,
                    file=unit.file,
                    line=line,
                    end_line=int(node.end_point.row) + 1,
                    callable_id=callable_id,
                    parameters=parameters,
                    parameter_types=parameter_types,
                    return_type=return_type,
                    node=node,
                    body=body,
                    evidence=evidence,
                )
            )
        if include_lambdas:
            self._extract_lambdas(unit, definitions)

    def _extract_lambdas(
        self, unit: _ParsedUnit, parent_definitions: Sequence[Node]
    ) -> None:
        bodies = [
            body
            for definition in parent_definitions
            if (body := definition.child_by_field_name("body")) is not None
        ]
        lambda_nodes = [
            node
            for body in bodies
            for node in _walk(body)
            if node.type == "lambda_expression"
        ]
        # Register outer lambdas before nested lambdas so the inner callable
        # retains its immediate lexical parent.
        lambda_nodes.sort(key=lambda item: (item.start_byte, -item.end_byte))
        for node in lambda_nodes:
            parent = min(
                (
                    item
                    for item in unit.callables
                    if item.body.start_byte <= node.start_byte
                    and node.end_byte <= item.body.end_byte
                    and item.node != node
                ),
                key=lambda item: item.body.end_byte - item.body.start_byte,
                default=None,
            )
            if parent is None:
                continue
            initializer = node.parent
            while initializer is not None and initializer.type not in {
                "init_declarator",
                "assignment_expression",
            }:
                initializer = initializer.parent
            if initializer is None:
                continue
            target_node = (
                initializer.child_by_field_name("declarator")
                if initializer.type == "init_declarator"
                else initializer.child_by_field_name("left")
            )
            variable = self._canonical_symbol(unit.text(_last_identifier(target_node)))
            if not variable:
                continue
            declarator = node.child_by_field_name("declarator")
            parameters_node = (
                declarator.child_by_field_name("parameters")
                if declarator is not None
                else None
            )
            parameters, parameter_types = self._parameter_list(
                unit, parameters_node
            )
            body = node.child_by_field_name("body")
            if body is None:
                continue
            captures, capture_arguments, capture_exact = self._lambda_captures(
                unit,
                node,
                parent,
                body,
                parameters,
            )
            trailing = next(
                (
                    child
                    for child in _walk(declarator)
                    if child.type == "trailing_return_type"
                ),
                None,
            ) if declarator is not None else None
            return_type = unit.text(
                trailing.child_by_field_name("type") if trailing is not None else None
            ).strip() or None
            full_name = f"{parent.name}::{variable}"
            all_parameters = [*captures, *parameters]
            callable_id = ":".join(
                [
                    unit.file,
                    str(unit.line(node)),
                    full_name,
                    ",".join(all_parameters),
                ]
            )
            unit.callables.append(
                _Callable(
                    name=full_name,
                    owner=parent.owner,
                    file=unit.file,
                    line=unit.line(node),
                    end_line=int(node.end_point.row) + 1,
                    callable_id=callable_id,
                    parameters=all_parameters,
                    parameter_types=[*("capture" for _ in captures), *parameter_types],
                    return_type=return_type,
                    node=node,
                    body=body,
                    evidence=unit.evidence(node),
                    parent_callable_id=parent.callable_id,
                    captures=captures,
                    capture_arguments=capture_arguments,
                    is_lambda=True,
                    capture_exact=capture_exact,
                )
            )

    def _lambda_captures(
        self,
        unit: _ParsedUnit,
        node: Node,
        parent: _Callable,
        body: Node,
        parameters: Sequence[str],
    ) -> tuple[list[str], list[str], bool]:
        """Derive lambda formals and caller expressions from lexical syntax."""
        captures_node = node.child_by_field_name("captures")
        captures: list[str] = []
        arguments: list[str] = []
        exact = True
        has_default = False

        for capture_node in captures_node.named_children if captures_node else ():
            if capture_node.type == "lambda_default_capture":
                has_default = True
                continue
            if capture_node.type == "lambda_capture_initializer":
                left = capture_node.child_by_field_name("left")
                right = capture_node.child_by_field_name("right")
                name = unit.text(_last_identifier(left)).strip()
                expression = self.profiler._normalize_source_expression(
                    unit.text(right)
                )
                if name and expression:
                    captures.append(name)
                    arguments.append(expression)
                else:
                    exact = False
                continue

            capture = unit.text(capture_node).strip().lstrip("&").strip()
            if capture in {"this", "*this"}:
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", capture):
                captures.append(capture)
                arguments.append(capture)
            else:
                exact = False

        if has_default:
            explicit = set(captures)
            for capture in self._implicit_lambda_captures(
                unit, parent, node, body, parameters
            ):
                if capture in explicit:
                    continue
                captures.append(capture)
                arguments.append(capture)
                explicit.add(capture)

        return captures, arguments, exact

    def _implicit_lambda_captures(
        self,
        unit: _ParsedUnit,
        parent: _Callable,
        lambda_node: Node,
        body: Node,
        parameters: Sequence[str],
    ) -> list[str]:
        """Find referenced parent locals whose scope contains the lambda."""
        visible = dict.fromkeys(parent.parameters)
        for declaration in _walk_operations(parent.body):
            if declaration.type != "declaration":
                continue
            if declaration.start_byte >= lambda_node.start_byte:
                continue
            if declaration.start_byte <= lambda_node.start_byte < declaration.end_byte:
                continue
            scope = self._declaration_scope(declaration, parent.body)
            if scope is None or not (
                scope.start_byte <= lambda_node.start_byte
                and lambda_node.end_byte <= scope.end_byte
            ):
                continue
            for declarator in _declaration_declarators(declaration):
                name = unit.text(_last_identifier(declarator)).strip()
                if name:
                    visible.setdefault(name, None)

        local_declarations: dict[str, list[tuple[int, Node]]] = {}
        for declaration in _walk_operations(body):
            if declaration.type != "declaration":
                continue
            scope = self._declaration_scope(declaration, body)
            if scope is None:
                continue
            for declarator in _declaration_declarators(declaration):
                name = unit.text(_last_identifier(declarator)).strip()
                if name:
                    local_declarations.setdefault(name, []).append(
                        (declarator.start_byte, scope)
                    )

        captured: list[str] = []
        seen: set[str] = set()
        parameter_names = set(parameters)
        for reference in _walk_operations(body):
            if reference.type != "identifier":
                continue
            name = unit.text(reference).strip()
            if name not in visible or name in parameter_names or name in seen:
                continue
            if any(
                declaration_start <= reference.start_byte
                and scope.start_byte <= reference.start_byte
                and reference.end_byte <= scope.end_byte
                for declaration_start, scope in local_declarations.get(name, ())
            ):
                continue
            seen.add(name)
            captured.append(name)
        return captured

    @staticmethod
    def _declaration_scope(declaration: Node, callable_body: Node) -> Optional[Node]:
        scope = declaration.parent
        scope_types = {
            "compound_statement",
            "for_statement",
            "for_range_loop",
            "while_statement",
            "if_statement",
            "switch_statement",
        }
        while scope is not None and scope != callable_body:
            if scope.type in scope_types:
                return scope
            scope = scope.parent
        return callable_body if scope == callable_body else None

    def _parameter_list(
        self, unit: _ParsedUnit, parameters_node: Optional[Node]
    ) -> tuple[list[str], list[str]]:
        parameters: list[str] = []
        parameter_types: list[str] = []
        if parameters_node is None:
            return parameters, parameter_types
        for param in parameters_node.named_children:
            if param.type not in {
                "parameter_declaration",
                "optional_parameter_declaration",
            }:
                continue
            declarator_node = param.child_by_field_name("declarator")
            identifier = _last_identifier(declarator_node)
            parameter_name = unit.text(identifier).strip()
            if parameter_name:
                parameters.append(parameter_name)
            type_node = param.child_by_field_name("type")
            type_text = unit.text(type_node).strip()
            if declarator_node is not None:
                declarator_text = unit.text(declarator_node)
                suffix = declarator_text[
                    : max(0, declarator_text.rfind(parameter_name))
                ]
                type_text = " ".join(f"{type_text} {suffix}".split())
            parameter_types.append(type_text)
        return parameters, parameter_types

    def _index_parameter_members(self, context: _SourceContext) -> None:
        for unit in context.units:
            for parameter, member, owner, _template in self._parameter_declarations(unit):
                if parameter and member and owner:
                    context.parameter_members[(owner, member)] = parameter

    @staticmethod
    def _parameter_name_from_text(text: str) -> Optional[str]:
        match = re.search(r"\bpx4::params::(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b", text)
        return match.group("name") if match else None

    def _parameter_declarations(
        self, unit: _ParsedUnit
    ) -> list[tuple[str, Optional[str], Optional[str], Node]]:
        templates = [
            node
            for node in _walk(unit.tree.root_node)
            if node.type in {"template_type", "template_method"}
            and self._parameter_name_from_text(unit.text(node))
        ]
        refs: list[tuple[str, Optional[str], Optional[str], Node]] = []
        for template in templates:
            parameter = self._parameter_name_from_text(unit.text(template))
            if not parameter:
                continue
            declaration = template.parent
            while declaration is not None and declaration.type not in {
                "declaration",
                "field_declaration",
            }:
                declaration = declaration.parent
            if declaration is None:
                continue
            next_template_start = min(
                (
                    candidate.start_byte
                    for candidate in templates
                    if template.end_byte <= candidate.start_byte < declaration.end_byte
                ),
                default=declaration.end_byte,
            )
            identifiers = [
                node
                for node in _walk(declaration)
                if node.type in {"identifier", "field_identifier"}
                and template.end_byte <= node.start_byte < next_template_start
            ]
            member = unit.text(min(identifiers, key=lambda item: item.start_byte)).strip() if identifiers else None
            refs.append((parameter, member, unit.class_owner(template), template))
        return refs

    def _extract_callable(
        self, unit: _ParsedUnit, context: _SourceContext, callable: _Callable
    ) -> _ExtractionState:
        struct_variables = self._struct_variables(unit, context, callable)
        state = _ExtractionState(
            unit=unit,
            context=context,
            callable=callable,
            struct_variables=struct_variables,
            lambda_bindings={
                item.name.rsplit("::", 1)[-1]: item
                for item in unit.callables
                if item.parent_callable_id == callable.callable_id
            },
        )
        self._walk_sequence(
            state,
            list(callable.body.named_children),
            controls=[],
            exact=not callable.body.has_error,
        )
        return state

    def _struct_variables(
        self, unit: _ParsedUnit, context: _SourceContext, callable: _Callable
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for owner in context.lineage(callable.owner):
            for (declaring_owner, name), member in context.members.items():
                if declaring_owner != owner or not member.type:
                    continue
                struct = self._struct_type(member.type)
                if struct:
                    mapping[name] = struct
        for name, type_text in zip(callable.parameters, callable.parameter_types):
            struct = self._struct_type(type_text)
            if struct:
                mapping[name] = struct
        for node in _walk_operations(callable.body):
            if node.type != "declaration":
                continue
            type_text = unit.text(node.child_by_field_name("type"))
            struct = self._struct_type(type_text)
            if not struct:
                continue
            for declarator in _declaration_declarators(node):
                name_node = _last_identifier(declarator)
                name = unit.text(name_node).strip()
                if name:
                    mapping[name] = struct
        return mapping

    @staticmethod
    def _struct_type(type_text: str) -> Optional[str]:
        match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*_s)\b", type_text or "")
        return match.group(1) if match else None

    def _walk_sequence(
        self,
        state: _ExtractionState,
        statements: Sequence[Node],
        *,
        controls: list[_ControlTerm],
        exact: bool,
        exit_context: Optional[str] = None,
    ) -> None:
        active_controls = list(controls)
        active_exact = exact
        for statement in statements:
            self._walk_statement(
                state,
                statement,
                controls=active_controls,
                exact=active_exact,
                exit_context=exit_context,
            )
            if exit_context == "switch":
                termination, termination_exact = self._switch_exit_for_statement(
                    state.unit, statement
                )
            else:
                termination, termination_exact = self._termination_expression(
                    state.unit, statement
                )
            if termination == "true":
                break
            if termination != "false":
                active_controls = [
                    *active_controls,
                    _ControlTerm(
                        expression=_not(termination),
                        line=state.unit.line(statement),
                        site_id=state.unit.site_id(statement),
                    ),
                ]
                active_exact = active_exact and termination_exact

    def _walk_statement(
        self,
        state: _ExtractionState,
        node: Node,
        *,
        controls: list[_ControlTerm],
        exact: bool,
        exit_context: Optional[str] = None,
    ) -> None:
        unit = state.unit
        exact = exact and not node.has_error
        if node.type == "compound_statement":
            saved_aliases = state.storage_aliases
            saved_capable = state.alias_capable_symbols
            saved_references = state.reference_alias_symbols
            state.storage_aliases = dict(saved_aliases)
            state.alias_capable_symbols = set(saved_capable)
            state.reference_alias_symbols = set(saved_references)
            try:
                self._walk_sequence(
                    state,
                    node.named_children,
                    controls=controls,
                    exact=exact,
                    exit_context=exit_context,
                )
            finally:
                state.storage_aliases = saved_aliases
                state.alias_capable_symbols = saved_capable
                state.reference_alias_symbols = saved_references
            return
        if node.type == "if_statement":
            condition_node = node.child_by_field_name("condition")
            condition = self._condition_text(unit, condition_node)
            self._collect_operations(
                state, condition_node, controls=controls, exact=exact
            )
            term = _ControlTerm(
                expression=condition,
                line=unit.line(node),
                site_id=unit.site_id(node),
            )
            self._record_branch(state, "if", term, node)
            consequence = node.child_by_field_name("consequence")
            if consequence is not None:
                self._walk_scoped_statement(
                    state,
                    consequence,
                    controls=[*controls, term],
                    exact=exact,
                    exit_context=exit_context,
                )
            alternative = node.child_by_field_name("alternative")
            if alternative is not None:
                alternative_body = (
                    alternative.named_children[0]
                    if alternative.type == "else_clause" and alternative.named_children
                    else alternative
                )
                self._walk_scoped_statement(
                    state,
                    alternative_body,
                    controls=[
                        *controls,
                        _ControlTerm(
                            expression=_not(condition),
                            line=unit.line(node),
                            site_id=unit.site_id(node),
                        ),
                    ],
                    exact=exact,
                    exit_context=exit_context,
                )
            return
        if node.type == "switch_statement":
            self._walk_switch(state, node, controls=controls, exact=exact)
            return
        if node.type in {"while_statement", "for_statement", "do_statement"}:
            condition_node = node.child_by_field_name("condition")
            condition = self._condition_text(unit, condition_node) or "true"
            self._collect_operations(
                state, condition_node, controls=controls, exact=False
            )
            term = _ControlTerm(
                expression=condition,
                line=unit.line(node),
                site_id=unit.site_id(node),
            )
            self._record_branch(state, node.type.removesuffix("_statement"), term, node)
            initializer = node.child_by_field_name("initializer")
            update = node.child_by_field_name("update")
            saved_aliases = state.storage_aliases
            saved_capable = state.alias_capable_symbols
            saved_references = state.reference_alias_symbols
            state.storage_aliases = dict(saved_aliases)
            state.alias_capable_symbols = set(saved_capable)
            state.reference_alias_symbols = set(saved_references)
            self._collect_operations(state, initializer, controls=controls, exact=False)
            if initializer is not None and initializer.type == "declaration":
                self._register_storage_aliases(state, initializer)
            body = node.child_by_field_name("body")
            try:
                if body is not None:
                    body_controls = controls if node.type == "do_statement" else [*controls, term]
                    self._walk_statement(
                        state,
                        body,
                        controls=body_controls,
                        exact=False,
                        # A transfer inside this loop belongs to the loop rather
                        # than to a switch that happens to contain the loop.
                        exit_context=None,
                    )
                self._collect_operations(
                    state, update, controls=[*controls, term], exact=False
                )
            finally:
                state.storage_aliases = saved_aliases
                state.alias_capable_symbols = saved_capable
                state.reference_alias_symbols = saved_references
            return
        if node.type == "return_statement":
            self._collect_operations(state, node, controls=controls, exact=exact)
            expression = unit.text(node.named_children[0]) if node.named_children else ""
            if expression and controls:
                condition = " && ".join(
                    f"({item.expression})" for item in controls
                )
                state.return_paths.append(
                    {
                        "condition": self._apply_storage_aliases(
                            self.profiler._normalize_source_expression(condition),
                            state.storage_aliases,
                        ),
                        "expression": self._apply_storage_aliases(
                            self.profiler._normalize_source_expression(expression),
                            state.storage_aliases,
                        ),
                        "file": unit.file,
                        "line": unit.line(node),
                        "source_site_id": unit.site_id(node),
                        "reachability_exact": exact,
                    }
                )
            return
        if node.type.startswith("preproc_"):
            self._collect_operations(state, node, controls=controls, exact=False)
            return
        self._collect_operations(state, node, controls=controls, exact=exact)
        if node.type == "declaration":
            self._register_storage_aliases(state, node)
        elif node.type == "expression_statement" and node.named_children:
            self._update_storage_alias(state, node.named_children[0])

    def _walk_scoped_statement(
        self,
        state: _ExtractionState,
        node: Node,
        *,
        controls: list[_ControlTerm],
        exact: bool,
        exit_context: Optional[str],
    ) -> None:
        saved_aliases = state.storage_aliases
        saved_capable = state.alias_capable_symbols
        saved_references = state.reference_alias_symbols
        state.storage_aliases = dict(saved_aliases)
        state.alias_capable_symbols = set(saved_capable)
        state.reference_alias_symbols = set(saved_references)
        try:
            self._walk_statement(
                state,
                node,
                controls=controls,
                exact=exact,
                exit_context=exit_context,
            )
        finally:
            state.storage_aliases = saved_aliases
            state.alias_capable_symbols = saved_capable
            state.reference_alias_symbols = saved_references

    def _walk_switch(
        self,
        state: _ExtractionState,
        node: Node,
        *,
        controls: list[_ControlTerm],
        exact: bool,
    ) -> None:
        unit = state.unit
        condition_node = node.child_by_field_name("condition")
        discriminant = self._condition_text(unit, condition_node)
        self._collect_operations(state, condition_node, controls=controls, exact=exact)
        body = node.child_by_field_name("body")
        sections = self._switch_sections(unit, body)
        labels = [label for label, _site, _body in sections if label is not None]
        no_match = " && ".join(
            f"!({discriminant} == {label})" for label in labels
        ) or "true"
        fallthrough = "false"
        switch_aliases = state.storage_aliases
        switch_capable = state.alias_capable_symbols
        switch_references = state.reference_alias_symbols
        fallthrough_alias_ambiguous = False
        for label, site, statements in sections:
            entry = no_match if label is None else f"{discriminant} == {label}"
            active = _or(entry, fallthrough)
            term = _ControlTerm(
                expression=active,
                line=unit.line(site),
                site_id=unit.site_id(site),
            )
            kind = "default" if label is None else "case"
            self._record_branch(state, kind, term, site)
            # A single alias environment cannot represent the different
            # storage bindings produced by mutually exclusive case entries.
            # Start every section from the pre-switch environment; this is
            # conservative for fallthrough aliases and prevents one case's
            # assignment from being applied to an unrelated case.
            state.storage_aliases = dict(switch_aliases)
            state.alias_capable_symbols = set(switch_capable)
            state.reference_alias_symbols = set(switch_references)
            self._walk_sequence(
                state,
                statements,
                controls=[*controls, term],
                exact=exact and not fallthrough_alias_ambiguous,
                exit_context="switch",
            )
            exit_expression, exit_exact = self._switch_exit_expression(
                unit, statements
            )
            if (
                exit_expression != "true"
                and state.storage_aliases != switch_aliases
            ):
                # The next section has a direct-entry environment and a
                # different fallthrough environment. One scalar alias map
                # cannot represent both, so later writes remain present but
                # explicitly non-exact until path-sensitive alias states are
                # modeled.
                fallthrough_alias_ambiguous = True
            fallthrough = _and(active, _not(exit_expression))
            exact = exact and exit_exact
        state.storage_aliases = switch_aliases
        state.alias_capable_symbols = switch_capable
        state.reference_alias_symbols = switch_references

    def _switch_sections(
        self, unit: _ParsedUnit, body: Optional[Node]
    ) -> list[tuple[Optional[str], Node, list[Node]]]:
        if body is None:
            return []
        sections: list[tuple[Optional[str], Node, list[Node]]] = []
        pending: list[tuple[Optional[str], Node]] = []
        for child in body.named_children:
            if child.type != "case_statement":
                continue
            value = child.child_by_field_name("value")
            label = unit.text(value).strip() if value is not None else None
            statements = [item for item in child.named_children if item != value]
            pending.append((label, child))
            if not statements:
                continue
            # Consecutive labels share the same body. Each label is an entry;
            # the walker combines them through the fallthrough expression.
            for pending_label, pending_site in pending[:-1]:
                sections.append((pending_label, pending_site, []))
            sections.append((pending[-1][0], pending[-1][1], statements))
            pending = []
        sections.extend((label, site, []) for label, site in pending)
        return sections

    def _record_branch(
        self,
        state: _ExtractionState,
        kind: str,
        term: _ControlTerm,
        node: Node,
    ) -> None:
        state.branches.append(
            BranchConditionRef(
                kind=kind,
                condition=self._apply_storage_aliases(
                    self.profiler._normalize_source_expression(term.expression),
                    state.storage_aliases,
                ),
                file=state.unit.file,
                line=term.line,
                evidence=state.unit.evidence(node),
                source_site_id=term.site_id,
            )
        )

    def _condition_text(self, unit: _ParsedUnit, node: Optional[Node]) -> str:
        if node is None:
            return ""
        value = node.child_by_field_name("value")
        if value is None and node.type in {"condition_clause", "parenthesized_expression"}:
            value = node.named_children[0] if node.named_children else None
        return self.profiler._normalize_source_expression(
            unit.text(value if value is not None else node)
        )

    def _termination_expression(
        self, unit: _ParsedUnit, node: Node
    ) -> tuple[str, bool]:
        if node.type in {"return_statement", "co_return_statement", "throw_statement"}:
            return "true", not node.has_error
        if node.type == "compound_statement":
            result = "false"
            exact = not node.has_error
            for child in node.named_children:
                child_result, child_exact = self._termination_expression(unit, child)
                result = _or(result, child_result)
                exact = exact and child_exact
                if result == "true":
                    break
            return result, exact
        if node.type == "if_statement":
            condition = self._condition_text(unit, node.child_by_field_name("condition"))
            consequence = node.child_by_field_name("consequence")
            then_result, then_exact = (
                self._termination_expression(unit, consequence)
                if consequence is not None
                else ("false", True)
            )
            alternative = node.child_by_field_name("alternative")
            if alternative is not None and alternative.type == "else_clause" and alternative.named_children:
                alternative = alternative.named_children[0]
            else_result, else_exact = (
                self._termination_expression(unit, alternative)
                if alternative is not None
                else ("false", True)
            )
            return (
                _or(_and(condition, then_result), _and(_not(condition), else_result)),
                then_exact and else_exact and not node.has_error,
            )
        if node.type == "switch_statement":
            discriminant = self._condition_text(
                unit, node.child_by_field_name("condition")
            )
            sections = self._switch_sections(unit, node.child_by_field_name("body"))
            labels = [label for label, _site, _body in sections if label is not None]
            no_match = " && ".join(
                f"!({discriminant} == {label})" for label in labels
            ) or "true"
            result = "false"
            fallthrough = "false"
            exact = not node.has_error
            for label, _site, statements in sections:
                entry = no_match if label is None else f"{discriminant} == {label}"
                active = _or(entry, fallthrough)
                section_result = "false"
                for statement in statements:
                    child_result, child_exact = self._termination_expression(unit, statement)
                    section_result = _or(section_result, child_result)
                    exact = exact and child_exact
                result = _or(result, _and(active, section_result))
                exit_result, exit_exact = self._switch_exit_expression(unit, statements)
                fallthrough = _and(active, _not(exit_result))
                exact = exact and exit_exact
            return result, exact
        if node.type in {"while_statement", "for_statement", "do_statement"}:
            body = node.child_by_field_name("body")
            body_result, _ = (
                self._termination_expression(unit, body)
                if body is not None
                else ("false", False)
            )
            condition = self._condition_text(unit, node.child_by_field_name("condition")) or "true"
            return _and(condition, body_result), False
        return "false", not node.has_error

    def _switch_exit_expression(
        self, unit: _ParsedUnit, statements: Sequence[Node]
    ) -> tuple[str, bool]:
        result = "false"
        exact = True
        for statement in statements:
            current, current_exact = self._switch_exit_for_statement(unit, statement)
            result = _or(result, current)
            exact = exact and current_exact
            if result == "true":
                break
        return result, exact

    def _switch_exit_for_statement(
        self, unit: _ParsedUnit, node: Node
    ) -> tuple[str, bool]:
        if node.type in {
            "break_statement",
            "continue_statement",
            "return_statement",
            "co_return_statement",
            "throw_statement",
        }:
            return "true", not node.has_error
        if node.type == "compound_statement":
            return self._switch_exit_expression(unit, node.named_children)
        if node.type == "if_statement":
            condition = self._condition_text(unit, node.child_by_field_name("condition"))
            consequence = node.child_by_field_name("consequence")
            then_result, then_exact = (
                self._switch_exit_for_statement(unit, consequence)
                if consequence is not None
                else ("false", True)
            )
            alternative = node.child_by_field_name("alternative")
            if alternative is not None and alternative.type == "else_clause" and alternative.named_children:
                alternative = alternative.named_children[0]
            else_result, else_exact = (
                self._switch_exit_for_statement(unit, alternative)
                if alternative is not None
                else ("false", True)
            )
            return (
                _or(_and(condition, then_result), _and(_not(condition), else_result)),
                then_exact and else_exact and not node.has_error,
            )
        if node.type in {"switch_statement", "for_statement", "while_statement", "do_statement"}:
            return "false", False
        if node.type == "goto_statement":
            return "false", False
        return "false", not node.has_error

    def _collect_operations(
        self,
        state: _ExtractionState,
        node: Optional[Node],
        *,
        controls: list[_ControlTerm],
        exact: bool,
    ) -> None:
        if node is None:
            return
        for candidate in _walk_operations(node):
            site = (candidate.start_byte, candidate.end_byte)
            if candidate.type == "assignment_expression" and site not in state.seen_assignments:
                state.seen_assignments.add(site)
                assignment = self._assignment_ref(
                    state, candidate, controls=controls, exact=exact
                )
                if assignment is not None:
                    state.assignments.append(assignment)
            elif candidate.type == "init_declarator" and site not in state.seen_assignments:
                state.seen_assignments.add(site)
                assignment = self._initializer_ref(
                    state, candidate, controls=controls, exact=exact
                )
                if assignment is not None:
                    state.assignments.append(assignment)
            elif candidate.type == "call_expression" and site not in state.seen_calls:
                state.seen_calls.add(site)
                call = self._call_ref(
                    state, candidate, controls=controls, exact=exact
                )
                if call is not None:
                    state.calls.append(call)

    def _assignment_ref(
        self,
        state: _ExtractionState,
        node: Node,
        *,
        controls: list[_ControlTerm],
        exact: bool,
    ) -> Optional[SourceAssignmentRef]:
        unit = state.unit
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is None or right is None:
            return None
        target = self._storage_target(state, unit.text(left))
        if not target:
            return None
        operator = unit.source[left.end_byte : right.start_byte].decode(
            "utf-8", errors="replace"
        ).strip()
        if operator not in {"=", "+=", "-=", "*=", "/=", "%=", "|=", "&=", "^="}:
            return None
        rhs = self._apply_storage_aliases(
            self.profiler._normalize_source_expression(unit.text(right)),
            state.storage_aliases,
        )
        expression = (
            rhs
            if operator == "="
            else self.profiler._compound_assignment_expression(target, operator, rhs)
        )
        return self._make_assignment(
            state,
            node,
            target=target,
            expression=expression,
            operator=operator,
            declaration_kind=None,
            controls=controls,
            exact=exact,
        )

    def _initializer_ref(
        self,
        state: _ExtractionState,
        node: Node,
        *,
        controls: list[_ControlTerm],
        exact: bool,
    ) -> Optional[SourceAssignmentRef]:
        unit = state.unit
        declarator = node.child_by_field_name("declarator")
        value = node.child_by_field_name("value")
        name_node = _last_identifier(declarator)
        target = unit.text(name_node).strip()
        if not target or value is None:
            return None
        if any(child.type == "lambda_expression" for child in _walk(value)):
            return None
        expression = self._apply_storage_aliases(
            self.profiler._normalize_source_expression(
                _strip_initializer_delimiters(unit.text(value))
            ),
            state.storage_aliases,
        )
        declaration = node.parent
        declaration_text = unit.text(declaration) if declaration is not None else unit.text(node)
        kind = "constexpr" if re.search(r"\bconstexpr\b", declaration_text) else None
        return self._make_assignment(
            state,
            node,
            target=target,
            expression=expression,
            operator="=",
            declaration_kind=kind,
            controls=controls,
            exact=exact,
        )

    def _make_assignment(
        self,
        state: _ExtractionState,
        node: Node,
        *,
        target: str,
        expression: str,
        operator: str,
        declaration_kind: Optional[str],
        controls: list[_ControlTerm],
        exact: bool,
    ) -> SourceAssignmentRef:
        root, field_name = split_source_field(target)
        struct = state.struct_variables.get(root)
        topic = self.profiler._topic_from_struct(struct) if struct else None
        predicates = [
            self._apply_storage_aliases(item.expression, state.storage_aliases)
            for item in controls
        ]
        combined = " ".join([target, expression, *predicates])
        return SourceAssignmentRef(
            target=target,
            expression=expression,
            file=state.unit.file,
            line=state.unit.line(node),
            evidence=state.unit.evidence(node),
            function=state.callable.name,
            callable_id=state.callable.callable_id,
            function_parameters=state.callable.parameters,
            target_topic=topic,
            target_field=field_name if topic else None,
            assignment_operator=operator,
            declaration_kind=declaration_kind,
            control_predicates=predicates,
            control_predicate_lines=[item.line for item in controls],
            control_predicate_site_ids=[item.site_id for item in controls],
            reachability_exact=exact and not node.has_error,
            symbol_bindings=self.profiler._source_symbol_bindings(
                combined, state.struct_variables
            ),
            struct_variables=state.struct_variables,
            source_site_id=state.unit.site_id(node),
        )

    def _call_ref(
        self,
        state: _ExtractionState,
        node: Node,
        *,
        controls: list[_ControlTerm],
        exact: bool,
    ) -> Optional[FunctionCallRef]:
        unit = state.unit
        function_node = node.child_by_field_name("function")
        arguments_node = node.child_by_field_name("arguments")
        if function_node is None:
            return None
        receiver: Optional[str] = None
        name = unit.text(function_node).strip()
        if function_node.type == "field_expression":
            receiver_node = function_node.child_by_field_name("argument")
            field_node = function_node.child_by_field_name("field")
            receiver = self._storage_target(state, unit.text(receiver_node))
            name = unit.text(field_node).strip()
        args = [
            self._apply_storage_aliases(unit.text(arg).strip(), state.storage_aliases)
            for arg in (arguments_node.named_children if arguments_node else [])
        ]
        lambda_callable = state.lambda_bindings.get(name) if receiver is None else None
        if lambda_callable is not None:
            args = [*lambda_callable.capture_arguments, *args]
        argument_topics: dict[str, str] = {}
        argument_owners: dict[str, str] = {}
        for arg in args:
            symbol = self.profiler._boundary_argument_symbol(arg)
            if not symbol:
                continue
            root = symbol.split(".", 1)[0]
            struct = state.struct_variables.get(root)
            topic = self.profiler._topic_from_struct(struct) if struct else None
            if topic:
                argument_topics[symbol] = topic
            member = state.context.declaring_member(state.callable.owner, root)
            if member is not None:
                argument_owners[root] = member.owner
        predicates = [
            self._apply_storage_aliases(item.expression, state.storage_aliases)
            for item in controls
        ]
        return FunctionCallRef(
            name=name,
            receiver=receiver,
            args=args,
            argument_topics=argument_topics,
            control_predicates=predicates,
            control_predicate_lines=[item.line for item in controls],
            control_predicate_site_ids=[item.site_id for item in controls],
            reachability_exact=exact and not node.has_error,
            symbol_bindings=self.profiler._source_symbol_bindings(
                " ".join([receiver or "", *args, *predicates]),
                state.struct_variables,
            ),
            argument_owners=argument_owners,
            function=state.callable.name,
            callable_id=state.callable.callable_id,
            file=unit.file,
            line=unit.line(node),
            evidence=unit.evidence(node),
            source_site_id=unit.site_id(node),
        )

    def _extract_global_constants(self, unit: _ParsedUnit) -> list[SourceAssignmentRef]:
        refs: list[SourceAssignmentRef] = []
        for node in _walk(unit.tree.root_node):
            if node.type == "enumerator":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                if name_node is None or value_node is None:
                    continue
                refs.append(
                    SourceAssignmentRef(
                        target=unit.text(name_node).strip(),
                        expression=self.profiler._normalize_source_expression(
                            unit.text(value_node)
                        ),
                        file=unit.file,
                        line=unit.line(node),
                        evidence=unit.evidence(node),
                        assignment_operator="=",
                        declaration_kind="enum",
                        source_site_id=unit.site_id(node),
                    )
                )
            elif node.type == "preproc_def":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                name = unit.text(name_node).strip()
                value = unit.text(value_node).strip()
                if not name or not value:
                    continue
                refs.append(
                    SourceAssignmentRef(
                        target=name,
                        expression=self.profiler._normalize_source_expression(value),
                        file=unit.file,
                        line=unit.line(node),
                        evidence=unit.evidence(node),
                        assignment_operator="=",
                        declaration_kind="define",
                        source_site_id=unit.site_id(node),
                    )
                )
        return refs

    def _extract_topics(
        self,
        unit: _ParsedUnit,
        context: _SourceContext,
        states: Sequence[_ExtractionState],
    ) -> dict[str, list[TopicRef]]:
        refs: dict[str, list[TopicRef]] = {
            "publish": [],
            "subscribe": [],
            "unknown": [],
        }
        state_by_callable = {state.callable.callable_id: state for state in states}
        consumed_orb_sites: set[tuple[int, int]] = set()

        for node in _walk(unit.tree.root_node):
            if node.type not in {"declaration", "field_declaration"}:
                continue
            type_node = node.child_by_field_name("type")
            type_text = unit.text(type_node)
            direction = self._uorb_direction(type_text)
            if not direction:
                continue
            callable = self._enclosing_callable(unit, node)
            state = state_by_callable.get(callable.callable_id) if callable else None
            class_owner = unit.class_owner(node)
            declarators = _declaration_declarators(node)
            topic_root = node.child_by_field_name("default_value")
            for declarator in declarators:
                name_node = _last_identifier(declarator)
                variable = unit.text(name_node).strip()
                if not variable:
                    continue
                topics = self._orb_topics(
                    unit, topic_root if topic_root is not None else node
                )
                for index, (topic, instance, orb_node) in enumerate(topics):
                    consumed_orb_sites.add((orb_node.start_byte, orb_node.end_byte))
                    scoped_variable = (
                        f"{variable}[{index}]" if len(topics) > 1 else variable
                    )
                    owner, endpoint_kind = self._endpoint_scope(
                        context,
                        callable,
                        scoped_variable,
                        declared_owner=class_owner,
                    )
                    refs[direction].append(
                        TopicRef(
                            topic=topic,
                            direction=direction,
                            file=unit.file,
                            line=unit.line(declarator),
                            evidence=unit.evidence(declarator),
                            struct=self._struct_type(type_text)
                            or self.profiler._struct_from_topic(topic),
                            variable=scoped_variable,
                            api=self._uorb_api(type_text),
                            instance=instance,
                            function=callable.name if callable else None,
                            callable_id=callable.callable_id if callable else None,
                            variable_owner=owner,
                            endpoint_kind=endpoint_kind,
                            source_site_id=unit.site_id(declarator),
                        )
                    )

        for node in _walk(unit.tree.root_node):
            if node.type != "assignment_expression":
                continue
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is None or right is None:
                continue
            type_node = right.child_by_field_name("type")
            type_text = unit.text(type_node)
            direction = self._uorb_direction(type_text)
            if not direction:
                continue
            callable = self._enclosing_callable(unit, node)
            variable = self._canonical_symbol(unit.text(left))
            topics = self._orb_topics(unit, right)
            for index, (topic, instance, orb_node) in enumerate(topics):
                consumed_orb_sites.add((orb_node.start_byte, orb_node.end_byte))
                scoped_variable = (
                    f"{variable}[{index}]" if len(topics) > 1 else variable
                )
                owner, endpoint_kind = self._endpoint_scope(
                    context, callable, scoped_variable
                )
                refs[direction].append(
                    TopicRef(
                        topic=topic,
                        direction=direction,
                        file=unit.file,
                        line=unit.line(node),
                        evidence=unit.evidence(node),
                        struct=self._struct_type(type_text)
                        or self.profiler._struct_from_topic(topic),
                        variable=scoped_variable,
                        api=self._uorb_api(type_text),
                        instance=instance,
                        function=callable.name if callable else None,
                        callable_id=callable.callable_id if callable else None,
                        variable_owner=owner,
                        endpoint_kind=endpoint_kind,
                        source_site_id=unit.site_id(node),
                    )
                )

        for node in _walk(unit.tree.root_node):
            if node.type != "field_initializer":
                continue
            callable = self._enclosing_callable(unit, node)
            if callable is None or not node.named_children:
                continue
            initializer_name = unit.text(node.named_children[0]).strip()
            root_name = initializer_name.rsplit("::", 1)[-1]
            member = context.declaring_member(callable.owner, root_name)
            type_text = initializer_name if self._uorb_direction(initializer_name) else str(member.type if member else "")
            direction = self._uorb_direction(type_text)
            if not direction:
                continue
            topics = self._orb_topics(unit, node)
            base_endpoint = member is None and self._uorb_direction(initializer_name)
            variable = "this" if base_endpoint else root_name
            for index, (topic, instance, orb_node) in enumerate(topics):
                consumed_orb_sites.add((orb_node.start_byte, orb_node.end_byte))
                scoped_variable = (
                    f"{variable}[{index}]" if len(topics) > 1 and not base_endpoint else variable
                )
                refs[direction].append(
                    TopicRef(
                        topic=topic,
                        direction=direction,
                        file=unit.file,
                        line=unit.line(node),
                        evidence=unit.evidence(node),
                        struct=self._struct_type(type_text)
                        or self.profiler._struct_from_topic(topic),
                        variable=scoped_variable,
                        api=self._uorb_api(type_text),
                        instance=instance,
                        function=callable.name,
                        callable_id=callable.callable_id,
                        variable_owner=callable.owner,
                        endpoint_kind="base" if base_endpoint else "member",
                        source_site_id=unit.site_id(node),
                    )
                )

        for node in _walk(unit.tree.root_node):
            if node.type != "call_expression":
                continue
            function_node = node.child_by_field_name("function")
            function_name = unit.text(function_node).strip()
            short_name = function_name.rsplit("::", 1)[-1]
            if not short_name.startswith("orb_"):
                continue
            direction = (
                "subscribe"
                if short_name.startswith(("orb_copy", "orb_subscribe"))
                else "publish"
                if short_name.startswith(("orb_publish", "orb_advertise"))
                else "unknown"
            )
            topics = self._orb_topics(unit, node)
            if not topics:
                continue
            arguments = node.child_by_field_name("arguments")
            args = list(arguments.named_children) if arguments is not None else []
            callable = self._enclosing_callable(unit, node)
            boundary_symbol: Optional[str] = None
            if short_name.startswith(("orb_copy", "orb_publish", "orb_advertise")) and args:
                boundary_symbol = self.profiler._boundary_argument_symbol(unit.text(args[-1]))
            elif short_name.startswith("orb_subscribe"):
                boundary_symbol = self._enclosing_assignment_target(unit, node)
            for topic, instance, orb_node in topics:
                consumed_orb_sites.add((orb_node.start_byte, orb_node.end_byte))
                owner, endpoint_kind = self._endpoint_scope(
                    context, callable, boundary_symbol or ""
                )
                refs[direction].append(
                    TopicRef(
                        topic=topic,
                        direction=direction,
                        file=unit.file,
                        line=unit.line(node),
                        evidence=unit.evidence(node),
                        struct=self.profiler._struct_from_topic(topic),
                        variable=boundary_symbol,
                        api=short_name,
                        instance=instance,
                        function=callable.name if callable else None,
                        callable_id=callable.callable_id if callable else None,
                        variable_owner=owner,
                        endpoint_kind=endpoint_kind,
                        source_site_id=unit.site_id(node),
                    )
                )

        for topic, instance, orb_node in self._orb_topics(unit, unit.tree.root_node):
            if (orb_node.start_byte, orb_node.end_byte) in consumed_orb_sites:
                continue
            callable = self._enclosing_callable(unit, orb_node)
            refs["unknown"].append(
                TopicRef(
                    topic=topic,
                    direction="unknown",
                    file=unit.file,
                    line=unit.line(orb_node),
                    evidence=unit.evidence(orb_node),
                    struct=self.profiler._struct_from_topic(topic),
                    api="ORB_ID",
                    instance=instance,
                    function=callable.name if callable else None,
                    callable_id=callable.callable_id if callable else None,
                    source_site_id=unit.site_id(orb_node),
                )
            )
        return refs

    def _orb_topics(
        self, unit: _ParsedUnit, root: Node
    ) -> list[tuple[str, Optional[int], Node]]:
        topics: list[tuple[str, Optional[int], Node]] = []
        for node in _walk(root):
            if node.type == "call_expression":
                function_node = node.child_by_field_name("function")
                function_name = unit.text(function_node).strip()
                if function_name.rsplit("::", 1)[-1] != "ORB_ID":
                    continue
                arguments = node.child_by_field_name("arguments")
                args = list(arguments.named_children) if arguments is not None else []
                if not args:
                    continue
                topic = unit.text(args[0]).strip()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", topic):
                    topics.append((topic, self._orb_instance(unit, node), node))
            elif node.type == "qualified_identifier":
                text = unit.text(node).strip()
                match = re.fullmatch(r"ORB_ID::([A-Za-z_][A-Za-z0-9_]*)", text)
                if match:
                    topics.append((match.group(1), self._orb_instance(unit, node), node))
        return sorted(topics, key=lambda item: item[2].start_byte)

    @staticmethod
    def _orb_instance(unit: _ParsedUnit, node: Node) -> Optional[int]:
        parent = node.parent
        while parent is not None and parent.type not in {
            "argument_list",
            "initializer_list",
            "field_initializer",
        }:
            parent = parent.parent
        if parent is None:
            return None
        siblings = list(parent.named_children)
        containing = next(
            (
                index
                for index, sibling in enumerate(siblings)
                if sibling.start_byte <= node.start_byte and node.end_byte <= sibling.end_byte
            ),
            None,
        )
        if containing is None or containing + 1 >= len(siblings):
            return None
        candidate = unit.text(siblings[containing + 1]).strip()
        return int(candidate) if candidate.isdigit() else None

    @staticmethod
    def _uorb_direction(type_text: str) -> Optional[str]:
        match = re.search(
            r"(?:^|::)(?P<kind>Publication|Subscription)[A-Za-z0-9_]*\b",
            type_text or "",
        )
        if not match:
            return None
        return "publish" if match.group("kind") == "Publication" else "subscribe"

    @staticmethod
    def _uorb_api(type_text: str) -> Optional[str]:
        match = re.search(
            r"(?:^|::)((?:Publication|Subscription)[A-Za-z0-9_]*)\b",
            type_text or "",
        )
        return f"uORB::{match.group(1)}" if match else None

    def _endpoint_scope(
        self,
        context: _SourceContext,
        callable: Optional[_Callable],
        variable: str,
        *,
        declared_owner: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        root = variable.split("[", 1)[0].replace("->", ".").split(".", 1)[0]
        if declared_owner:
            return declared_owner, "member"
        if callable is None:
            return None, "global"
        member = context.declaring_member(callable.owner, root)
        if member is not None:
            return member.owner, "member"
        return None, "local"

    @staticmethod
    def _enclosing_callable(unit: _ParsedUnit, node: Node) -> Optional[_Callable]:
        candidates = [
            item
            for item in unit.callables
            if item.node.start_byte <= node.start_byte and node.end_byte <= item.node.end_byte
        ]
        return min(candidates, key=lambda item: item.node.end_byte - item.node.start_byte) if candidates else None

    def _enclosing_assignment_target(
        self, unit: _ParsedUnit, node: Node
    ) -> Optional[str]:
        parent = node.parent
        while parent is not None and parent.type not in {
            "assignment_expression",
            "init_declarator",
            "expression_statement",
        }:
            parent = parent.parent
        if parent is None:
            return None
        if parent.type == "assignment_expression":
            return self.profiler._clean_field_path(
                unit.text(parent.child_by_field_name("left"))
            )
        if parent.type == "init_declarator":
            return unit.text(_last_identifier(parent.child_by_field_name("declarator"))).strip() or None
        return None

    def _extract_parameters(
        self, unit: _ParsedUnit, context: _SourceContext
    ) -> list[ParameterRef]:
        refs: list[ParameterRef] = []
        covered: set[tuple[int, int]] = set()
        for parameter, member, _owner, template in self._parameter_declarations(unit):
            refs.append(
                ParameterRef(
                    name=parameter,
                    member=member,
                    file=unit.file,
                    line=unit.line(template),
                    evidence=unit.evidence(template),
                    access_pattern="typed_parameter_declaration",
                )
            )
            covered.add((template.start_byte, template.end_byte))

        for node in _walk(unit.tree.root_node):
            if node.type == "qualified_identifier":
                name = self._parameter_name_from_text(unit.text(node))
                if not name or any(
                    start <= node.start_byte and node.end_byte <= end
                    for start, end in covered
                ):
                    continue
                refs.append(
                    ParameterRef(
                        name=name,
                        file=unit.file,
                        line=unit.line(node),
                        evidence=unit.evidence(node),
                        access_pattern="px4_parameter_identifier",
                    )
                )
            elif node.type == "call_expression":
                function_node = node.child_by_field_name("function")
                function_text = unit.text(function_node).strip()
                args_node = node.child_by_field_name("arguments")
                args = list(args_node.named_children) if args_node is not None else []
                if function_text.rsplit("::", 1)[-1] == "param_find" and args:
                    literal = unit.text(args[0]).strip()
                    if len(literal) >= 2 and literal[0] == literal[-1] == '"':
                        refs.append(
                            ParameterRef(
                                name=literal[1:-1],
                                file=unit.file,
                                line=unit.line(node),
                                evidence=unit.evidence(node),
                                access_pattern="param_find",
                            )
                        )
                if function_node is None or function_node.type != "field_expression":
                    continue
                field_node = function_node.child_by_field_name("field")
                if unit.text(field_node).strip() != "get":
                    continue
                receiver_node = function_node.child_by_field_name("argument")
                receiver = self._canonical_symbol(unit.text(receiver_node))
                callable = self._enclosing_callable(unit, node)
                parameter = context.parameter_for_member(
                    callable.owner if callable else unit.class_owner(node),
                    receiver.split(".", 1)[0],
                )
                refs.append(
                    ParameterRef(
                        name=parameter,
                        member=receiver,
                        file=unit.file,
                        line=unit.line(node),
                        evidence=unit.evidence(node),
                        access_pattern="typed_parameter_get",
                        confidence="high" if parameter else "low",
                    )
                )
        return refs

    def _parameter_predicates(
        self,
        unit: _ParsedUnit,
        context: _SourceContext,
        parameters: Sequence[ParameterRef],
    ) -> list[ParameterPredicateRef]:
        refs: list[ParameterPredicateRef] = []
        direct_members = {
            item.member: item.name
            for item in parameters
            if item.member and item.name
        }
        for node in _walk(unit.tree.root_node):
            if node.type != "binary_expression":
                continue
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is None or right is None:
                continue
            operator = unit.source[left.end_byte : right.start_byte].decode(
                "utf-8", errors="replace"
            ).strip()
            if operator not in {"==", "!=", "<", ">", "<=", ">="}:
                continue
            left_name, left_member = self._parameter_operand(
                unit, context, node, left, direct_members
            )
            right_name, right_member = self._parameter_operand(
                unit, context, node, right, direct_members
            )
            if not left_name and not right_name:
                continue
            if left_name:
                name, member, compared = left_name, left_member, unit.text(right).strip()
            else:
                name, member, compared = right_name, right_member, unit.text(left).strip()
            refs.append(
                ParameterPredicateRef(
                    name=name,
                    member=member,
                    predicate=self.profiler._normalize_source_expression(unit.text(node)),
                    operator=operator,
                    compared_value=compared,
                    file=unit.file,
                    line=unit.line(node),
                    evidence=unit.evidence(node),
                )
            )
        return refs

    def _parameter_operand(
        self,
        unit: _ParsedUnit,
        context: _SourceContext,
        comparison: Node,
        operand: Node,
        direct_members: dict[str, str],
    ) -> tuple[Optional[str], Optional[str]]:
        text = unit.text(operand).strip()
        direct = self._parameter_name_from_text(text)
        if direct:
            return direct, None
        callable = self._enclosing_callable(unit, comparison)
        owner = callable.owner if callable else unit.class_owner(comparison)
        candidates: set[tuple[str, Optional[str]]] = set()
        for node in _walk(operand):
            if node.type == "qualified_identifier":
                name = self._parameter_name_from_text(unit.text(node))
                if name:
                    candidates.add((name, None))
            if node.type != "call_expression":
                continue
            function = node.child_by_field_name("function")
            if function is None or function.type != "field_expression":
                continue
            if unit.text(function.child_by_field_name("field")).strip() != "get":
                continue
            receiver = self._canonical_symbol(
                unit.text(function.child_by_field_name("argument"))
            )
            name = direct_members.get(receiver) or context.parameter_for_member(
                owner, receiver.split(".", 1)[0]
            )
            if name:
                candidates.add((name, receiver))
        return next(iter(candidates)) if len(candidates) == 1 else (None, None)

    def _field_refs(
        self,
        unit: _ParsedUnit,
        context: _SourceContext,
        states: Sequence[_ExtractionState],
    ) -> tuple[list[FieldRef], list[FieldRef]]:
        assigned: list[FieldRef] = []
        read: list[FieldRef] = []
        state_by_callable = {state.callable.callable_id: state for state in states}
        assigned_ranges: list[tuple[int, int]] = []
        for state in states:
            for assignment in state.assignments:
                if "." not in assignment.target:
                    continue
                root, field_name = split_source_field(assignment.target)
                struct = state.struct_variables.get(root)
                assigned.append(
                    FieldRef(
                        field=field_name,
                        variable=root,
                        topic=self.profiler._topic_from_struct(struct) if struct else None,
                        struct=struct,
                        assignment_operator=assignment.assignment_operator,
                        file=assignment.file,
                        line=assignment.line,
                        evidence=assignment.evidence,
                    )
                )
            for node in _walk_operations(state.callable.body):
                if node.type == "assignment_expression":
                    left = node.child_by_field_name("left")
                    if left is not None:
                        assigned_ranges.append((left.start_byte, left.end_byte))

        for node in _walk(unit.tree.root_node):
            if node.type != "field_expression":
                continue
            if node.parent is not None and node.parent.type == "field_expression":
                continue
            if any(start <= node.start_byte and node.end_byte <= end for start, end in assigned_ranges):
                continue
            symbol = self._canonical_symbol(unit.text(node))
            root, field_name = split_source_field(symbol)
            if not root or not field_name:
                continue
            callable = self._enclosing_callable(unit, node)
            state = state_by_callable.get(callable.callable_id) if callable else None
            struct = state.struct_variables.get(root) if state else None
            read.append(
                FieldRef(
                    field=field_name,
                    variable=root,
                    topic=self.profiler._topic_from_struct(struct) if struct else None,
                    struct=struct,
                    file=unit.file,
                    line=unit.line(node),
                    evidence=unit.evidence(node),
                )
            )
        return assigned, read

    def _register_storage_aliases(
        self, state: _ExtractionState, declaration: Node
    ) -> None:
        """Record pointer/reference locals as aliases of source storage.

        The environment is lexical: callers copy it when entering a block,
        branch, or loop. Only syntax-proven pointer/reference declarators are
        eligible, and every initializer is resolved through aliases already in
        scope before the new name is installed.
        """
        unit = state.unit
        for declarator in (
            node
            for node in declaration.named_children
            if node.type == "init_declarator"
        ):
            declared = declarator.child_by_field_name("declarator")
            value = declarator.child_by_field_name("value")
            if declared is None:
                continue
            is_alias_capable = any(
                node.type in {"pointer_declarator", "reference_declarator"}
                for node in _walk(declared)
            )
            if not is_alias_capable:
                continue
            name_node = _last_identifier(declared)
            name = unit.text(name_node).strip()
            if not name:
                continue
            state.alias_capable_symbols.add(name)
            if any(
                node.type == "reference_declarator" for node in _walk(declared)
            ):
                state.reference_alias_symbols.add(name)
            replacement = self._storage_alias_value(state, value)
            if replacement:
                state.storage_aliases[name] = replacement
            else:
                state.storage_aliases.pop(name, None)

    def _update_storage_alias(self, state: _ExtractionState, node: Node) -> None:
        """Update or invalidate a previously declared pointer/reference alias."""
        if node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is None or right is None:
                return
            target = self._canonical_symbol(state.unit.text(left))
            if "." in target or target not in state.alias_capable_symbols:
                return
            if target in state.reference_alias_symbols:
                # C++ references cannot be rebound. ``ref = value`` writes
                # through the alias and leaves its storage identity intact.
                return
            replacement = self._storage_alias_value(state, right)
            if replacement:
                state.storage_aliases[target] = replacement
            else:
                state.storage_aliases.pop(target, None)
        elif node.type == "update_expression":
            argument = node.child_by_field_name("argument")
            target = self._canonical_symbol(state.unit.text(argument))
            if target in state.alias_capable_symbols:
                state.storage_aliases.pop(target, None)

    def _storage_alias_value(
        self, state: _ExtractionState, value: Optional[Node]
    ) -> Optional[str]:
        if value is None:
            return None
        while value.type == "parenthesized_expression" and value.named_children:
            value = value.named_children[0]
        if value.type == "new_expression":
            return None
        if value.type == "pointer_expression":
            text = state.unit.text(value).lstrip()
            if not text.startswith("&") or not value.named_children:
                return None
            value = value.named_children[0]
        if value.type not in {
            "identifier",
            "field_expression",
            "subscript_expression",
            "call_expression",
            "qualified_identifier",
        }:
            return None
        expression = self._canonical_symbol(state.unit.text(value))
        expression = self._apply_storage_aliases(
            expression, state.storage_aliases
        )
        return expression or None

    def _storage_target(self, state: _ExtractionState, text: str) -> str:
        target = self._canonical_symbol(text)
        if target in state.reference_alias_symbols:
            return state.storage_aliases.get(target, target)
        if "." not in target:
            return target
        return self._apply_storage_aliases(target, state.storage_aliases)

    @staticmethod
    def _apply_storage_aliases(expression: str, aliases: dict[str, str]) -> str:
        value = str(expression or "")
        for name, replacement in sorted(
            aliases.items(), key=lambda item: len(item[0]), reverse=True
        ):
            value = re.sub(
                rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_])",
                replacement,
                value,
            )
        return value

    @staticmethod
    def _canonical_symbol(text: str) -> str:
        value = MechanismSourceProfiler._clean_field_path(text)
        if value.startswith("this."):
            value = value[5:]
        return value

    def _helper_from_state(self, state: _ExtractionState) -> HelperExpressionRef:
        statements = self._helper_statements(state.unit, state.callable.body)
        unresolved = self._helper_unsupported_reason(state.callable.body)
        if state.callable.is_lambda and not state.callable.capture_exact:
            unresolved = "lambda capture bindings are not structurally resolvable"
        lowered: Optional[str] = None
        if unresolved is None:
            try:
                lowered = self.profiler._lower_helper_statements(statements)
            except _HelperLoweringFailed as exc:
                unresolved = exc.reason
        returns = [
            node
            for node in _walk_operations(state.callable.body)
            if node.type in {"return_statement", "co_return_statement"}
            and node.named_children
        ]
        return_expression = (
            self.profiler._normalize_source_expression(
                state.unit.text(returns[0].named_children[0])
            )
            if len(returns) == 1
            else None
        )
        # Keep helper metadata aligned with the ordinary assignment facts.
        # The latter already carry lexical storage-alias resolution and exact
        # source scopes; rebuilding this map from the raw helper IR would
        # reintroduce pre-alias targets such as ``local_ptr.alt``.
        assignments = {
            assignment.target: assignment.expression
            for assignment in state.assignments
        }
        output_alias_params = {
            name
            for name, type_text in zip(
                state.callable.parameters, state.callable.parameter_types
            )
            if "*" in type_text or "&" in type_text
        }
        pointer_writes: list[dict[str, str]] = []
        for assignment in state.assignments:
            root, field_name = split_source_field(assignment.target)
            # The declaration proves aliasing; an actual assignment through
            # that alias proves output behavior. This covers pointers and
            # non-const references without guessing from helper names.
            if root in output_alias_params:
                pointer_writes.append(
                    {
                        "param": root,
                        "field": field_name or "",
                        "expression": assignment.expression,
                    }
                )
        if unresolved is None and not return_expression and not lowered and not state.return_paths and not pointer_writes:
            unresolved = "helper has no return value and no pointer-output writes routable through source assignments"
        helper_calls = list(dict.fromkeys(call.name for call in state.calls))
        call_resolutions = self._helper_call_resolutions(state)
        symbol_bindings = self.profiler._source_symbol_bindings(
            state.unit.text(state.callable.body), state.struct_variables
        )
        for call in state.calls:
            if call.receiver and call.name == "get":
                parameter = state.context.parameter_for_member(
                    state.callable.owner, call.receiver.split(".", 1)[0]
                )
                if parameter:
                    symbol_bindings[f"{call.receiver}.get()"] = parameter
        return HelperExpressionRef(
            name=state.callable.name,
            owner=state.callable.owner,
            file=state.unit.file,
            line=state.callable.line,
            evidence=state.callable.evidence,
            callable_id=state.callable.callable_id,
            parameters=state.callable.parameters,
            statements=statements if unresolved is None else [],
            assignments=assignments if unresolved is None else {},
            return_expression=return_expression if unresolved is None else None,
            lowered_return_expression=lowered if unresolved is None else None,
            branches=state.return_paths if unresolved is None else [],
            symbol_bindings=symbol_bindings if unresolved is None else {},
            call_resolutions=call_resolutions,
            helper_calls=helper_calls,
            pointer_output_writes=pointer_writes if unresolved is None else [],
            return_type=state.callable.return_type,
            struct_variables=state.struct_variables if unresolved is None else {},
            unresolved_reason=unresolved,
        )

    def _helper_statements(self, unit: _ParsedUnit, node: Node) -> list[dict[str, Any]]:
        statements: list[dict[str, Any]] = []
        children = node.named_children if node.type == "compound_statement" else [node]
        for child in children:
            converted = self._helper_statement(unit, child)
            if isinstance(converted, list):
                statements.extend(converted)
            elif converted is not None:
                statements.append(converted)
        return statements

    def _helper_statement(
        self, unit: _ParsedUnit, node: Node
    ) -> Optional[dict[str, Any] | list[dict[str, Any]]]:
        if node.type == "compound_statement":
            return self._helper_statements(unit, node)
        if node.type == "return_statement" and node.named_children:
            return {
                "kind": "return",
                "expression": self.profiler._normalize_source_expression(
                    unit.text(node.named_children[0])
                ),
            }
        if node.type == "if_statement":
            consequence = node.child_by_field_name("consequence")
            alternative = node.child_by_field_name("alternative")
            if alternative is not None and alternative.type == "else_clause" and alternative.named_children:
                alternative = alternative.named_children[0]
            return {
                "kind": "if",
                "condition": self._condition_text(
                    unit, node.child_by_field_name("condition")
                ),
                "then": self._helper_statements(unit, consequence) if consequence is not None else [],
                "else": self._helper_statements(unit, alternative) if alternative is not None else [],
            }
        if node.type == "switch_statement":
            discriminant = self._condition_text(
                unit, node.child_by_field_name("condition")
            )
            cases: list[dict[str, Any]] = []
            default: Optional[list[dict[str, Any]]] = None
            pending: list[str] = []
            for label, _site, body in self._switch_sections(
                unit, node.child_by_field_name("body")
            ):
                if label is None:
                    default = [
                        item
                        for statement in body
                        for item in self._as_statement_list(
                            self._helper_statement(unit, statement)
                        )
                    ]
                    continue
                pending.append(label)
                if body:
                    cases.append(
                        {
                            "conditions": list(pending),
                            "body": [
                                item
                                for statement in body
                                if statement.type != "break_statement"
                                for item in self._as_statement_list(
                                    self._helper_statement(unit, statement)
                                )
                            ],
                        }
                    )
                    pending = []
            return {
                "kind": "switch",
                "discriminant": discriminant,
                "cases": cases,
                "default": default,
            }
        if node.type == "for_statement":
            initializer = node.child_by_field_name("initializer")
            update = node.child_by_field_name("update")
            body = node.child_by_field_name("body")
            init_statements = self._as_statement_list(
                self._helper_statement(unit, initializer) if initializer is not None else None
            )
            update_statement = self._helper_loop_increment(unit, update)
            return {
                "kind": "for",
                "init": init_statements[0] if init_statements else None,
                "init_text": unit.text(initializer).rstrip(";") if initializer is not None else "",
                "condition": self._condition_text(unit, node.child_by_field_name("condition")),
                "increment": update_statement,
                "increment_text": unit.text(update) if update is not None else "",
                "body": self._helper_statements(unit, body) if body is not None else [],
            }
        if node.type == "for_range_loop":
            return {"kind": "for_range", "header": unit.evidence(node), "body": []}
        if node.type in {"while_statement", "do_statement"}:
            body = node.child_by_field_name("body")
            return {
                "kind": "while" if node.type == "while_statement" else "do_while",
                "condition": self._condition_text(unit, node.child_by_field_name("condition")),
                "body": self._helper_statements(unit, body) if body is not None else [],
            }
        if node.type == "declaration":
            out: list[dict[str, Any]] = []
            for child in node.named_children:
                if child.type != "init_declarator":
                    continue
                declarator = child.child_by_field_name("declarator")
                value = child.child_by_field_name("value")
                if value is not None and any(
                    nested.type == "lambda_expression" for nested in _walk(value)
                ):
                    continue
                name = unit.text(_last_identifier(declarator)).strip()
                if name and value is not None:
                    out.append(
                        {
                            "kind": "declare",
                            "target": name,
                            "expression": self.profiler._normalize_source_expression(
                                _strip_initializer_delimiters(unit.text(value))
                            ),
                        }
                    )
            return out
        if node.type == "expression_statement" and node.named_children:
            return self._helper_simple_expression(unit, node.named_children[0])
        if node.type in {"assignment_expression", "update_expression"}:
            return self._helper_simple_expression(unit, node)
        return None

    def _helper_loop_increment(
        self, unit: _ParsedUnit, node: Optional[Node]
    ) -> Optional[dict[str, Any]]:
        """Adapt a C++ for-update to the shared helper-lowering step IR.

        Ordinary assignment IR stores the resulting value expression. The
        loop lowerer instead expects the signed step expression, so reusing
        ``_helper_simple_expression`` turns ``i++`` into a non-constant
        ``i + 1`` step and prevents otherwise literal loops from unrolling.
        """
        if node is None:
            return None
        if node.type == "update_expression":
            argument = node.child_by_field_name("argument")
            target = self._canonical_symbol(unit.text(argument))
            text = unit.text(node)
            return {
                "target": target,
                "operator": "+=" if "++" in text else "-=",
                "expression": "1",
            }
        if node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is None or right is None:
                return None
            operator = unit.source[left.end_byte : right.start_byte].decode(
                "utf-8", errors="replace"
            ).strip()
            if operator not in {"+=", "-="}:
                return None
            return {
                "target": self._canonical_symbol(unit.text(left)),
                "operator": operator,
                "expression": self.profiler._normalize_source_expression(
                    unit.text(right)
                ),
            }
        return None

    def _helper_simple_expression(
        self, unit: _ParsedUnit, node: Optional[Node]
    ) -> Optional[dict[str, Any]]:
        if node is None:
            return None
        if node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is None or right is None:
                return None
            target = self._canonical_symbol(unit.text(left))
            operator = unit.source[left.end_byte : right.start_byte].decode(
                "utf-8", errors="replace"
            ).strip()
            rhs = self.profiler._normalize_source_expression(unit.text(right))
            expression = rhs if operator == "=" else self.profiler._compound_assignment_expression(target, operator, rhs)
            return {
                "kind": "assign",
                "target": target,
                "operator": operator,
                "expression": expression,
            }
        if node.type == "update_expression":
            argument = node.child_by_field_name("argument")
            target = self._canonical_symbol(unit.text(argument))
            text = unit.text(node)
            operator = "+=" if "++" in text else "-="
            return {
                "kind": "assign",
                "target": target,
                "operator": operator,
                "expression": self.profiler._compound_assignment_expression(
                    target, operator, "1"
                ),
            }
        return None

    @staticmethod
    def _as_statement_list(
        value: Optional[dict[str, Any] | list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    @staticmethod
    def _helper_unsupported_reason(body: Node) -> Optional[str]:
        unsupported = next(
            (
                node.type
                for node in _walk_operations(body)
                if node.type
                in {
                    "goto_statement",
                    "try_statement",
                    "co_yield_statement",
                }
            ),
            None,
        )
        return f"helper body uses unsupported {unsupported}" if unsupported else None

    def _helper_call_resolutions(
        self, state: _ExtractionState
    ) -> list[dict[str, Any]]:
        resolutions: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for call in state.calls:
            call_name = f"{call.receiver}.{call.name}" if call.receiver else call.name
            if call.receiver and call.name == "get":
                parameter = state.context.parameter_for_member(
                    state.callable.owner, call.receiver.split(".", 1)[0]
                )
                resolution = (
                    {
                        "call": f"{call_name}()",
                        "kind": "parameter_accessor",
                        "parameter": parameter,
                    }
                    if parameter
                    else {
                        "call": f"{call_name}()",
                        "kind": "unresolved_runtime_call",
                    }
                )
            elif call.receiver:
                resolution = {
                    "call": f"{call_name}()",
                    "kind": "unresolved_runtime_call",
                }
            elif is_safe_math_function_name(call.name):
                resolution = {
                    "call": f"{call.name}()",
                    "kind": "math_function",
                    "canonical_name": canonical_math_function_name(call.name),
                }
            else:
                resolution = {
                    "call": f"{call.name}()",
                    "kind": "source_helper_candidate",
                }
            key = (str(resolution["call"]), str(resolution["kind"]))
            if key not in seen:
                seen.add(key)
                resolutions.append(resolution)
        return resolutions

    @staticmethod
    def _dedupe(items: Iterable[Any]) -> list[Any]:
        seen: set[str] = set()
        out: list[Any] = []
        for item in items:
            payload = item.model_dump_json(exclude_none=True) if hasattr(item, "model_dump_json") else repr(item)
            if payload in seen:
                continue
            seen.add(payload)
            out.append(item)
        return out

    def _diagnostics(self, unit: _ParsedUnit) -> dict[str, Any]:
        errors = [
            {
                "type": node.type,
                "line": unit.line(node),
                "site_id": unit.site_id(node),
                "evidence": unit.evidence(node),
            }
            for node in _walk(unit.tree.root_node)
            if node.type == "ERROR" or node.is_missing
        ]
        return {
            "has_error": unit.tree.root_node.has_error,
            "error_nodes": errors,
        }
