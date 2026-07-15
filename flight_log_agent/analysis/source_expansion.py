"""Source-proven identities and candidate resolution for DAG expansion.

Search is only candidate generation. A file contributes facts to discovery
only after this module verifies that it declares the exact class, callable,
constant, or scoped source symbol requested by a DAG frontier record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel

from flight_log_agent.analysis.source_expression import source_expression_names
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler
from flight_log_agent.px4.source_facts_cache import SourceFileFacts, extract_facts_for_file
from flight_log_agent.symbols import exact_symbol, symbol_produces_reference
from flight_log_agent.utils import dedupe_keep_order


SymbolKind = Literal["local", "member", "global", "unknown"]
GapKind = Literal["symbol", "callable", "constant", "class"]


class SourceSymbolIdentity(BaseModel):
    """Lossless source identity used for producer admission and wiring."""

    kind: SymbolKind
    symbol: str
    root: str
    file: str = ""
    callable_id: str = ""
    class_owner: str = ""
    declaring_class: str = ""

    def key(self) -> tuple[str, ...]:
        return (
            self.kind,
            self.symbol,
            self.file,
            self.callable_id,
            self.class_owner,
            self.declaring_class,
        )

    def storage_key(self) -> tuple[str, ...]:
        """Identity of the storage location, independent of use site.

        A local belongs to one callable, while a member remains the same
        storage when different methods of its declaring class write it.
        Globals are source-unit scoped. Unknown identities retain every
        available scope component rather than being merged by spelling.
        """
        if self.kind == "local":
            return (self.kind, self.symbol, self.file, self.callable_id)
        if self.kind == "member":
            return (
                self.kind,
                self.symbol,
                self.declaring_class or self.class_owner,
            )
        if self.kind == "global":
            return (self.kind, self.symbol, self.file)
        return (
            self.kind,
            self.symbol,
            self.file,
            self.callable_id,
            self.class_owner,
            self.declaring_class,
        )


class UnresolvedSourceReference(BaseModel):
    """A typed DAG frontier item with its originating source context."""

    symbol: str
    kind: GapKind = "symbol"
    file: str = ""
    line: Optional[int] = None
    callable_id: str = ""
    class_owner: str = ""
    receiver: str = ""
    argument_count: Optional[int] = None
    source_expression: str = ""
    identity: Optional[SourceSymbolIdentity] = None

    def visit_key(self) -> tuple[Any, ...]:
        return (
            self.kind,
            exact_symbol(self.symbol),
            self.file,
            self.callable_id,
            self.class_owner,
            self.receiver,
            self.argument_count,
            self.identity.key() if self.identity is not None else None,
        )


@dataclass
class SourceStructureIndex:
    """In-memory structural index derived from the currently loaded facts."""

    direct_bases: dict[str, set[str]] = field(default_factory=dict)
    members: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    callables_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    callables_by_name: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    includes: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_facts(cls, facts: Iterable[Any]) -> "SourceStructureIndex":
        index = cls()
        raw_bases: dict[str, set[str]] = {}
        for raw in facts:
            entry = raw.model_dump(exclude_none=True) if hasattr(raw, "model_dump") else dict(raw)
            for class_ref in entry.get("classes") or []:
                item = _as_dict(class_ref)
                name = str(item.get("name") or "")
                if name:
                    raw_bases.setdefault(name, set()).update(
                        str(base) for base in item.get("bases") or [] if base
                    )
            for member_ref in entry.get("members") or []:
                item = _as_dict(member_ref)
                owner = str(item.get("owner") or "")
                name = str(item.get("name") or "")
                if owner and name:
                    index.members.setdefault((owner, name), item)
            for callable_ref in entry.get("callables") or []:
                item = _as_dict(callable_ref)
                callable_id = str(item.get("callable_id") or "")
                name = str(item.get("name") or "")
                if callable_id:
                    index.callables_by_id.setdefault(callable_id, item)
                if name:
                    for key in {name, name.rsplit("::", 1)[-1]}:
                        bucket = index.callables_by_name.setdefault(key, [])
                        if not any(
                            str(existing.get("callable_id") or "") == callable_id
                            for existing in bucket
                        ):
                            bucket.append(item)
            for include_ref in entry.get("includes") or []:
                item = _as_dict(include_ref)
                source = str(item.get("file") or "")
                included = str(item.get("included_file") or "")
                if source and included:
                    index.includes.setdefault(source, set()).add(included)
        declared = set(raw_bases)
        for owner, bases in raw_bases.items():
            owner_parts = owner.split("::")[:-1]
            for raw_base in bases:
                base = re.sub(r"^virtual\s+", "", raw_base.strip().lstrip(":"))
                base_name = base.split("<", 1)[0].strip()
                candidates = [
                    "::".join([*owner_parts[:depth], base_name])
                    for depth in range(len(owner_parts), -1, -1)
                    if base_name
                ]
                resolved = next(
                    (candidate for candidate in candidates if candidate in declared),
                    base,
                )
                if resolved:
                    index.direct_bases.setdefault(owner, set()).add(resolved)
        return index

    def lineage(self, class_name: str) -> list[str]:
        """Return ``class_name`` followed by all derivable base classes."""
        ordered: list[str] = []
        frontier = [class_name] if class_name else []
        seen: set[str] = set()
        while frontier:
            current = frontier.pop(0)
            if not current or current in seen:
                continue
            seen.add(current)
            ordered.append(current)
            frontier.extend(sorted(self.direct_bases.get(current, ())))
        return ordered

    def declaring_member_owner(self, class_name: str, member: str) -> Optional[str]:
        matches = [
            owner
            for owner in self.lineage(class_name)
            if (owner, member) in self.members
        ]
        return matches[0] if matches else None

    def callable_owner(self, callable_id: str, function_name: str = "") -> str:
        record = self.callables_by_id.get(callable_id)
        if record and record.get("owner"):
            return str(record["owner"])
        owner, separator, _name = str(function_name or "").rpartition("::")
        if separator:
            return owner
        matches = re.findall(
            r"(?<![A-Za-z0-9_])"
            r"([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)",
            str(callable_id or ""),
        )
        owner, separator, _name = matches[-1].rpartition("::") if matches else ("", "", "")
        return owner if separator else ""

    def member_receiver_type(self, class_name: str, receiver: str) -> str:
        root = receiver.replace("->", ".").split(".", 1)[0].lstrip("&*")
        declaring = self.declaring_member_owner(class_name, root)
        member = self.members.get((declaring, root)) if declaring else None
        receiver_type = str((member or {}).get("type") or "")
        return receiver_type.rstrip("*& ").split("<", 1)[0].strip()

    def symbol_identity(
        self,
        symbol: str,
        *,
        file: str,
        callable_id: str,
        function_name: str = "",
        function_parameters: Sequence[str] = (),
    ) -> SourceSymbolIdentity:
        canonical = exact_symbol(symbol)
        normalized = canonical.replace("->", ".")
        parts = [part for part in normalized.split(".") if part]
        root = parts[0].lstrip("&*") if parts else canonical
        explicit_this = root == "this" and len(parts) > 1
        if explicit_this:
            root = parts[1]
        owner = self.callable_owner(callable_id, function_name)
        if root in {str(value) for value in function_parameters}:
            return SourceSymbolIdentity(
                kind="local",
                symbol=canonical,
                root=root,
                file=file,
                callable_id=callable_id,
                class_owner=owner,
            )
        declaring_owner = self.declaring_member_owner(owner, root) if owner else None
        if declaring_owner or (explicit_this and owner):
            return SourceSymbolIdentity(
                kind="member",
                symbol=canonical,
                root=root,
                file=file,
                callable_id=callable_id,
                class_owner=owner,
                declaring_class=declaring_owner or owner,
            )
        if callable_id:
            return SourceSymbolIdentity(
                kind="local",
                symbol=canonical,
                root=root,
                file=file,
                callable_id=callable_id,
                class_owner=owner,
            )
        if canonical:
            return SourceSymbolIdentity(
                kind="global",
                symbol=canonical,
                root=root,
                file=file,
            )
        return SourceSymbolIdentity(kind="unknown", symbol=canonical, root=root, file=file)

    def compatible(
        self,
        reference: SourceSymbolIdentity,
        producer: SourceSymbolIdentity,
    ) -> bool:
        if not symbol_produces_reference(producer.symbol, reference.symbol):
            return False
        if reference.kind == "local" or producer.kind == "local":
            return (
                reference.kind == producer.kind == "local"
                and bool(reference.callable_id)
                and reference.callable_id == producer.callable_id
            )
        if reference.kind == "member" or producer.kind == "member":
            owner_compatible = (
                not reference.class_owner
                or producer.class_owner
                in set(self.lineage(reference.class_owner))
            )
            return (
                reference.kind == producer.kind == "member"
                and bool(reference.declaring_class)
                and reference.declaring_class == producer.declaring_class
                and owner_compatible
            )
        if reference.kind == producer.kind == "global":
            return True
        return reference.file == producer.file and bool(reference.file)

    @staticmethod
    def storage_compatible(
        reference: SourceSymbolIdentity,
        producer: SourceSymbolIdentity,
    ) -> bool:
        """Whether ``producer`` writes the requested storage location."""
        if not symbol_produces_reference(producer.symbol, reference.symbol):
            return False
        if reference.kind != producer.kind:
            return False
        if reference.kind == "local":
            return (
                bool(reference.file)
                and reference.file == producer.file
                and bool(reference.callable_id)
                and reference.callable_id == producer.callable_id
            )
        if reference.kind == "member":
            return (
                bool(reference.declaring_class)
                and reference.declaring_class == producer.declaring_class
            )
        if reference.kind == "global":
            return bool(reference.file) and reference.file == producer.file
        return reference.storage_key() == producer.storage_key()

    def enrich_bindings(self, bindings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for raw in bindings:
            binding = dict(raw)
            path = binding.get("assignment_path") or []
            site = path[0] if path else {}
            file = str((site or {}).get("file") or "")
            callable_id = str(binding.get("callable_id") or binding.get("function") or "")
            function = str(binding.get("function") or "")
            parameters = [str(value) for value in binding.get("function_parameters") or []]
            target = str(binding.get("target_symbol") or binding.get("target") or "")
            binding["target_identity"] = self.symbol_identity(
                target,
                file=file,
                callable_id=callable_id,
                function_name=function,
                function_parameters=parameters,
            ).model_dump()
            expression = str(binding.get("source_symbol") or binding.get("expression") or "")
            reference_identities: dict[str, dict[str, Any]] = {}
            for symbol in source_expression_names(expression):
                reference_identities[exact_symbol(symbol)] = self.symbol_identity(
                    symbol,
                    file=file,
                    callable_id=callable_id,
                    function_name=function,
                    function_parameters=parameters,
                ).model_dump()
            binding["reference_identities"] = reference_identities
            binding["function_owner"] = self.callable_owner(callable_id, function)
            enriched.append(binding)
        return enriched

    def enrich_calls(self, calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for raw in calls:
            call = dict(raw)
            callable_id = str(call.get("callable_id") or call.get("function") or "")
            function = str(call.get("function") or "")
            owner = self.callable_owner(callable_id, function)
            receiver = str(call.get("receiver") or "").replace("->", ".")
            root = receiver.split(".", 1)[0].lstrip("&*")
            declaring_owner = self.declaring_member_owner(owner, root) if owner and root else None
            member = self.members.get((declaring_owner, root)) if declaring_owner else None
            receiver_type = str((member or {}).get("type") or "")
            receiver_type = receiver_type.rstrip("*& ").split("<", 1)[0].strip()
            call["caller_owner"] = owner
            call["receiver_type"] = receiver_type
            enriched.append(call)
        return enriched


@dataclass
class ExpansionCandidate:
    file: str
    facts: SourceFileFacts
    matched_kind: GapKind
    matched_identity: str


class SourceExpansionResolver:
    """Retrieve broadly, then admit only exact source definitions."""

    def __init__(
        self,
        profiler: MechanismSourceProfiler,
        source_hash: str,
    ) -> None:
        self.profiler = profiler
        self.source_hash = source_hash
        self._facts: dict[str, SourceFileFacts] = {}
        self._candidate_facts: dict[tuple[Any, ...], SourceFileFacts] = {}

    def facts_for(self, file_path: str) -> SourceFileFacts:
        if file_path not in self._facts:
            self._facts[file_path] = extract_facts_for_file(
                self.profiler, file_path, self.source_hash
            )
        return self._facts[file_path]

    def companion_files(self, file_path: str) -> list[str]:
        path = Path(file_path)
        out: list[str] = []
        for suffix in (".h", ".hpp", ".hh", ".hxx", ".cpp", ".cc", ".cxx", ".c"):
            candidate = path.with_suffix(suffix).as_posix()
            if candidate != file_path and self.profiler.source.file_exists(candidate):
                out.append(candidate)
        return out

    def resolve(
        self,
        reference: UnresolvedSourceReference,
        structure: SourceStructureIndex,
    ) -> list[ExpansionCandidate]:
        if (
            reference.kind == "symbol"
            and reference.identity is not None
            and reference.identity.kind == "local"
        ):
            # All producers for a local belong to the same callable, whose
            # admitted file facts have already been indexed. A tree-wide
            # spelling search cannot discover a valid local producer.
            return []
        if reference.kind == "callable" and reference.receiver:
            receiver_type = structure.member_receiver_type(
                reference.class_owner, reference.receiver
            )
            if not receiver_type:
                # A method name has no source identity without the receiver's
                # declared type. Bare-name retrieval cannot make it safer.
                return []
        queries = self._queries(reference)
        if not queries:
            return []
        if reference.kind == "callable" and len(queries) > 1:
            query_groups = [[queries[0]], queries[1:]]
        elif reference.kind == "symbol" and len(queries) > 1:
            # Assignment-shaped queries have structural meaning. The final
            # bare-name query preserves completeness for unusual formatting,
            # but only after no exact-shaped query yields a definition.
            query_groups = [queries[:-1], queries[-1:]]
        else:
            query_groups = [queries]
        for query_group in query_groups:
            if not query_group:
                continue
            hits = self.profiler.search_related_source_files(
                query_group,
                max_files=None,
                expand_query_tokens=False,
            )
            candidates: list[ExpansionCandidate] = []
            for file_path in dedupe_keep_order(hit.file for hit in hits):
                candidate_facts = self._candidate_facts_for(
                    file_path, reference
                )
                if not self._contains_exact_definition(reference, candidate_facts):
                    continue
                match = self._exact_match(reference, candidate_facts, structure)
                if match:
                    facts = self.facts_for(file_path)
                    candidates.append(
                        ExpansionCandidate(
                            file=file_path,
                            facts=facts,
                            matched_kind=reference.kind,
                            matched_identity=match,
                        )
                    )
            if candidates:
                if reference.kind in {"callable", "class"}:
                    # A callable or class definition is one source entity.
                    # Multiple exact definitions usually represent overloads,
                    # platform alternatives, or duplicate ownership; without
                    # build/type evidence, admitting all of them invents a
                    # union mechanism.
                    return candidates if len(candidates) == 1 else []
                return candidates
        return []

    def _candidate_facts_for(
        self,
        file_path: str,
        reference: UnresolvedSourceReference,
    ) -> SourceFileFacts:
        key = (file_path, *reference.visit_key())
        cached = self._candidate_facts.get(key)
        if cached is not None:
            return cached
        backend = str(
            getattr(self.profiler, "source_parser_backend", "legacy")
        )
        if backend in {"tree_sitter", "compare"}:
            from flight_log_agent.px4.tree_sitter_source import (
                TreeSitterSourceExtractor,
            )

            extractor = getattr(
                self.profiler, "_tree_sitter_source_extractor", None
            )
            if extractor is None:
                extractor = TreeSitterSourceExtractor(self.profiler)
                setattr(
                    self.profiler,
                    "_tree_sitter_source_extractor",
                    extractor,
                )
            facts = extractor.extract_admission(
                file_path,
                self.source_hash,
                kind=reference.kind,
                symbol=reference.symbol,
            )
        else:
            structure: dict[str, list[Any]] = {}
            if reference.kind in {"callable", "class", "symbol"}:
                structure = self.profiler.extract_source_structure_from_source(
                    [file_path], expand_companions=False
                )
            assignments = []
            if reference.kind in {"symbol", "constant"}:
                assignments = [
                    item
                    for item in self.profiler.extract_source_assignments_from_source(
                        [file_path],
                        expand_companions=False,
                        include_pointer_outputs=False,
                        include_control_flow=False,
                    )
                    if item.file == file_path
                ]
            facts = SourceFileFacts(
                file=file_path,
                source_hash=self.source_hash,
                parser_backend="legacy:admission",
                source_assignments=assignments,
                classes=list(structure.get("classes") or []),
                members=list(structure.get("members") or []),
                callables=list(structure.get("callables") or []),
                includes=list(structure.get("includes") or []),
            )
        self._candidate_facts[key] = facts
        return facts

    @staticmethod
    def _contains_exact_definition(
        reference: UnresolvedSourceReference,
        facts: SourceFileFacts,
    ) -> bool:
        bare = reference.symbol.replace("->", ".").rsplit(".", 1)[-1].strip("()")
        if reference.kind == "class":
            return any(item.name == bare for item in facts.classes)
        if reference.kind == "callable":
            return any(
                (
                    item.name == reference.symbol
                    or item.name.rsplit("::", 1)[-1] == bare
                )
                and (
                    reference.argument_count is None
                    or len(item.parameters) == reference.argument_count
                )
                for item in facts.callables
            )
        symbol = exact_symbol(reference.symbol)
        for assignment in facts.source_assignments:
            if reference.kind == "constant" and assignment.declaration_kind not in {
                "enum",
                "define",
                "constexpr",
            }:
                continue
            if symbol_produces_reference(exact_symbol(assignment.target), symbol):
                return True
        return False

    @staticmethod
    def _queries(reference: UnresolvedSourceReference) -> list[str]:
        symbol = str(reference.symbol or "").strip()
        if not symbol:
            return []
        bare = symbol.replace("->", ".").rsplit(".", 1)[-1].strip("()")
        root = symbol.replace("->", ".").split(".", 1)[0].lstrip("&*")
        if reference.kind == "callable":
            return [f"::{bare}(", f"{bare}("]
        if reference.kind == "class":
            return [f"class {bare}", f"struct {bare}"]
        if reference.kind == "constant":
            return [bare]
        return dedupe_keep_order(
            [f"{root} =", f"{root}=", f"{root}{{", f"{bare} =", bare]
        )

    @staticmethod
    def _exact_match(
        reference: UnresolvedSourceReference,
        facts: SourceFileFacts,
        structure: SourceStructureIndex,
    ) -> str:
        symbol = exact_symbol(reference.symbol)
        bare = reference.symbol.replace("->", ".").rsplit(".", 1)[-1].strip("()")
        if reference.kind == "class":
            return next(
                (item.name for item in facts.classes if item.name == bare), ""
            )
        if reference.kind == "callable":
            matches = [
                item
                for item in facts.callables
                if item.name == reference.symbol
                or item.name.rsplit("::", 1)[-1] == bare
            ]
            if reference.argument_count is not None:
                matches = [
                    item
                    for item in matches
                    if len(item.parameters) == reference.argument_count
                ]
            if reference.class_owner:
                expected_owner = reference.class_owner
                if reference.receiver:
                    receiver_type = structure.member_receiver_type(
                        reference.class_owner, reference.receiver
                    )
                    if not receiver_type:
                        return ""
                    expected_owner = receiver_type
                lineage = set(structure.lineage(expected_owner))
                if reference.receiver:
                    matches = [item for item in matches if item.owner in lineage]
                else:
                    owned = [item for item in matches if item.owner in lineage]
                    if owned:
                        matches = owned
            else:
                matches = [item for item in matches if not item.owner]
            return matches[0].callable_id if len(matches) == 1 else ""

        bindings = []
        for assignment in facts.source_assignments:
            target = exact_symbol(assignment.target)
            if symbol_produces_reference(target, symbol):
                bindings.append(assignment)
        if reference.kind == "constant":
            bindings = [
                item
                for item in bindings
                if item.declaration_kind in {"enum", "define", "constexpr"}
            ]
        if not bindings:
            return ""
        declared_constants = [
            item
            for item in bindings
            if item.declaration_kind in {"enum", "define", "constexpr"}
        ]
        if declared_constants:
            return exact_symbol(declared_constants[0].target)
        if reference.identity is None:
            return exact_symbol(bindings[0].target)
        local_structure = SourceStructureIndex.from_facts([facts])
        direct_bases = {
            key: set(value) for key, value in structure.direct_bases.items()
        }
        for key, value in local_structure.direct_bases.items():
            direct_bases.setdefault(key, set()).update(value)
        callables_by_name = {
            key: list(value) for key, value in structure.callables_by_name.items()
        }
        for key, value in local_structure.callables_by_name.items():
            bucket = callables_by_name.setdefault(key, [])
            known = {str(item.get("callable_id") or "") for item in bucket}
            bucket.extend(
                item
                for item in value
                if str(item.get("callable_id") or "") not in known
            )
        includes = {key: set(value) for key, value in structure.includes.items()}
        for key, value in local_structure.includes.items():
            includes.setdefault(key, set()).update(value)
        combined = SourceStructureIndex(
            direct_bases=direct_bases,
            members={**structure.members, **local_structure.members},
            callables_by_id={**structure.callables_by_id, **local_structure.callables_by_id},
            callables_by_name=callables_by_name,
            includes=includes,
        )
        for assignment in bindings:
            candidate = combined.symbol_identity(
                assignment.target,
                file=assignment.file,
                callable_id=str(assignment.callable_id or assignment.function or ""),
                function_name=str(assignment.function or ""),
                function_parameters=assignment.function_parameters,
            )
            if combined.compatible(reference.identity, candidate):
                return exact_symbol(assignment.target)
        return ""


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return dict(vars(value))
