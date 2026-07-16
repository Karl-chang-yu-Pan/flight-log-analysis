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
from flight_log_agent.px4.mechanism_source_profiler import (
    MechanismSourceProfiler,
    callable_accepts_argument_count,
    callable_parameter_count,
)
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
    declared_classes: set[str] = field(default_factory=set)
    class_files: dict[str, set[str]] = field(default_factory=dict)
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
                    index.declared_classes.add(name)
                    file = str(item.get("file") or "")
                    if file:
                        index.class_files.setdefault(name, set()).add(file)
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
            index.direct_bases.setdefault(owner, set())
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
                    base_name,
                )
                if resolved:
                    index.direct_bases.setdefault(owner, set()).add(resolved)
        return index

    def resolve_class_name(self, class_name: str, lexical_owner: str = "") -> str:
        """Resolve a source-declared type in its C++ lexical class scope."""
        name = _source_type_name(class_name)
        if not name:
            return ""
        declared = set(self.declared_classes)
        declared.update(self.direct_bases)
        declared.update(owner for owner, _member in self.members)
        declared.update(
            str(item.get("owner") or "")
            for item in self.callables_by_id.values()
            if item.get("owner")
        )
        if name in declared:
            return name

        owner = _source_type_name(lexical_owner)
        owner_parts = owner.split("::") if owner else []
        for depth in range(len(owner_parts), -1, -1):
            candidate = "::".join([*owner_parts[:depth], name])
            if candidate in declared:
                return candidate

        suffix = f"::{name}"
        matches = sorted(
            candidate
            for candidate in declared
            if candidate == name or candidate.endswith(suffix)
        )
        return matches[0] if len(matches) == 1 else name

    def lineage(self, class_name: str, lexical_owner: str = "") -> list[str]:
        """Return ``class_name`` followed by all derivable base classes."""
        ordered: list[str] = []
        resolved = self.resolve_class_name(class_name, lexical_owner)
        frontier = [resolved] if resolved else []
        seen: set[str] = set()
        while frontier:
            current = self.resolve_class_name(frontier.pop(0))
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
        return self.resolve_class_name(
            receiver_type,
            lexical_owner=declaring or class_name,
        )

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
            receiver_type = self.member_receiver_type(owner, root) if owner and root else ""
            call["caller_owner"] = owner
            call["receiver_type"] = receiver_type
            enriched.append(call)
        return enriched


def callable_dispatch_context(
    reference: UnresolvedSourceReference,
    structure: SourceStructureIndex,
) -> tuple[str, str, tuple[str, ...]]:
    """Return ``(dispatch kind, short name, source-derived owners)``.

    Owners come from explicit qualification, the receiver's declared type,
    or the lexical caller class. They are source identity, not a filename or
    method-name heuristic. A genuinely non-member call has no owner.
    """
    symbol = str(reference.symbol or "").replace("->", ".").strip()
    bare = symbol.rsplit(".", 1)[-1].rsplit("::", 1)[-1].strip("()")
    explicitly_qualified = "::" in symbol and not reference.receiver
    if reference.receiver:
        receiver_type = structure.member_receiver_type(
            reference.class_owner, reference.receiver
        )
        owners = tuple(structure.lineage(receiver_type)) if receiver_type else ()
        return "receiver", bare, owners
    if explicitly_qualified:
        owner = symbol.rpartition("::")[0].strip(":")
        owners = tuple(structure.lineage(owner)) or ((owner,) if owner else ())
        return "qualified", bare, owners
    if reference.class_owner:
        return (
            "unqualified_member",
            bare,
            tuple(structure.lineage(reference.class_owner)),
        )
    return "free", bare, ()


def source_reference_resolution_key(
    reference: UnresolvedSourceReference,
    structure: SourceStructureIndex,
) -> tuple[Any, ...]:
    """Identity of one source lookup, independent of runtime call sites."""
    if reference.kind == "callable":
        dispatch, bare, owners = callable_dispatch_context(reference, structure)
        contextual_file = reference.file if dispatch == "free" else ""
        return (
            reference.kind,
            dispatch,
            bare,
            owners,
            reference.argument_count,
            contextual_file,
        )
    if reference.identity is not None:
        return (reference.kind, reference.identity.storage_key())
    return reference.visit_key()


def reference_receiver_is_source_boundary(
    reference: UnresolvedSourceReference,
    boundary_bindings: Sequence[dict[str, Any]],
    structure: SourceStructureIndex,
) -> bool:
    """Whether a callable receiver is one exact source-proven I/O endpoint."""
    if reference.kind != "callable" or not reference.receiver:
        return False
    receiver = exact_symbol(reference.receiver)
    if receiver.startswith("this."):
        receiver = receiver[5:]
    caller = str(reference.callable_id or "").split("::@call:", 1)[0]
    lineage = set(structure.lineage(reference.class_owner))
    candidates: list[dict[str, Any]] = []
    for item in boundary_bindings:
        source_symbol = exact_symbol(str(item.get("source_symbol") or ""))
        if source_symbol != receiver or not item.get("topic"):
            continue
        endpoint_kind = str(item.get("endpoint_kind") or "")
        item_owner = str(item.get("source_owner") or "")
        item_callable = str(item.get("callable_id") or "")
        item_file = str(item.get("file") or "")
        if not endpoint_kind:
            endpoint_kind = (
                "member" if item_owner else "local" if item_callable else "global"
            )
        if endpoint_kind in {"member", "base"}:
            if item_owner and item_owner in lineage:
                candidates.append(item)
        elif endpoint_kind == "local":
            if caller and item_callable == caller:
                candidates.append(item)
        elif endpoint_kind == "global":
            if reference.file and item_file == reference.file:
                candidates.append(item)
    placements = {
        (str(item.get("topic") or ""), item.get("instance"))
        for item in candidates
    }
    return len(placements) == 1


@dataclass
class ExpansionCandidate:
    file: str
    facts: SourceFileFacts
    matched_kind: GapKind
    matched_identity: str


class SourceExpansionResolver:
    """Resolve owners first, then admit only exact source definitions."""

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
        if reference.kind == "callable":
            direct_files = self._callable_owner_files(reference, structure)
            candidates = self._admit_files(reference, structure, direct_files)
            if candidates:
                return self._unique_entity(reference, candidates)
            query_groups = self._callable_query_groups(reference, structure)
        else:
            queries = self._queries(reference)
            if not queries:
                return []
            query_groups: list[list[str]]
            if reference.kind == "symbol" and len(queries) > 1:
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
            hit_files = dedupe_keep_order(hit.file for hit in hits)
            if reference.kind == "callable" and query_group and all(
                query.startswith(("class ", "struct ", "namespace "))
                for query in query_group
            ):
                hit_files = dedupe_keep_order(
                    file_path
                    for hit_file in hit_files
                    for file_path in [hit_file, *self.companion_files(hit_file)]
                )
            candidates = self._admit_files(reference, structure, hit_files)
            if candidates:
                return self._unique_entity(reference, candidates)
        return []

    def resolution_key(
        self,
        reference: UnresolvedSourceReference,
        structure: SourceStructureIndex,
    ) -> tuple[Any, ...]:
        return source_reference_resolution_key(reference, structure)

    def _callable_owner_files(
        self,
        reference: UnresolvedSourceReference,
        structure: SourceStructureIndex,
    ) -> list[str]:
        _dispatch, _bare, owners = callable_dispatch_context(reference, structure)
        return dedupe_keep_order(
            file_path
            for owner in owners
            for declared_file in sorted(structure.class_files.get(owner, ()))
            for file_path in [declared_file, *self.companion_files(declared_file)]
        )

    @staticmethod
    def _callable_query_groups(
        reference: UnresolvedSourceReference,
        structure: SourceStructureIndex,
    ) -> list[list[str]]:
        dispatch, bare, owners = callable_dispatch_context(reference, structure)
        groups: list[list[str]] = []
        if owners:
            groups.append([f"{owner}::{bare}(" for owner in owners])
            class_queries = dedupe_keep_order(
                query
                for owner in owners
                for query in (
                    f"class {owner.rsplit('::', 1)[-1]}",
                    f"struct {owner.rsplit('::', 1)[-1]}",
                )
            )
            if class_queries:
                groups.append(class_queries)
            if dispatch == "qualified":
                namespace_queries = dedupe_keep_order(
                    query
                    for owner in owners
                    for query in (
                        f"namespace {owner}",
                        f"namespace {owner.rsplit('::', 1)[-1]}",
                    )
                )
                if namespace_queries:
                    groups.append(namespace_queries)
        if dispatch in {"free", "unqualified_member"}:
            groups.append([f"{bare}("])
        return groups

    def _admit_files(
        self,
        reference: UnresolvedSourceReference,
        structure: SourceStructureIndex,
        files: Iterable[str],
    ) -> list[ExpansionCandidate]:
        candidates: list[ExpansionCandidate] = []
        for file_path in dedupe_keep_order(str(value) for value in files if value):
            candidate_facts = self._candidate_facts_for(file_path, reference)
            if reference.kind == "callable":
                if not self._callable_matches(
                    reference,
                    candidate_facts,
                    structure,
                    defaults_required=False,
                ):
                    continue
                full_facts = self.facts_for(file_path)
                match = self._exact_match(reference, full_facts, structure)
                if not match:
                    continue
                candidates.append(
                    ExpansionCandidate(
                        file=file_path,
                        facts=full_facts,
                        matched_kind=reference.kind,
                        matched_identity=match,
                    )
                )
                continue
            if not self._contains_exact_definition(reference, candidate_facts):
                continue
            match = self._exact_match(reference, candidate_facts, structure)
            if not match:
                continue
            candidates.append(
                ExpansionCandidate(
                    file=file_path,
                    facts=self.facts_for(file_path),
                    matched_kind=reference.kind,
                    matched_identity=match,
                )
            )
        return candidates

    @staticmethod
    def _unique_entity(
        reference: UnresolvedSourceReference,
        candidates: list[ExpansionCandidate],
    ) -> list[ExpansionCandidate]:
        if reference.kind in {"callable", "class"}:
            # Multiple exact definitions can be overloads, build variants, or
            # ambiguous base members. Source facts must disambiguate them.
            identities = {item.matched_identity for item in candidates}
            return candidates[:1] if len(identities) == 1 else []
        return candidates

    def _candidate_facts_for(
        self,
        file_path: str,
        reference: UnresolvedSourceReference,
    ) -> SourceFileFacts:
        key = (
            file_path,
            reference.kind,
            exact_symbol(reference.symbol),
        )
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
                    or callable_accepts_argument_count(
                        item, reference.argument_count
                    )
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
            return [f"{bare}("]
        if reference.kind == "class":
            return [f"class {bare}", f"struct {bare}"]
        if reference.kind == "constant":
            return [bare]
        return dedupe_keep_order(
            [f"{root} =", f"{root}=", f"{root}{{", f"{bare} =", bare]
        )

    @staticmethod
    def _callable_matches(
        reference: UnresolvedSourceReference,
        facts: SourceFileFacts,
        structure: SourceStructureIndex,
        *,
        defaults_required: bool,
    ) -> list[Any]:
        bare = reference.symbol.replace("->", ".").rsplit(".", 1)[-1].strip("()")
        matches = [
            item
            for item in facts.callables
            if item.name == reference.symbol
            or item.name.rsplit("::", 1)[-1] == bare
        ]
        if reference.argument_count is not None:
            if defaults_required:
                matches = [
                    item
                    for item in matches
                    if callable_accepts_argument_count(
                        item, reference.argument_count
                    )
                ]
            else:
                # Admission sees only one physical file, so a declaration in
                # a companion header may still supply defaults. It may reject
                # impossible over-arity calls, but final acceptance waits for
                # full companion-aware facts.
                matches = [
                    item
                    for item in matches
                    if reference.argument_count <= callable_parameter_count(item)
                ]
        dispatch, _short, owners = callable_dispatch_context(reference, structure)
        owner_set = set(owners)
        if dispatch == "receiver":
            matches = [item for item in matches if item.owner in owner_set]
        elif dispatch == "qualified":
            matches = [
                item
                for item in matches
                if item.owner in owner_set
                or (
                    not item.owner
                    and item.name.rpartition("::")[0] in owner_set
                )
            ]
        elif dispatch == "unqualified_member":
            owned = [item for item in matches if item.owner in owner_set]
            matches = owned or [item for item in matches if not item.owner]
        else:
            matches = [item for item in matches if not item.owner]
        return matches

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
            matches = SourceExpansionResolver._callable_matches(
                reference,
                facts,
                structure,
                defaults_required=True,
            )
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
        class_files = {
            key: set(value) for key, value in structure.class_files.items()
        }
        for key, value in local_structure.class_files.items():
            class_files.setdefault(key, set()).update(value)
        combined = SourceStructureIndex(
            direct_bases=direct_bases,
            declared_classes={
                *structure.declared_classes,
                *local_structure.declared_classes,
            },
            class_files=class_files,
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


def _source_type_name(value: str) -> str:
    """Return the class-like part of a declared C++ type expression."""
    text = str(value or "").strip()
    text = re.sub(r"\b(?:const|volatile|class|struct|typename)\b", " ", text)
    text = text.replace("*", " ").replace("&", " ")
    text = " ".join(text.split()).strip().lstrip(":")
    return text.split("<", 1)[0].strip()
