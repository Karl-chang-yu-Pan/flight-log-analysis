"""Lossless C++ source projection for parser-obstructing macros.

Tree-sitter parses syntax but does not run the C++ preprocessor.  This module
projects source-proven macro uses into parseable syntax while preserving every
byte offset.  Callers must retain the original source for evidence extraction;
the projected bytes are only an input to the parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from tree_sitter import Node, Parser, Tree


_IDENTIFIER = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class MacroDefinition:
    name: str
    parameters: Optional[tuple[str, ...]]
    replacement: str


@dataclass(frozen=True)
class MacroProjection:
    source: bytes
    tree: Tree
    applied_macros: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Edit:
    start: int
    end: int
    replacement: Optional[bytes]
    macro: str

    def __post_init__(self) -> None:
        if self.replacement is not None and self.end - self.start != len(
            self.replacement
        ):
            raise ValueError("macro projection edits must preserve byte length")


@dataclass(frozen=True)
class _Argument:
    start: int
    end: int
    delimiter: Optional[int]


@dataclass(frozen=True)
class _DeclarationArgument:
    opening: int
    closing: int


DefinitionResolver = Callable[[str], Sequence[MacroDefinition]]


def extract_macro_definitions(text: str, name: str) -> list[MacroDefinition]:
    """Extract every exact ``#define name`` variant from one source file."""
    escaped = re.escape(name)
    start_pattern = re.compile(
        rf"^[ \t]*#[ \t]*define[ \t]+{escaped}(?=[ \t(]|$)(?P<tail>.*)$"
    )
    lines = text.splitlines()
    definitions: list[MacroDefinition] = []
    index = 0
    while index < len(lines):
        match = start_pattern.match(lines[index])
        if match is None:
            index += 1
            continue
        logical = match.group("tail")
        while logical.rstrip().endswith("\\") and index + 1 < len(lines):
            logical = logical.rstrip()[:-1] + " " + lines[index + 1].strip()
            index += 1
        parameters: Optional[tuple[str, ...]] = None
        replacement = logical
        # Function-like macros require the opening parenthesis immediately
        # after the macro name. Whitespace denotes an object-like replacement.
        if logical.startswith("("):
            close = _matching_delimiter(logical.encode("utf-8"), 0, 40, 41)
            if close is None:
                index += 1
                continue
            parameters = tuple(
                item.strip()
                for item in logical[1:close].split(",")
                if item.strip()
            )
            replacement = logical[close + 1 :]
        definitions.append(
            MacroDefinition(
                name=name,
                parameters=parameters,
                replacement=replacement.strip(),
            )
        )
        index += 1
    return definitions


class CppMacroProjector:
    """Produce an offset-preserving parse view of source-defined macros."""

    def __init__(
        self,
        parser: Parser,
        resolve_definitions: DefinitionResolver,
    ) -> None:
        self._parser = parser
        self._resolve_definitions = resolve_definitions

    def project(self, source: bytes) -> MacroProjection:
        tree = self._parser.parse(source)
        if not tree.root_node.has_error:
            return MacroProjection(source=source, tree=tree)

        projected = source
        applied: list[str] = []
        while tree.root_node.has_error:
            current_score = _error_score(tree)
            candidates = [
                *self._declaration_list_edits(source, projected, tree),
                *self._attribute_edits(source, projected, tree),
            ]
            accepted = False
            for macro, edits in candidates:
                tentative = _apply_edits(projected, edits)
                tentative_tree = self._parser.parse(tentative)
                if _error_score(tentative_tree) >= current_score:
                    continue
                projected = tentative
                tree = tentative_tree
                applied.append(macro)
                accepted = True
                break
            if not accepted:
                break
        return MacroProjection(
            source=projected,
            tree=tree,
            applied_macros=tuple(applied),
        )

    def _declaration_list_edits(
        self,
        original: bytes,
        projected: bytes,
        tree: Tree,
    ) -> list[tuple[str, list[_Edit]]]:
        candidates: list[tuple[str, list[_Edit]]] = []
        for node in _walk(tree.root_node):
            if node.type != "field_declaration" or not node.has_error:
                continue
            type_node = node.child_by_field_name("type")
            if type_node is None or type_node.type not in {
                "identifier",
                "type_identifier",
            }:
                continue
            name = _text(original, type_node).strip()
            if not name:
                continue
            definitions = self._resolve_definitions(name)
            if not definitions or any(
                definition.parameters is None for definition in definitions
            ):
                continue
            opening = _skip_space(projected, type_node.end_byte, node.end_byte)
            if opening >= node.end_byte or projected[opening] != ord("("):
                continue
            closing = _matching_delimiter(projected, opening, ord("("), ord(")"))
            if closing is None or closing >= node.end_byte:
                continue
            arguments = _split_arguments(projected, opening + 1, closing)
            declaration_arguments = [
                self._declaration_argument(projected, argument)
                for argument in arguments
            ]
            first_declaration = next(
                (
                    index
                    for index, declaration in enumerate(declaration_arguments)
                    if declaration is not None
                ),
                None,
            )
            if first_declaration is None or any(
                declaration is None
                for declaration in declaration_arguments[first_declaration:]
            ):
                continue
            if not all(
                self._definition_exposes_arguments(
                    definition,
                    argument_count=len(arguments),
                    first_declaration=first_declaration,
                )
                for definition in definitions
            ):
                continue
            edits = [
                _blank_edit(type_node.start_byte, opening + 1, name),
            ]
            for argument in arguments[:first_declaration]:
                end = (
                    argument.delimiter + 1
                    if argument.delimiter is not None
                    else argument.end
                )
                edits.append(_blank_edit(argument.start, end, name))
            final_terminator_used = False
            for argument, declaration in zip(
                arguments[first_declaration:],
                declaration_arguments[first_declaration:],
            ):
                assert declaration is not None
                edits.extend(
                    [
                        _blank_edit(
                            declaration.opening,
                            declaration.opening + 1,
                            name,
                        ),
                        _blank_edit(
                            declaration.closing,
                            declaration.closing + 1,
                            name,
                        ),
                    ]
                )
                if argument.delimiter is not None:
                    edits.append(
                        _Edit(
                            argument.delimiter,
                            argument.delimiter + 1,
                            b";",
                            name,
                        )
                    )
                else:
                    edits.append(_Edit(closing, closing + 1, b";", name))
                    final_terminator_used = True
            if not final_terminator_used:
                edits.append(_blank_edit(closing, closing + 1, name))
            candidates.append((name, edits))
        return candidates

    @staticmethod
    def _definition_exposes_arguments(
        definition: MacroDefinition,
        *,
        argument_count: int,
        first_declaration: int,
    ) -> bool:
        parameters = definition.parameters
        if parameters is None:
            return False
        replacement = definition.replacement
        variadic_index = next(
            (
                index
                for index, parameter in enumerate(parameters)
                if parameter == "..."
            ),
            None,
        )
        if variadic_index is None and len(parameters) != argument_count:
            return False
        for index in range(first_declaration, argument_count):
            if variadic_index is not None and index >= variadic_index:
                token = "__VA_ARGS__"
            elif index < len(parameters):
                token = parameters[index]
            else:
                return False
            if re.search(rf"\b{re.escape(token)}\b", replacement) is None:
                return False
        return True

    def _declaration_argument(
        self, source: bytes, argument: _Argument
    ) -> Optional[_DeclarationArgument]:
        opening = _skip_trivia(source, argument.start, argument.end)
        if opening >= argument.end or source[opening] != ord("("):
            return None
        closing = _matching_delimiter(source, opening, ord("("), ord(")"))
        if closing is None or closing >= argument.end:
            return None
        identifier_start = _skip_trivia(source, closing + 1, argument.end)
        identifier = _IDENTIFIER.match(source, identifier_start, argument.end)
        if identifier is None:
            return None
        if _skip_trivia(source, identifier.end(), argument.end) != argument.end:
            return None
        type_text = source[opening + 1 : closing].strip()
        if not type_text or not self._is_declaration_type(type_text):
            return None
        return _DeclarationArgument(opening=opening, closing=closing)

    def _is_declaration_type(self, type_text: bytes) -> bool:
        probe = b"class __macro_probe { " + type_text + b" __member; };"
        tree = self._parser.parse(probe)
        if tree.root_node.has_error:
            return False
        return any(
            node.type == "field_declaration" for node in _walk(tree.root_node)
        )

    def _attribute_edits(
        self,
        original: bytes,
        _projected: bytes,
        tree: Tree,
    ) -> list[tuple[str, list[_Edit]]]:
        candidates: list[tuple[str, list[_Edit]]] = []
        seen: set[tuple[int, int]] = set()
        for error in _error_nodes(tree):
            preceding = _preceding_macro_span(original, error.start_byte)
            spans = [
                *([preceding] if preceding is not None else []),
                *_identifiers_in_span(
                    original, error.start_byte, error.end_byte
                ),
            ]
            for start, end in spans:
                if (start, end) in seen:
                    continue
                seen.add((start, end))
                name_match = _IDENTIFIER.match(original, start, end)
                if name_match is None:
                    continue
                name = original[start:name_match.end()].decode("ascii")
                invocation_end = end
                args: list[bytes] = []
                definitions = self._resolve_definitions(name)
                if not definitions:
                    continue
                function_like = all(
                    definition.parameters is not None for definition in definitions
                )
                if function_like:
                    opening = _skip_space(original, name_match.end(), len(original))
                    if opening >= len(original) or original[opening] != ord("("):
                        continue
                    closing = _matching_delimiter(
                        original, opening, ord("("), ord(")")
                    )
                    if closing is None:
                        continue
                    invocation_end = closing + 1
                    args = [
                        original[item.start : item.end].strip()
                        for item in _split_arguments(original, opening + 1, closing)
                    ]
                elif any(
                    definition.parameters is not None for definition in definitions
                ):
                    continue
                if not all(
                    self._definition_is_attribute(definition, args)
                    for definition in definitions
                ):
                    continue
                candidates.append(
                    (name, [_blank_edit(start, invocation_end, name)])
                )
        return candidates

    def _definition_is_attribute(
        self, definition: MacroDefinition, args: Sequence[bytes]
    ) -> bool:
        replacement = definition.replacement
        parameters = definition.parameters
        if parameters is not None:
            if len(parameters) != len(args) or "..." in parameters:
                return False
            for parameter, argument in zip(parameters, args):
                replacement = re.sub(
                    rf"\b{re.escape(parameter)}\b",
                    argument.decode("utf-8", errors="replace"),
                    replacement,
                )
        probe = (replacement + " int __macro_probe;").encode("utf-8")
        tree = self._parser.parse(probe)
        if tree.root_node.has_error:
            return False
        declarations = [
            node
            for node in tree.root_node.named_children
            if node.type == "declaration"
        ]
        if len(declarations) != 1:
            return False
        declaration = declarations[0]
        extras = [
            child
            for child in declaration.named_children
            if child != declaration.child_by_field_name("type")
            and child != declaration.child_by_field_name("declarator")
        ]
        return bool(extras) and all(
            "attribute" in child.type for child in extras
        )


def _walk(node: Node) -> Iterable[Node]:
    yield node
    for child in node.named_children:
        yield from _walk(child)


def _error_nodes(tree: Tree) -> list[Node]:
    return [
        node
        for node in _walk(tree.root_node)
        if node.type == "ERROR" or node.is_missing
    ]


def _error_score(tree: Tree) -> tuple[int, int]:
    errors = _error_nodes(tree)
    return (
        len(errors),
        sum(max(node.end_byte - node.start_byte, 1) for node in errors),
    )


def _text(source: bytes, node: Node) -> str:
    return source[node.start_byte : node.end_byte].decode(
        "utf-8", errors="replace"
    )


def _skip_space(source: bytes, start: int, end: int) -> int:
    while start < end and source[start] in b" \t\r\n":
        start += 1
    return start


def _skip_trivia(source: bytes, start: int, end: int) -> int:
    while start < end:
        advanced = _skip_space(source, start, end)
        if source[advanced : advanced + 2] == b"//":
            newline = source.find(b"\n", advanced + 2, end)
            start = end if newline < 0 else newline + 1
            continue
        if source[advanced : advanced + 2] == b"/*":
            closing = source.find(b"*/", advanced + 2, end)
            start = end if closing < 0 else closing + 2
            continue
        return advanced
    return start


def _matching_delimiter(
    source: bytes,
    opening: int,
    open_byte: int,
    close_byte: int,
) -> Optional[int]:
    depth = 0
    quote: Optional[int] = None
    escaped = False
    index = opening
    while index < len(source):
        current = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif current == ord("\\"):
                escaped = True
            elif current == quote:
                quote = None
            index += 1
            continue
        if current in {ord('"'), ord("'")}:
            quote = current
        elif source[index : index + 2] == b"//":
            newline = source.find(b"\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
            continue
        elif source[index : index + 2] == b"/*":
            closing = source.find(b"*/", index + 2)
            index = len(source) if closing < 0 else closing + 2
            continue
        elif current == open_byte:
            depth += 1
        elif current == close_byte:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _split_arguments(source: bytes, start: int, end: int) -> list[_Argument]:
    arguments: list[_Argument] = []
    argument_start = start
    stack: list[int] = []
    pairs = {
        ord("("): ord(")"),
        ord("["): ord("]"),
        ord("{"): ord("}"),
        ord("<"): ord(">"),
    }
    closing = set(pairs.values())
    quote: Optional[int] = None
    escaped = False
    index = start
    while index < end:
        current = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif current == ord("\\"):
                escaped = True
            elif current == quote:
                quote = None
        elif current in {ord('"'), ord("'")}:
            quote = current
        elif source[index : index + 2] == b"//":
            newline = source.find(b"\n", index + 2, end)
            index = end if newline < 0 else newline + 1
            continue
        elif source[index : index + 2] == b"/*":
            comment_end = source.find(b"*/", index + 2, end)
            index = end if comment_end < 0 else comment_end + 2
            continue
        elif current in pairs:
            stack.append(pairs[current])
        elif current in closing and stack and current == stack[-1]:
            stack.pop()
        elif current == ord(",") and not stack:
            arguments.append(_Argument(argument_start, index, index))
            argument_start = index + 1
        index += 1
    arguments.append(_Argument(argument_start, end, None))
    return arguments


def _identifiers_in_span(
    source: bytes, start: int, end: int
) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for match in _IDENTIFIER.finditer(source, start, end)
    ]


def _preceding_macro_span(
    source: bytes, position: int
) -> Optional[tuple[int, int]]:
    end = position
    while end > 0 and source[end - 1] in b" \t\r\n":
        end -= 1
    if end <= 0:
        return None
    if source[end - 1] == ord(")"):
        opening = _matching_opening(source, end - 1, ord("("), ord(")"))
        if opening is None:
            return None
        name_end = opening
        while name_end > 0 and source[name_end - 1] in b" \t\r\n":
            name_end -= 1
        match = _identifier_ending_at(source, name_end)
        return (match[0], end) if match is not None else None
    return _identifier_ending_at(source, end)


def _matching_opening(
    source: bytes,
    closing: int,
    open_byte: int,
    close_byte: int,
) -> Optional[int]:
    depth = 0
    for index in range(closing, -1, -1):
        current = source[index]
        if current == close_byte:
            depth += 1
        elif current == open_byte:
            depth -= 1
            if depth == 0:
                return index
    return None


def _identifier_ending_at(source: bytes, end: int) -> Optional[tuple[int, int]]:
    start = end
    while start > 0 and (
        source[start - 1 : start].isalnum() or source[start - 1] == ord("_")
    ):
        start -= 1
    if start == end or _IDENTIFIER.fullmatch(source[start:end]) is None:
        return None
    return start, end


def _blank_edit(start: int, end: int, macro: str) -> _Edit:
    return _Edit(start, end, None, macro)


def _apply_edits(source: bytes, edits: Sequence[_Edit]) -> bytes:
    projected = bytearray(source)
    occupied: set[int] = set()
    for edit in sorted(edits, key=lambda item: (item.start, item.end)):
        if any(index in occupied for index in range(edit.start, edit.end)):
            raise ValueError("overlapping macro projection edits")
        replacement = edit.replacement
        if replacement is None:
            replacement = bytes(
                byte if byte in {ord("\r"), ord("\n")} else ord(" ")
                for byte in source[edit.start : edit.end]
            )
        projected[edit.start : edit.end] = replacement
        occupied.update(range(edit.start, edit.end))
    return bytes(projected)
