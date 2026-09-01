"""Function entry class for extension registry."""

from enum import Enum
from typing import Optional, Union

from substrait.type_pb2 import Type
from substrait_extensions.extensions import simple_extensions as se

from substrait.derivation_expression import _parse, evaluate

from .signature_checker_helpers import covers, normalize_substrait_type_names

_MISSING = object()


class FunctionType(Enum):
    SCALAR = "scalar"
    AGGREGATE = "aggregate"
    WINDOW = "window"


class FunctionEntry:
    def __init__(
        self,
        urn: str,
        name: str,
        impl: Union[se.Impl, se.Impl1, se.Impl2],
        function_type: FunctionType = FunctionType.SCALAR,
    ) -> None:
        self.name = name
        self.impl = impl
        self.normalized_inputs: list = []
        self.urn: str = urn
        self.function_type = function_type
        self.arguments = []
        # Argument names parallel to ``arguments`` / ``normalized_inputs``. Used to
        # match enumeration selections, which arrive by name (e.g. component="YEAR")
        # rather than by position, back to their declared signature slot.
        self.arg_names: list = []
        self.nullability = (
            impl.nullability if impl.nullability else se.NullabilityHandling.MIRROR
        )
        if impl.args:
            for arg in impl.args:
                if isinstance(arg, se.ValueArg):
                    self.arguments.append(_parse(arg.value))
                    self.normalized_inputs.append(
                        normalize_substrait_type_names(arg.value)
                    )
                    self.arg_names.append(arg.name)
                elif isinstance(arg, se.EnumerationArg):
                    self.arguments.append(arg.options)
                    self.normalized_inputs.append("req")
                    self.arg_names.append(arg.name)

    def __repr__(self) -> str:
        return f"{self.name}:{'_'.join(self.normalized_inputs)}"

    def interleave_arguments(self, value_items, options):
        """Order value operands and enumeration selections per the signature.

        Substrait interleaves enumeration arguments with value arguments in
        declared order (``extract`` is ``[component (enum), x (value)]``), but the
        two arrive separately: value operands positionally, enumeration selections
        by argument name in ``options``. This walks the declared arguments and
        weaves them back together.

        ``value_items`` is an iterator of value operands (consumed in order for
        each value argument). Returns ``(ordered, remaining_options)`` where each
        entry of ``ordered`` is ``("value", item)`` or ``("enum", selection)``, and
        ``remaining_options`` are the options *not* consumed as enumeration
        arguments -- i.e. the behavioral options. Returns ``None`` if the value
        arity is wrong or a required enumeration argument is absent from
        ``options``.
        """
        remaining = dict(options or {})
        if self.impl.variadic:
            # Variadic functions have no enumeration arguments; every operand is a
            # value and every option is behavioral.
            return [("value", item) for item in value_items], remaining
        ordered: list = []
        for kind, name in zip(self.normalized_inputs, self.arg_names):
            if kind == "req":  # enumeration argument
                if name not in remaining:
                    return None
                ordered.append(("enum", str(remaining.pop(name))))
            else:
                item = next(value_items, _MISSING)
                if item is _MISSING:
                    return None
                ordered.append(("value", item))
        if next(value_items, _MISSING) is not _MISSING:
            return None  # too many value operands
        return ordered, remaining

    def _resolve_signature(
        self, signature: tuple | list, options: Optional[dict]
    ) -> Optional[list]:
        """Interleave ``signature`` with enumeration selections for matching.

        Enumeration selections may be supplied two ways: interleaved into
        ``signature`` as strings in declared order (the low-level registry
        contract), or by argument name in ``options`` (how the DataFrame builders
        pass them, keeping the value-only signature intact). An enum position
        prefers ``options`` and otherwise consumes the next ``signature`` item.
        Returns the value-and-enum sequence to match against ``self.arguments``,
        or ``None`` on an arity mismatch.
        """
        remaining = dict(options or {})
        items = iter(signature)
        interleaved: list = []
        for kind, name in zip(self.normalized_inputs, self.arg_names):
            if kind == "req" and name in remaining:  # enum selection by name
                interleaved.append(str(remaining.pop(name)))
                continue
            item = next(items, _MISSING)
            if item is _MISSING:
                return None
            interleaved.append(item)
        if next(items, _MISSING) is not _MISSING:
            return None  # more operands than the signature declares
        return interleaved

    def satisfies_signature(
        self, signature: tuple | list, options: Optional[dict] = None
    ) -> Optional[str]:
        if self.impl.variadic:
            min_args_allowed = self.impl.variadic.min or 0
            if len(signature) < min_args_allowed:
                return None
            inputs = [self.arguments[0]] * len(signature)
            interleaved: list = list(signature)
        else:
            interleaved = self._resolve_signature(signature, options)
            if interleaved is None:
                return None
            inputs = self.arguments
        if len(inputs) != len(interleaved):
            return None
        zipped_args = list(zip(inputs, interleaved))
        parameters = {}
        for x, y in zipped_args:
            if isinstance(y, str):
                if y not in x:
                    return None
            else:
                if not covers(
                    y,
                    x,
                    parameters,
                    check_nullability=self.nullability
                    == se.NullabilityHandling.DISCRETE,
                ):
                    return None
        output_type = evaluate(self.impl.return_, parameters)
        if self.nullability == se.NullabilityHandling.MIRROR and isinstance(
            output_type, Type
        ):
            sig_contains_nullable = any(
                [
                    p.__getattribute__(p.WhichOneof("kind")).nullability
                    == Type.NULLABILITY_NULLABLE
                    for p in interleaved
                    if isinstance(p, Type)
                ]
            )
            kind = output_type.WhichOneof("kind")
            if kind is not None:
                output_type.__getattribute__(kind).nullability = (
                    Type.NULLABILITY_NULLABLE
                    if sig_contains_nullable
                    else Type.NULLABILITY_REQUIRED
                )
        return output_type
