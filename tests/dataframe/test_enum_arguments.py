"""Enumeration-argument functions (``extract``, ``round_temporal``, ...).

Some standard functions take *enumeration arguments* -- the Substrait spec
(``functions_datetime.yaml``) declares them under ``args:`` with an ``options:``
domain rather than a ``value:`` type. ``extract`` is the canonical example::

    - name: "extract"
      impls:
        - args:
            - name: component        # enumeration argument
              options: [ YEAR, ISO_YEAR, US_YEAR, UNIX_TIME ]
            - name: x                # value argument
              value: date
          return: i64

Per the spec these serialize into ``ScalarFunction.arguments`` as
``FunctionArgument.enum`` (interleaved with the value arguments, in signature
order) -- NOT into ``ScalarFunction.options``, which carries only *behavioral*
options (``overflow``, ``rounding``, ...). Consumers such as DuckDB read enum
selections exclusively from ``arguments``.

These tests guard two properties of the DataFrame / builder pipeline:

1. **Resolution.** ``sub.f.extract(col, component="YEAR")`` must build. The
   overload's arity counts the enum positions, so the registry has to fold the
   enum selection into the signature it matches -- otherwise the value-only
   signature never matches and resolution raises ``Unknown function extract``.
2. **Serialization.** The enum selection must land in ``arguments`` as a
   ``FunctionArgument.enum``, not in ``options``. Routing it into ``options``
   produces a plan that omits the enum from ``arguments`` -- which DuckDB's
   consumer then reads as an empty enum vector and crashes on.

The call site here -- enum selections passed as keyword arguments -- is the
user-facing API and is independent of how resolution is implemented internally.
"""

import substrait.algebra_pb2 as stalg

import substrait.dataframe as sub
from substrait.builders.type import precision_timestamp


def _scalar_functions(message) -> list:
    """Every ``Expression.ScalarFunction`` reachable anywhere in a proto tree."""
    target = stalg.Expression.ScalarFunction.DESCRIPTOR.full_name
    found: list = []

    def walk(msg):
        for field, value in msg.ListFields():
            if field.message_type is None:
                continue
            items = value if field.is_repeated else [value]
            for item in items:
                if field.message_type.full_name == target:
                    found.append(item)
                walk(item)

    walk(message)
    return found


def _function_name(plan, scalar_function) -> str:
    """Resolve a ScalarFunction's declared extension name via its anchor."""
    ref = scalar_function.function_reference
    for decl in plan.extensions:
        fn = decl.extension_function
        if fn.function_anchor == ref:
            return fn.name
    return ""


def _arg_kinds(scalar_function) -> list:
    return [a.WhichOneof("arg_type") for a in scalar_function.arguments]


def test_extract_with_a_single_enum_argument_resolves():
    """The two-argument ``extract(component, date)`` overload must build."""
    df = sub.read_named_table("t", {"d": sub.date})

    # Must not raise "Unknown function extract".
    plan = df.with_columns(y=sub.f.extract(sub.col("d"), component="YEAR")).to_plan()

    (extract,) = [
        sf
        for sf in _scalar_functions(plan)
        if _function_name(plan, sf).startswith("extract")
    ]
    assert extract, "extract did not resolve to a registry function"


def test_extract_serializes_the_enum_as_an_argument_not_an_option():
    """component=YEAR must be a FunctionArgument.enum, in signature position.

    Spec order for this overload is [component (enum), x (value)], so the
    positional value operand follows the enum selection.
    """
    df = sub.read_named_table("t", {"d": sub.date})

    plan = df.with_columns(y=sub.f.extract(sub.col("d"), component="YEAR")).to_plan()

    (extract,) = [
        sf
        for sf in _scalar_functions(plan)
        if _function_name(plan, sf).startswith("extract")
    ]
    assert _arg_kinds(extract) == ["enum", "value"], (
        f"enum not carried in arguments in signature order: {_arg_kinds(extract)}"
    )
    assert extract.arguments[0].enum == "YEAR"
    assert list(extract.options) == [], (
        f"enum selection leaked into options (crashes DuckDB): {extract.options}"
    )


def test_extract_day_of_month_carries_both_enum_arguments():
    """The exact combination behind the reported DuckDB crash.

    ``extract(component="DAY", indexing="ONE", <timestamp>)`` has two enum
    arguments; both must appear in ``arguments`` (order [component, indexing, x])
    and neither in ``options``.
    """
    df = sub.read_named_table("t", {"ts": precision_timestamp(6)})

    plan = df.with_columns(
        dom=sub.f.extract(sub.col("ts"), component="DAY", indexing="ONE")
    ).to_plan()

    (extract,) = [
        sf
        for sf in _scalar_functions(plan)
        if _function_name(plan, sf).startswith("extract")
    ]
    assert _arg_kinds(extract) == ["enum", "enum", "value"], _arg_kinds(extract)
    assert [extract.arguments[0].enum, extract.arguments[1].enum] == ["DAY", "ONE"]
    assert list(extract.options) == [], (
        f"enum selections leaked into options (crashes DuckDB): {extract.options}"
    )
