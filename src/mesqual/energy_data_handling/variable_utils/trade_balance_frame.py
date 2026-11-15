"""Trade balance frame from net exports per area and partner."""

from collections.abc import Mapping

import pandas as pd

NET_EXP_VAR = 'net_exp'
EXP_VAR = 'exp'
IMP_VAR = 'imp'
TOTAL_VAR = 'total'
VARIABLE_LEVEL = 'variable'


def trade_balance_frame(
    net_exports: Mapping[tuple, pd.Series],
    primary_name: str,
    partner_name: str,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Build the trade balance frame from the net export of each area to each partner.

    For each `(primary, partner)` pair, the frame has the variables
    `net_exp`, `exp = max(net_exp, 0)`, `imp = max(-net_exp, 0)` and
    `total = exp + imp`. `exp` and `imp` come from the net value of each
    time step, not from gross flows.

    Args:
        net_exports: Net export per `(primary, partner)` pair, positive from
            primary to partner.
        primary_name: Name of the first column level.
        partner_name: Name of the second column level.
        index: Index of the result if `net_exports` is empty.

    Returns:
        A frame with the column levels `(primary_name, partner_name, 'variable')`.

    Examples:
        >>> net = pd.Series([100.0, -40.0])
        >>> trade_balance_frame({('C1', 'C2'): net, ('C2', 'C1'): -net}, 'country', 'partner_country')
        country              C1                          C2
        partner_country      C2                          C1
        variable        net_exp    exp   imp  total net_exp   exp    imp  total
        0                 100.0  100.0   0.0  100.0  -100.0   0.0  100.0  100.0
        1                 -40.0    0.0  40.0   40.0    40.0  40.0    0.0   40.0
    """
    names = [primary_name, partner_name, VARIABLE_LEVEL]
    if not net_exports:
        return pd.DataFrame(index=index, columns=pd.MultiIndex.from_tuples([], names=names))
    columns = {}
    for (primary, partner), net_exp in net_exports.items():
        exp = net_exp.clip(0)
        imp = net_exp.clip(None, 0).abs()
        columns[(primary, partner, NET_EXP_VAR)] = net_exp
        columns[(primary, partner, EXP_VAR)] = exp
        columns[(primary, partner, IMP_VAR)] = imp
        columns[(primary, partner, TOTAL_VAR)] = exp + imp
    return pd.concat(columns, axis=1, names=names)
