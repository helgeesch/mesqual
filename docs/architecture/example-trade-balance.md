# Example: trade balance per area and partner

This example adds a trade balance to the synthetic platform `mesqual.testing`: for each country and each partner country, the net export per time step. It shows an area spec on a border base, a parameter, a schema with a `dimension` level, a custom aggregation, and the fetch, `flag_info` and cache key output of a real run. The rules are in [flags-and-layers.md](flags-and-layers.md).

## 1. What a trade balance is

For each area `A`, each partner `B` that shares a border with `A`, and each time step:

- `net_exp`: the net export of `A` to `B`. Positive when power flows from `A` to `B`.
- `exp = max(net_exp, 0)`, `imp = max(-net_exp, 0)`, `total = exp + imp`.

The sign follows the line flows: a line flow `up` runs from `node_from` to `node_to`, a border flow `up` from `area_from` to `area_to`, and `net_exp` from the area to the partner.

## 2. Which kind of spec

A trade balance is an `AreaFlagSpec` with a border base, the border flow: its aggregation gets the border model as mapping and builds one column per area, partner and variable.

## 3. Losses

The synthetic platform has the line flow `Line.Results.flow_{side}_{direction}` with the sides `sent` (where the flow enters the line, before losses) and `received` (where it leaves the line, after losses), and the border flow `CountryBorder.Results.flow_{side}_{direction}` on it. For the border `A - B`:

| `losses` | Export of `A` to `B` | Import of `A` from `B` |
|---|---|---|
| `'at_own_end'` (default) | `flow_sent_up` | `flow_received_down` |
| `'excluded'` | `flow_sent_up` | `flow_sent_down` |

With `'at_own_end'`, each area sees the flow at its own end of the lines, and `net_exp(A, B) + net_exp(B, A)` equals the losses on the border. With `'excluded'`, the two sides cancel.

## 4. Specs, aggregation and registration

```python
from dataclasses import dataclass

from mesqual.cache import CacheKey, PickleCache
from mesqual.energy_data_handling.variable_utils import trade_balance_frame
from mesqual.flag import AreaFlagSpec, Dimension, FrameSchema, Level
from mesqual.layer_flags import AreaAggregation, AreaFlagInterpreter
from mesqual.testing import DummyDataset, DummyScenario
from mesqual.testing.dataset import DummyBorderFlags

SIDES, DIRECTIONS = ('sent', 'received'), ('up', 'down')

LOSSES = Dimension('losses', 'How line losses enter the trade balance.', {
    'at_own_end': 'export before losses, import after losses',
    'excluded': 'export and import before losses',
}, default='at_own_end')


@dataclass(frozen=True, kw_only=True)
class TradeBalancePerPartner(AreaAggregation):
    """Net export of each area to each partner, from the border flows."""

    def input_flags(self, dataset, layer, variant):
        return tuple(variant.base_flag(side=s, direction=d) for s in SIDES for d in DIRECTIONS)

    def aggregate(self, values, mapping, dataset, layer, variant, effective_config, losses, **params):
        flows = {
            (s, d): self.fetch(dataset, variant.base_flag(side=s, direction=d), effective_config, params)
            for s in SIDES
            for d in DIRECTIONS
        }
        import_side = {'at_own_end': 'received', 'excluded': 'sent'}[losses]
        net_exp = flows['sent', 'up'] - flows[import_side, 'down']  # area_from to area_to
        net_exports = {(row['area_from'], row['area_to']): net_exp[border] for border, row in mapping.iterrows()}
        return trade_balance_frame(net_exports, layer.index, f'partner_{layer.index}', net_exp.index)


TRADE_BALANCE_SPEC = AreaFlagSpec(
    name='{area_layer}.Results.trade_balance',
    base=DummyBorderFlags.FLOW_SPEC,
    aggregation=TradeBalancePerPartner(),
    description='Trade balance of each area with each partner area. net_exp is positive from the area to the partner.',
    consumes=('side', 'direction'),
    params=(LOSSES,),
    schema=lambda variant, layer: FrameSchema(
        index=(Level('time', 'time'),),
        columns=(
            Level(layer.index, 'object'),
            Level(f'partner_{layer.index}', 'counterpart'),
            Level('variable', 'dimension'),
        ),
    ),
)


class StudyDataset(DummyDataset):
    pass


@StudyDataset.register_interpreter
class TradeBalanceFlags(AreaFlagInterpreter):
    flag_specs = (TRADE_BALANCE_SPEC,)
```

- The spec consumes `side` and `direction`: they are not in its name, and `aggregate` gets `values=None`. The aggregation fetches the four border flows with `variant.base_flag(...)`, so it needs no knowledge of the flag grammar, and through `self.fetch`, which passes each flow the call config and only the parameters that its spec declares.
- With consumed dimensions, the availability check reads only the base flag of their first values (`flow_sent_up`). So the aggregation lists all four flows in `input_flags`.
- The border model has one row per border and direction, so one assignment per row gives the net export of every pair.

## 5. Output of a run

The dataset caches its results in a `PickleCache` in the folder `cache`:

```python
ds = StudyDataset(DummyScenario(name='base', days=1), cache=PickleCache('cache'))
trade_balance = ds.fetch('Country.Results.trade_balance')
print(trade_balance.head(3).round(1))
```

```text
Country                        C1                         C2
partner_Country                C2                         C1
variable                  net_exp  exp    imp  total net_exp    exp  imp  total
time
2024-01-01 00:00:00+01:00  -361.7  0.0  361.7  361.7   394.2  394.2  0.0  394.2
2024-01-01 01:00:00+01:00  -274.3  0.0  274.3  274.3   302.4  302.4  0.0  302.4
2024-01-01 02:00:00+01:00  -431.6  0.0  431.6  431.6   462.7  462.7  0.0  462.7
```

```python
info = ds.flag_info('Country.Results.trade_balance').as_record()
```

`info`, abridged to the fields that show a rule:

```text
{'variant': {'area_layer': 'Country'},
 'params': [{'name': 'losses',
             'owner': 'spec',
             'description': 'How line losses enter the trade balance.',
             'values': [{'value': 'at_own_end', 'description': 'export before losses, import after losses'},
                        {'value': 'excluded', 'description': 'export and import before losses'}],
             'default': 'at_own_end',
             'default_source': 'spec'}],
 'model_flag': 'Country.Model',
 'availability_reason': 'TradeBalanceFlags._is_available: inputs available: '
                        'CountryBorder.Results.flow_sent_up, CountryBorder.Model, '
                        'CountryBorder.Results.flow_sent_down, CountryBorder.Results.flow_received_up, '
                        'CountryBorder.Results.flow_received_down',
 'chain': ['TradeBalanceFlags'],
 'base': '{border_layer}.Results.flow_{side}_{direction}',
 'aggregation': 'TradeBalancePerPartner'}
```

## 6. Cache key

```python
record = ds.get_fetch_record('Country.Results.trade_balance')
print(CacheKey.of(record))
print(record.version)
```

```text
CacheKey(dataset_class='__main__.StudyDataset', dataset_name='base', flag='Country.Results.trade_balance', params=(('losses', "'at_own_end'"),))
(('TradeBalanceFlags', 1), ('TradeBalancePerPartner', {'version': 1}))
```

So `losses='excluded'` gives another entry, and a version bump of the aggregation or of an interpreter in the record tree makes the entry stale.
