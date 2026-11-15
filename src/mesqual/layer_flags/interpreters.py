"""Interpreters that answer the flags of layer specs on a platform dataset."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pandas as pd

from mesqual.datasets.dataset import Dataset
from mesqual.datasets.dataset_config import DatasetConfig
from mesqual.datasets.describe import Availability
from mesqual.energy_data_handling.area_accounting import AreaMembership
from mesqual.layer_flags.aggregations import LayerAggregation
from mesqual.flag.layer_spec import AreaFlagSpec, BorderFlagSpec, LayerFlagSpec, LayerVariantValues
from mesqual.flag.exceptions import UnknownFlagError
from mesqual.flag.layers import AreaLayer, BorderLayer


class LayerFlagInterpreter(Dataset):
    """Common base of `AreaFlagInterpreter` and `BorderFlagInterpreter`. Not used directly.

    A subclass lists layer specs in `flag_specs` and is registered on a
    platform dataset with `register_interpreter`. It fetches the base flags
    and the layer model through `parent_dataset.fetch`, so overrides of the
    base flags reach the layer flags. It passes `effective_config` to each
    of these fetches and to `aggregate`, so the config of the call reaches
    every input, also the inputs that the aggregation fetches. It passes to
    each fetch only the parameters that the target spec declares, and the
    aggregation parameters only to `aggregate`.

    A spec that consumes base dimensions gets `values=None` in `aggregate`:
    the interpreter does not fetch the base flag, and the aggregation fetches
    what it needs. Only a model spec on a node base always gets the base
    model as values.

    Args:
        parent_dataset: The platform dataset.
        **kwargs: Takes the interpreter arguments of any platform; not used.
            The platform dataset rejects unknown names.
    """

    def __init__(self, parent_dataset: Dataset | None = None, **kwargs: Any):
        super().__init__(parent_dataset=parent_dataset)

    @property
    def accepted_flags(self) -> set[str]:
        """One flag per matching layer and variant value of each spec in the flag index of the dataset."""
        index = self.flag_index
        specs = (index.get_spec_by_name(spec.name) for spec in self.flag_specs)
        return {flag for spec in specs if spec is not None for flag in index.static_flags_of(spec) or ()}

    def flag_is_accepted(self, flag: Any) -> bool:
        """`True` if a spec of this interpreter declares the flag on a layer of the dataset."""
        try:
            spec = self.flag_index.get_spec(flag)
        except UnknownFlagError:
            return False
        return any(spec.name == own.name for own in self.flag_specs)

    def _is_available(self, flag: Any) -> Availability | None:
        """Availability of a layer flag: the combined availability of the flags that its fetch reads.

        The inputs are the base flag, the model flags that the interpreter
        fetches (`_model_inputs`) and the `input_flags` of the aggregation.
        `Availability.of_inputs` combines them. Unavailable if the fetch
        needs a border model that the flag index does not have. `None`
        (unknown) without a parent dataset.
        """
        parent = self.parent_dataset
        if parent is None:
            return None
        index = self.flag_index
        spec, layer = index.get_spec(flag), index.get_layer(flag)
        model_inputs = self._model_inputs(spec, layer)
        if model_inputs is None:
            return Availability(False, f'no border model for layer {layer.name!r}')
        variant = index.get_layer_variant(flag)
        inputs = (index.get_base_flag(flag), *model_inputs, *spec.aggregation.input_flags(parent, layer, variant))
        return Availability.of_inputs({f: parent.availability(f) for f in dict.fromkeys(inputs)})

    @abstractmethod
    def _model_inputs(self, spec: LayerFlagSpec, layer: AreaLayer | BorderLayer) -> tuple[str, ...] | None:
        """The model flags that the fetch reads besides the base flag. `None` if a border model it needs is missing."""

    def _border_model_inputs(self, border_layer: BorderLayer | None) -> tuple[str, ...] | None:
        model_flag = None if border_layer is None else self.flag_index.get_layer_model_flag(border_layer)
        return None if model_flag is None else (model_flag,)

    def _fetch(self, flag: Any, effective_config: DatasetConfig, **params) -> pd.DataFrame:
        index = self.flag_index
        spec = index.get_spec(flag)
        if not isinstance(spec, LayerFlagSpec):
            raise ValueError(f'{type(self).__name__}: {flag!r} is not a flag of a layer spec.')
        return self._aggregate(spec, index.get_layer(flag), index.get_layer_variant(flag), effective_config, params)

    @abstractmethod
    def _aggregate(
        self,
        spec: LayerFlagSpec,
        layer: AreaLayer | BorderLayer,
        variant: LayerVariantValues,
        effective_config: DatasetConfig,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Fetch the inputs of the layer flag with `effective_config` and call the aggregation of `spec`."""

    def _base_values(
        self, spec: LayerFlagSpec, variant: LayerVariantValues, effective_config: DatasetConfig, params
    ) -> pd.DataFrame | None:
        """The base flag for the variant values, or `None` if the spec consumes base dimensions."""
        if spec.consumes:
            return None
        return LayerAggregation.fetch(self.parent_dataset, variant.base_flag(), effective_config, params)

    def _aggregate_on_border_model(
        self,
        spec: LayerFlagSpec,
        layer: AreaLayer | BorderLayer,
        variant: LayerVariantValues,
        effective_config: DatasetConfig,
        params: dict[str, Any],
        border_layer: BorderLayer | None,
    ) -> pd.DataFrame:
        """Aggregate the base values with the border model of `border_layer` (`None`: no border model)."""
        border_model = None if border_layer is None else self._layer_model(border_layer, effective_config)
        values = self._base_values(spec, variant, effective_config, params)
        return spec.aggregation.aggregate(
            values, border_model, self.parent_dataset, layer, variant, effective_config, **params
        )

    def _layer_model(self, layer: AreaLayer | BorderLayer, effective_config: DatasetConfig) -> pd.DataFrame:
        model_flag = self.flag_index.get_layer_model_flag(layer)
        if model_flag is None:
            raise ValueError(
                f'Layer {layer.name!r} has no model flag. Declare a BorderFlagSpec with BorderModel() on the edge model.'
            )
        return self.parent_dataset.fetch(model_flag, effective_config)


class AreaFlagInterpreter(LayerFlagInterpreter):
    """Answers the flags of `AreaFlagSpec`s.

    On a node or node-connected base, it fetches the base flag and the base
    model, takes the mapping from the layer membership column, leaves out
    the objects without area, and calls the aggregation. The aggregation
    maps one object level: each column of the base is one object. A flag
    with several column levels from node values comes from an interpreter
    on the layer that builds it from its inputs, not from an aggregation of
    an object flag.

    On a border base, it fetches the border model of the matching border
    layer as the mapping. The aggregation may build several column levels,
    for example a trade balance per area and partner.
    """

    def _aggregate(self, spec: AreaFlagSpec, layer, variant, effective_config, params) -> pd.DataFrame:
        if spec._has_layer_base:
            border_layer = self.flag_index.get_base_layer(spec, layer)
            return self._aggregate_on_border_model(spec, layer, variant, effective_config, params, border_layer)
        model = LayerAggregation.fetch(self.parent_dataset, layer.base_model, effective_config, params)
        mapping = self._mapping(model, layer)
        values = model if spec.is_model else self._base_values(spec, variant, effective_config, params)
        if values is not None:
            values = (
                values.loc[values.index.intersection(mapping.index, sort=False)]
                if spec.is_model
                else (values[values.columns.intersection(mapping.index, sort=False)])
            )
        return spec.aggregation.aggregate(
            values, mapping, self.parent_dataset, layer, variant, effective_config, **params
        )

    def _model_inputs(self, spec: AreaFlagSpec, layer: AreaLayer) -> tuple[str, ...] | None:
        """The base model of the layer, or for a border base the border model of the matching border layer."""
        if spec._has_layer_base:
            return self._border_model_inputs(self.flag_index.get_base_layer(spec, layer))
        return (layer.base_model,)

    def _mapping(self, model: pd.DataFrame, layer: AreaLayer) -> pd.Series:
        areas = model[self.flag_index.get_mapping_column(layer)]
        return areas[AreaMembership.has_area(areas)]


class BorderFlagInterpreter(LayerFlagInterpreter):
    """Answers the flags of `BorderFlagSpec`s.

    It fetches the base flag (with the area layer of the border layer for an
    area base) and the border model of the layer, and calls the aggregation.
    For the border model flag itself, it passes the edge model as values and
    no border model.
    """

    def _aggregate(self, spec: BorderFlagSpec, layer, variant, effective_config, params) -> pd.DataFrame:
        border_layer = None if spec.is_model else layer
        return self._aggregate_on_border_model(spec, layer, variant, effective_config, params, border_layer)

    def _model_inputs(self, spec: BorderFlagSpec, layer: BorderLayer) -> tuple[str, ...] | None:
        """The border model of the layer. Empty for the border model flag itself."""
        return () if spec.is_model else self._border_model_inputs(layer)
