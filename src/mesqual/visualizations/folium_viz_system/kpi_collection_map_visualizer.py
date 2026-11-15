import logging
from typing import Any, TYPE_CHECKING, List, Literal, Union

import folium

from mesqual.kpis import KPICollection, KPI
from mesqual.visualizations.folium_viz_system.base_viz_system import (
    FoliumObjectGenerator,
    PropertyMapper,
    _DefaultTableMapper,
    _escape_text,
)
from mesqual.visualizations.folium_viz_system.visualizable_data_item import KPIDataItem, VisualizableDataItem
from mesqual.units import Units, QuantityToTextConverter

if TYPE_CHECKING:
    from mesqual.study_manager import StudyManager

logger = logging.getLogger(__name__)

SHOW_OPTIONS = Literal['first', 'last', 'none']
VALUE_FORMATTING_OPTIONS = Union[Literal['per_feature_group', 'per_collection'], QuantityToTextConverter, None]


class KPIQuantityToTextConverterResolver:
    """Resolves the value converter of the KPIs of a feature group.

    The strategies:

    - `'per_feature_group'`: one converter per feature group, for all KPIs of the group.
    - `'per_collection'`: one converter per base unit, for the whole collection.
    - A `QuantityToTextConverter`: this converter for all KPIs.
    - `None`: a `QuantityToTextConverter` with default settings.

    The resolver caches the converters per feature group.

    Args:
        strategy: Formatting strategy.
        kpi_collection: The whole KPI collection, for `'per_collection'`.

    Examples:
        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
        >>> study.scen.add_kpis_from_definitions_to_all_child_datasets(
        ...     FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
        ... )
        >>> prices = study.scen.get_merged_kpi_collection()
        >>> resolver = KPIQuantityToTextConverterResolver('per_feature_group', prices)
        >>> converter = resolver.get_converter_for_kpi('Prices', prices)
        >>> [converter.convert(kpi.quantity) for kpi in prices][:4]
        ['42 €/MWh', '46 €/MWh', '53 €/MWh', '59 €/MWh']
    """

    def __init__(self, strategy: VALUE_FORMATTING_OPTIONS, kpi_collection: KPICollection):
        self.strategy = strategy
        self.kpi_collection = kpi_collection
        self._group_cache = {}

    def get_converter_for_kpi(self, group_id: str, kpi_group: KPICollection) -> QuantityToTextConverter:
        """
        Get converter for a specific KPI, using group-level caching.

        Args:
            group_id: Unique identifier for the feature group (used for caching)
            kpi_group: Full KPI group for building converters

        Returns:
            QuantityToTextConverter configured for this KPI
        """
        if self.strategy is None:
            return QuantityToTextConverter()

        if isinstance(self.strategy, QuantityToTextConverter):
            return self.strategy

        if group_id not in self._group_cache:
            self._group_cache[group_id] = self._build_converters_for_group(kpi_group)

        group_converters = self._group_cache[group_id]

        if isinstance(group_converters, dict):
            base_unit = str(Units.get_base_unit_for_unit(kpi_group._kpis[0].quantity.units))
            return group_converters[base_unit]
        else:
            return group_converters

    def _build_converters_for_group(
        self, kpi_group: KPICollection
    ) -> QuantityToTextConverter | dict[str, QuantityToTextConverter]:
        """
        Build converter(s) for a feature group.

        Returns either a single converter (per_feature_group) or a dict of
        converters keyed by base unit string (per_collection).
        """
        if self.strategy == 'per_feature_group':
            quantities = [kpi.quantity for kpi in kpi_group]
            return QuantityToTextConverter.from_quantities(quantities)

        # per_collection: group by base unit
        by_base_unit = {}
        for kpi in self.kpi_collection:
            base_unit = str(Units.get_base_unit_for_unit(kpi.quantity.units))
            if base_unit not in by_base_unit:
                by_base_unit[base_unit] = []
            by_base_unit[base_unit].append(kpi.quantity)

        return {
            base_unit: QuantityToTextConverter.from_quantities(quantities)
            for base_unit, quantities in by_base_unit.items()
        }


class KPIGroupingManager:
    """
    Groups KPIs into the feature groups of a map.

    KPIs with the same attribute values go into one group, apart from the
    attributes in `kpi_attribute_keys_to_exclude_from_grouping`, such as the
    object name. The groups are sorted by the attributes in
    `kpi_attribute_sort_order` and by the value order in
    `kpi_attribute_category_orders`. The manager also names the groups and finds
    the related KPIs of an object for the tooltips.

    Args:
        kpi_attribute_category_orders: Custom ordering for specific attribute values
        kpi_attribute_keys_to_exclude_from_grouping: Attributes to ignore during grouping
        kpi_attribute_sort_order: Order of attributes for group sorting

    Examples:
        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
        >>> study.scen.add_kpis_from_definitions_to_all_child_datasets(
        ...     FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
        ... )
        >>> prices = study.scen.get_merged_kpi_collection()
        >>> manager = KPIGroupingManager(
        ...     kpi_attribute_category_orders={'dataset_name': ['short', 'base', 'high_demand', 'without_N4']},
        ... )
        >>> groups = manager.get_kpi_groups(prices)
        >>> [len(group) for group in groups]
        [2, 2, 2, 2]
        >>> manager.get_feature_group_name(groups[0])
        'Mean Country.Results.price short EUR_per_MWh   2024-03-28 2024-03-31 4 1.0 1'

        The name joins the values of all attributes that the KPIs of the group
        have in common, including the dataset attributes (here the dates, days,
        demand factor and periods of the dataset) and empty values.
    """

    DEFAULT_EXCLUDE_FROM_GROUPING = [
        'name',
        'object_name',
        'column_subset',
        'custom_name',
        'name_prefix',
        'name_suffix',
        'unit',
        'target_unit',
    ]
    DEFAULT_SORT_ORDER = [
        'name_prefix',
        'model_flag',
        'flag',
        'aggregation',
        'reference_dataset_name',
        'variation_dataset_name',
        'dataset_name',
        'value_comparison',
        'arithmetic_operation',
        'name_suffix',
    ]
    DEFAULT_INCLUDE_ATTRIBUTES = ['arithmetic_operation', 'aggregation', 'flag', 'dataset_name', 'unit']
    DEFAULT_EXCLUDE_ATTRIBUTES = [
        'variation_dataset_name',
        'reference_dataset_name',
        'model_flag',
        'dataset_type',
        'target_unit',
        'dataset_attributes',
    ]

    def __init__(
        self,
        kpi_attribute_category_orders: dict[str, list[str]] = None,
        kpi_attribute_keys_to_exclude_from_grouping: list[str] = None,
        kpi_attribute_sort_order: list[str] = None,
    ):
        self.kpi_attribute_category_orders = kpi_attribute_category_orders or {}
        self.kpi_attribute_keys_to_exclude_from_grouping = (
            kpi_attribute_keys_to_exclude_from_grouping or self.DEFAULT_EXCLUDE_FROM_GROUPING.copy()
        )
        self.kpi_attribute_sort_order = kpi_attribute_sort_order or self.DEFAULT_SORT_ORDER.copy()

    def get_kpi_groups(self, kpi_collection: KPICollection) -> list[KPICollection]:
        """
        Group KPIs by their attributes, in the configured sort order.

        Creates logical groups of KPIs based on their attributes, excluding
        specified attributes from grouping and applying custom sort orders.

        Args:
            kpi_collection: Collection of KPIs to group

        Returns:
            List of KPICollection objects, each representing a logical group
        """
        attribute_sets = kpi_collection.get_all_kpi_attributes_and_value_sets(primitive_values=True)
        relevant_keys = [k for k in attribute_sets.keys() if k not in self.kpi_attribute_keys_to_exclude_from_grouping]

        grouping_keys = [k for k in self.kpi_attribute_sort_order if k in relevant_keys] + [
            k for k in relevant_keys if k not in self.kpi_attribute_sort_order
        ]

        # Build attribute value rankings (used for sorting groups)
        attribute_value_rank: dict[str, dict] = {}
        for attr in grouping_keys:
            existing_values = list(attribute_sets.get(attr, set()))
            manual_order = [v for v in self.kpi_attribute_category_orders.get(attr, []) if v in existing_values]
            remaining = [v for v in existing_values if v not in manual_order]
            try:
                remaining = sorted(remaining)
            except TypeError:
                pass
            full_order = manual_order + remaining
            attribute_value_rank[attr] = {val: idx for idx, val in enumerate(full_order)}

        # Bucket each KPI by its actual attribute tuple. KPIs missing an attribute
        # land in the None bucket for that key instead of being filtered out of
        # every group. This is required when an attribute is set on some scenarios but
        # not others (e.g. heterogeneous dataset_attributes across a study).
        groups_by_key: dict[tuple, list] = {}
        for kpi in kpi_collection:
            attrs = kpi.attributes.as_dict(primitive_values=True)
            key = tuple(attrs.get(k) for k in grouping_keys)
            groups_by_key.setdefault(key, []).append(kpi)

        def sorting_index(key: tuple) -> tuple:
            return tuple(attribute_value_rank[attr].get(val, float('inf')) for attr, val in zip(grouping_keys, key))

        sorted_keys = sorted(groups_by_key.keys(), key=sorting_index)
        return [KPICollection(groups_by_key[k]) for k in sorted_keys]

    def get_feature_group_name(self, kpi_group: KPICollection) -> str:
        """
        Return the feature group name of a KPI group.

        The name joins, with spaces, the values of the attributes that all KPIs
        of the group have in common: first `arithmetic_operation`,
        `aggregation`, `flag`, `dataset_name` and `unit`, then the other
        attributes in their order. The attributes of `DEFAULT_EXCLUDE_ATTRIBUTES`
        and the values None are left out.

        Args:
            kpi_group: KPI group to generate name for

        Returns:
            Human-readable feature group name
        """
        attributes = kpi_group.get_in_common_kpi_attributes(primitive_values=True)

        for k in self.DEFAULT_EXCLUDE_ATTRIBUTES:
            attributes.pop(k, None)

        components = []
        include_attrs = self.DEFAULT_INCLUDE_ATTRIBUTES + [
            k for k in attributes.keys() if k not in self.DEFAULT_INCLUDE_ATTRIBUTES
        ]
        for k in include_attrs:
            value = attributes.pop(k, None)
            if value is not None:
                components.append(str(value))

        return ' '.join(components)

    def get_related_kpi_groups(self, kpi: KPI, study_manager: 'StudyManager | None') -> dict[str, KPICollection]:
        """
        Get related KPIs grouped by relationship type.

        The candidates are the KPIs of the merged KPI collection of
        `study_manager.scen_comp` with the same object, flag and model flag as
        `kpi`. Each candidate goes into the first group it matches:

        - Different Aggregations: same dataset, value comparison and arithmetic
          operation; another aggregation.
        - Different Datasets: same aggregation, value comparison and arithmetic
          operation; another dataset.
        - Different Comparisons / Operations: same dataset and aggregation;
          another value comparison or arithmetic operation.

        Without `study_manager`, or if `kpi` has no flag or aggregation, all
        groups are empty.

        Args:
            kpi: Source KPI to find relatives for
            study_manager: StudyManager for accessing merged KPI collection

        Returns:
            Dict mapping relationship type to KPICollection of related KPIs
        """
        groups = {
            'Different Comparisons / Operations': KPICollection(),
            'Different Aggregations': KPICollection(),
            'Different Datasets': KPICollection(),
        }

        if not study_manager:
            return groups

        # Get reference KPI attributes
        object_name = kpi.attributes.object_name
        flag = kpi.attributes.flag
        model_flag = kpi.attributes.model_flag
        aggregation = kpi.attributes.aggregation
        dataset_name = kpi.attributes.dataset_name
        value_comparison = kpi.attributes.value_comparison
        arithmetic_operation = kpi.attributes.arithmetic_operation

        if not flag or aggregation is None:
            return groups

        try:
            # Get all KPIs for same object/flag/model_flag
            all_related = study_manager.scen_comp.get_merged_kpi_collection()
            pre_filtered = all_related.filter(object_name=object_name, flag=flag, model_flag=model_flag)
        except:
            return groups

        # Determine if main KPI has comparison/operation
        main_has_comparison = value_comparison is not None or arithmetic_operation is not None

        for potential in pre_filtered:
            # Skip self
            if potential is kpi:
                continue

            pot_agg = potential.attributes.aggregation
            pot_dataset = potential.attributes.dataset_name
            pot_comparison = potential.attributes.value_comparison
            pot_operation = potential.attributes.arithmetic_operation
            pot_has_comparison = pot_comparison is not None or pot_operation is not None

            # Category 1: Different Aggregations
            # Same dataset, same comparison status, different aggregation
            if (
                pot_dataset == dataset_name
                and pot_agg != aggregation
                and pot_comparison == value_comparison
                and pot_operation == arithmetic_operation
            ):
                groups['Different Aggregations'].add(potential)
                continue

            # Category 2: Different Datasets
            # Same aggregation, same comparison status, different dataset
            if (
                pot_agg == aggregation
                and pot_dataset != dataset_name
                and pot_comparison == value_comparison
                and pot_operation == arithmetic_operation
            ):
                groups['Different Datasets'].add(potential)
                continue

            # Category 3: Different Comparisons/Operations
            # Same dataset, same aggregation, different comparison/operation
            if (
                pot_dataset == dataset_name
                and pot_agg == aggregation
                and (pot_comparison != value_comparison or pot_operation != arithmetic_operation)
            ):
                groups['Different Comparisons / Operations'].add(potential)
                continue

        return groups


class KPICollectionMapVisualizer:
    """
    Draws a KPI collection on a folium map, one feature group per KPI group.

    Groups the KPIs with a `KPIGroupingManager`, creates one feature group per
    group and applies all generators to each KPI, for example areas with text
    overlays, or lines with arrows. The tooltip can list related KPIs of the
    same object.

    Args:
        generators: Single generator or list of generators for visualization
        study_manager: StudyManager for enhanced KPI relationships and tooltips
        include_related_kpis_in_tooltip: Add related KPIs to tooltip display
        kpi_grouping_manager: Custom grouping manager (optional)
        value_formatting: Number format of the KPI values in the text overlays
            and in the default tooltip and popup (`tooltip=True`,
            `popup=True`): `'per_feature_group'`,
            `'per_collection'` or a `QuantityToTextConverter`. `None`: the text
            of each KPI in its own pretty unit.
        **kwargs: Additional arguments passed to data items

    Examples:
        Areas colored by price, with the price as text, one feature group per
        scenario:

        >>> import folium
        >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
        >>> study.scen.add_kpis_from_definitions_to_all_child_datasets(
        ...     FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
        ... )
        >>> prices = study.scen.get_merged_kpi_collection()
        >>> from mesqual.visualizations.folium_viz_system import (
        ...     AreaFeatureResolver,
        ...     AreaGenerator,
        ...     PropertyMapper,
        ...     TextOverlayGenerator,
        ... )
        >>> from mesqual.visualizations.value_mapping_system import SegmentedContinuousColorscale
        >>> colors = SegmentedContinuousColorscale(segments={(30, 60): 'YlOrRd'})
        >>> visualizer = KPICollectionMapVisualizer(
        ...     generators=[
        ...         AreaGenerator(AreaFeatureResolver(fill_color=PropertyMapper.from_kpi_value(colors))),
        ...         TextOverlayGenerator(),
        ...     ],
        ...     study_manager=study,
        ... )
        >>> m = folium.Map(location=[48, 9], zoom_start=6)
        >>> feature_groups = visualizer.generate_and_add_feature_groups_to_map(prices, m, show='first')
        >>> len(feature_groups)
        4
        >>> m.save('prices.html')
    """

    def __init__(
        self,
        generators: FoliumObjectGenerator | List[FoliumObjectGenerator],
        study_manager: 'StudyManager | None' = None,
        include_related_kpis_in_tooltip: bool = False,
        kpi_grouping_manager: KPIGroupingManager = None,
        value_formatting: VALUE_FORMATTING_OPTIONS = None,
        **kwargs: Any,
    ):
        self.generators: List[FoliumObjectGenerator] = generators if isinstance(generators, list) else [generators]
        self.study_manager = study_manager
        self.include_related_kpis_in_tooltip = include_related_kpis_in_tooltip
        self.value_formatting = value_formatting

        self.grouping_manager = kpi_grouping_manager or KPIGroupingManager()
        self.kwargs = kwargs

    def generate_and_add_feature_groups_to_map(
        self,
        kpi_collection: KPICollection,
        folium_map: folium.Map,
        show: SHOW_OPTIONS = 'none',
        overlay: bool = False,
    ) -> list[folium.FeatureGroup]:
        """Generate feature groups and add them to map."""
        fgs = self.get_feature_groups(kpi_collection, show=show, overlay=overlay)
        for fg in fgs:
            folium_map.add_child(fg)
        return fgs

    def get_feature_groups(
        self, kpi_collection: KPICollection, show: SHOW_OPTIONS = 'none', overlay: bool = False
    ) -> list[folium.FeatureGroup]:
        """
        Create one feature group per KPI group, drawn with all generators.

        The KPI groups come from the grouping manager. A generator that raises
        an error for a KPI is skipped for that KPI, and a warning is logged.

        Args:
            kpi_collection: Collection of KPIs to visualize
            show: Which feature groups to show initially ('first', 'last', 'none')
            overlay: Whether feature groups should be overlay controls

        Returns:
            List of folium FeatureGroup objects ready to add to map
        """
        from tqdm import tqdm

        feature_groups = []
        failed = []

        kpi_to_text_converter_resolver = KPIQuantityToTextConverterResolver(self.value_formatting, kpi_collection)

        pbar = tqdm(kpi_collection, total=kpi_collection.size, desc=f'{self.__class__.__name__}')
        with pbar:
            kpi_groups = self.grouping_manager.get_kpi_groups(kpi_collection)

            for kpi_group in kpi_groups:
                group_name = self.grouping_manager.get_feature_group_name(kpi_group)

                if show == 'first':
                    show_fg = kpi_group == kpi_groups[0]
                elif show == 'last':
                    show_fg = kpi_group == kpi_groups[-1]
                else:
                    show_fg = False

                fg = folium.FeatureGroup(name=group_name, overlay=overlay, show=show_fg)
                converter = kpi_to_text_converter_resolver.get_converter_for_kpi(group_name, kpi_group)

                _generators = [
                    self._create_generator_with_updated_converters_if_needed(converter, generator)
                    for generator in self.generators
                ]

                for kpi in kpi_group:
                    data_item = KPIDataItem(kpi, kpi_collection, study_manager=self.study_manager, **self.kwargs)
                    for gen in _generators:
                        try:
                            gen.generate(data_item, fg)
                        except Exception as e:
                            failed.append((kpi.name, group_name, e))

                    pbar.update(1)

                feature_groups.append(fg)

        if failed:
            logger.warning(f'Exception while trying to add {len(failed)} KPIs: {failed[:3]}')
        return feature_groups

    def _create_generator_with_updated_converters_if_needed(self, converter, generator):
        """Create generator copy with converter-specific mappers if needed."""
        import copy

        default_tables = (
            []
            if self.value_formatting is None
            else [
                key
                for key in ('tooltip', 'popup')
                if isinstance(generator.feature_resolver.property_mappers.get(key), _DefaultTableMapper)
            ]
        )
        needs_modification = (
            self.include_related_kpis_in_tooltip
            or 'text_print_content' in generator.feature_resolver.property_mappers
            or default_tables
        )

        if not needs_modification:
            return generator

        # Create shallow copies
        gen = copy.copy(generator)
        gen.feature_resolver = copy.copy(generator.feature_resolver)
        gen.feature_resolver.property_mappers = generator.feature_resolver.property_mappers.copy()

        for key in default_tables:
            gen.feature_resolver.property_mappers[key] = type(generator.feature_resolver.property_mappers[key])(
                converter
            )
        if self.include_related_kpis_in_tooltip:
            gen.feature_resolver.property_mappers['tooltip'] = self._create_enhanced_tooltip_generator(converter)

        if (self.value_formatting is not None) and ('text_print_content' in gen.feature_resolver.property_mappers):
            gen.feature_resolver.property_mappers['text_print_content'] = self._create_text_print_generator(converter)

        return gen

    def _create_text_print_generator(self, converter: QuantityToTextConverter) -> PropertyMapper:
        """
        Create text print generator with fixed converter.

        Args:
            converter: QuantityToTextConverter to use for formatting

        Returns:
            PropertyMapper with converter baked into the closure
        """

        def get_text(data_item: VisualizableDataItem) -> str:
            return data_item.get_text_representation(converter)

        return PropertyMapper(get_text)

    def _create_enhanced_tooltip_generator(self, converter: QuantityToTextConverter) -> PropertyMapper:
        """
        Create tooltip generator that includes related KPIs with fixed converter.

        Args:
            converter: QuantityToTextConverter to use for formatting all values

        Returns:
            PropertyMapper with converter baked into the closure
        """

        def generate_tooltip(data_item: KPIDataItem) -> str:
            kpi = data_item.kpi
            kpi_name = _escape_text(kpi.get_kpi_name_with_dataset_name())
            kpi_text = _escape_text(converter.convert(kpi.quantity))

            html = '<table style="border-collapse: collapse;">\n'
            html += (
                f'  <tr><td style="padding: 4px 8px;"><strong>{kpi_name}</strong></td>'
                f'<td style="text-align: right; padding: 4px 8px;">{kpi_text}</td></tr>\n'
            )

            if self.include_related_kpis_in_tooltip and self.study_manager:
                related_groups = self.grouping_manager.get_related_kpi_groups(kpi, self.study_manager)

                if any(not g.empty for g in related_groups.values()):
                    for name, group in related_groups.items():
                        if group.empty:
                            continue
                        html += '<tr><p>&nbsp;</p></tr>'
                        html += f'  <tr><th colspan="2" style="text-align: left; padding: 8px;">{_escape_text(name)}</th></tr>\n'
                        for related_kpi in group:
                            related_kpi_name = _escape_text(related_kpi.get_kpi_name_with_dataset_name())
                            related_kpi_value_text = _escape_text(converter.convert(related_kpi.quantity))
                            html += (
                                f'  <tr><td style="padding: 4px 8px;">{related_kpi_name}</td>'
                                f'<td style="text-align: right; padding: 4px 8px;">{related_kpi_value_text}</td></tr>\n'
                            )

            html += '<br><p>&nbsp;</p></table>'
            return html

        return PropertyMapper(generate_tooltip)
