from typing import TYPE_CHECKING, List, Literal

import folium

from mescal.kpis import KPICollection, KPI
from mescal.visualizations.folium_viz_system.base_viz_system import FoliumObjectGenerator, PropertyMapper
from mescal.visualizations.folium_viz_system.visualizable_data_item import KPIDataItem, VisualizableDataItem

if TYPE_CHECKING:
    from mescal.study_manager import StudyManager


SHOW_OPTIONS = Literal['first', 'last', 'none']


class KPIGroupingManager:
    """Handles KPI grouping logic extracted from KPIToMapVisualizerBase."""

    DEFAULT_EXCLUDE_FROM_GROUPING = ['name', 'object_name', 'column_subset']
    DEFAULT_SORT_ORDER = [
        'name_prefix', 'model_flag', 'flag', 'model_query', 'aggregation',
        'reference_dataset', 'variation_dataset', 'dataset',
        'value_comparison', 'value_operation', 'name_suffix'
    ]
    DEFAULT_INCLUDE_ATTRIBUTES = ['value_operation', 'aggregation', 'flag', 'dataset', 'unit']
    DEFAULT_EXCLUDE_ATTRIBUTES = ['variation_dataset', 'reference_dataset', 'model_flag', 'base_unit', 'dataset_type']

    def __init__(
            self,
            kpi_attribute_category_orders: dict[str, list[str]] = None,
            kpi_attribute_keys_to_exclude_from_grouping: list[str] = None,
            kpi_attribute_sort_order: list[str] = None
    ):
        self.kpi_attribute_category_orders = kpi_attribute_category_orders or {}
        self.kpi_attribute_keys_to_exclude_from_grouping = (
                kpi_attribute_keys_to_exclude_from_grouping or self.DEFAULT_EXCLUDE_FROM_GROUPING.copy()
        )
        self.kpi_attribute_sort_order = (
                kpi_attribute_sort_order or self.DEFAULT_SORT_ORDER.copy()
        )

    def get_kpi_groups(self, kpi_collection: KPICollection) -> list[KPICollection]:
        """Group KPIs by attributes with sophisticated sorting."""
        from mescal.utils.dict_combinations import dict_combination_iterator

        attribute_sets = kpi_collection.get_all_kpi_attributes_and_value_sets(primitive_values=True)
        relevant_attribute_sets = {
            k: v for k, v in attribute_sets.items()
            if k not in self.kpi_attribute_keys_to_exclude_from_grouping
        }

        ordered_keys = [k for k in self.kpi_attribute_sort_order if k in relevant_attribute_sets]

        # Build attribute value rankings
        attribute_value_rank: dict[str, dict[str, int]] = {}
        for attr in ordered_keys:
            existing_values = set(relevant_attribute_sets.get(attr, []))
            manual_order = [v for v in self.kpi_attribute_category_orders.get(attr, []) if v in existing_values]
            remaining = sorted(existing_values - set(manual_order))
            full_order = manual_order + remaining
            attribute_value_rank[attr] = {val: idx for idx, val in enumerate(full_order)}

        def sorting_index(group_kwargs: dict[str, str]) -> tuple:
            return tuple(
                attribute_value_rank[attr].get(group_kwargs.get(attr), float("inf"))
                for attr in ordered_keys
            )

        # Create and sort groups
        group_kwargs_list = list(dict_combination_iterator(relevant_attribute_sets))
        group_kwargs_list.sort(key=sorting_index)

        groups: list[KPICollection] = []
        for group_kwargs in group_kwargs_list:
            g = kpi_collection.get_filtered_kpi_collection_by_attributes(**group_kwargs)
            if not g.empty:
                groups.append(g)

        return groups

    def get_feature_group_name(self, kpi_group: KPICollection) -> str:
        """Generate meaningful feature group name from KPI group."""
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

    def get_related_kpi_groups(self, kpi: KPI, study_manager) -> dict[str, KPICollection]:
        """Get related KPIs grouped by relationship type."""
        from mescal.kpis import ValueComparisonKPI, ArithmeticValueOperationKPI

        groups = {
            'Different Comparisons / ValueOperations': KPICollection(),
            'Different Aggregations': KPICollection(),
            'Different Datasets': KPICollection(),
        }

        if not study_manager:
            return groups

        kpi_atts = kpi.attributes.as_dict(primitive_values=True)

        _must_contain = ['flag', 'aggregation']
        if any(kpi_atts.get(k, None) is None for k in _must_contain):
            return groups

        try:
            pre_filtered = study_manager.scen_comp.get_merged_kpi_collection()
            pre_filtered = pre_filtered.get_filtered_kpi_collection_by_attributes(
                object_name=kpi.get_attributed_object_name(),
                flag=kpi_atts['flag'],
                model_flag=kpi.get_attributed_model_flag(),
            )
        except:
            return groups

        _main_kpi_is_value_op = isinstance(kpi, (ValueComparisonKPI, ArithmeticValueOperationKPI))

        for potential_relative in pre_filtered:
            pratts = potential_relative.attributes.as_dict(primitive_values=True)
            if pratts.get('dataset') == kpi_atts.get('dataset'):  # same ds
                if pratts.get('aggregation', None) == kpi_atts.get('aggregation'):  # same ds, agg
                    if pratts.get('value_operation', None) != kpi_atts.get('value_operation', None):
                        groups['Different Comparisons / ValueOperations'].add_kpi(potential_relative)
                        continue
                else:  # same ds, diff agg
                    if pratts.get('value_operation', None) is None:
                        groups['Different Aggregations'].add_kpi(potential_relative)
                        continue
                    elif pratts.get('value_operation') == kpi_atts.get('value_operation', None):
                        groups['Different Aggregations'].add_kpi(potential_relative)
                        continue
            elif pratts.get('aggregation', None) == kpi_atts.get('aggregation'):  # same agg, diff ds
                if pratts.get('value_operation', None) == kpi_atts.get('value_operation', None):
                    groups['Different Datasets'].add_kpi(potential_relative)
                    continue
                if not _main_kpi_is_value_op:
                    groups['Different Comparisons / ValueOperations'].add_kpi(potential_relative)
                    continue

        return groups


class KPICollectionMapVisualizer:
    """High-level KPI collection map visualizer that replicates KPIToMapVisualizerBase functionality."""

    def __init__(
            self,
            generators: FoliumObjectGenerator | List[FoliumObjectGenerator],
            study_manager: 'StudyManager' = None,
            include_related_kpis_in_tooltip: bool = False,
            kpi_grouping_manager: KPIGroupingManager = None,
            **kwargs
    ):
        self.generators: List[FoliumObjectGenerator] = generators if isinstance(generators, list) else [generators]
        self.study_manager = study_manager
        self.include_related_kpis_in_tooltip = include_related_kpis_in_tooltip

        self.grouping_manager = kpi_grouping_manager or KPIGroupingManager()

        # Enhanced tooltip if needed
        if self.include_related_kpis_in_tooltip:
            for g in self.generators:
                g.tooltip_generator = self._create_enhanced_tooltip_generator()
        self.kwargs = kwargs

    def generate_and_add_feature_groups_to_map(
            self,
            kpi_collection: KPICollection,
            folium_map: folium.Map,
            show: SHOW_OPTIONS = 'none',
            overlay: bool = False,
    ) -> None:
        fgs = self.get_feature_groups(kpi_collection, show=show, overlay=overlay)
        for fg in fgs:
            folium_map.add_child(fg)

    def get_feature_groups(
            self,
            kpi_collection: KPICollection,
            show: SHOW_OPTIONS = 'none',
            overlay: bool = False
    ) -> list[folium.FeatureGroup]:
        """Create feature groups for KPI collection, replicating original functionality."""
        from tqdm import tqdm
        from mescal.utils.logging import get_logger

        logger = get_logger(__name__)
        feature_groups = []

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

                for kpi in kpi_group:
                    data_item = KPIDataItem(kpi, kpi_collection, study_manager=self.study_manager, **self.kwargs)
                    for generator in self.generators:
                        if self.include_related_kpis_in_tooltip:
                            _tmp = generator.feature_resolver.property_mappers.get('tooltip', None)
                            generator.feature_resolver.property_mappers['tooltip'] = self._create_enhanced_tooltip_generator()
                        try:
                            generator.generate(data_item, fg)
                        except Exception as e:
                            logger.warning(
                                f'Exception while trying to add KPI {kpi.name} to FeatureGroup {group_name}: {e}'
                            )
                        finally:
                            if self.include_related_kpis_in_tooltip:
                                if _tmp is not None:
                                    generator.feature_resolver.property_mappers['tooltip'] = _tmp
                                else:
                                    generator.feature_resolver.property_mappers.pop('tooltip')
                    pbar.update(1)

                feature_groups.append(fg)

        return feature_groups

    def _create_enhanced_tooltip_generator(self) -> PropertyMapper:
        """Create tooltip generator that includes related KPIs."""

        def generate_tooltip(data_item: KPIDataItem) -> str:

            kpi = data_item.kpi
            kpi_name = kpi.get_kpi_name_with_dataset_name()

            from mescal.units import Units
            kpi_quantity = Units.get_quantity_in_pretty_unit(kpi.quantity)
            kpi_text = Units.get_pretty_text_for_quantity(kpi_quantity, thousands_separator=' ')

            html = '<table style="border-collapse: collapse;">\n'
            html += f'  <tr><td style="padding: 4px 8px;"><strong>{kpi_name}</strong></td>' \
                    f'<td style="text-align: right; padding: 4px 8px;">{kpi_text}</td></tr>\n'

            if self.include_related_kpis_in_tooltip and self.study_manager:
                related_groups = self.grouping_manager.get_related_kpi_groups(
                    kpi, self.study_manager
                )

                if any(not g.empty for g in related_groups.values()):
                    for name, group in related_groups.items():
                        if group.empty:
                            continue
                        html += "<tr><p>&nbsp;</p></tr>"
                        html += f'  <tr><th colspan="2" style="text-align: left; padding: 8px;">{name}</th></tr>\n'
                        for related_kpi in group:
                            related_kpi_name = related_kpi.get_kpi_name_with_dataset_name()
                            related_kpi_quantity = Units.get_quantity_in_pretty_unit(related_kpi.quantity)
                            related_kpi_value_text = Units.get_pretty_text_for_quantity(
                                related_kpi_quantity,
                                thousands_separator=' ',
                            )
                            html += f'  <tr><td style="padding: 4px 8px;">{related_kpi_name}</td>' \
                                    f'<td style="text-align: right; padding: 4px 8px;">{related_kpi_value_text}</td></tr>\n'

            html += '<br><p>&nbsp;</p></table>'
            return html

        return PropertyMapper.from_item(generate_tooltip)
