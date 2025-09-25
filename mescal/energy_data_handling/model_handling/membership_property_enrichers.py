from enum import Enum

import pandas as pd

from mescal.datasets import Dataset
from mescal.utils.multi_key_utils.common_base_key_finder import CommonBaseKeyFinder
from mescal.utils.logging import get_logger

logger = get_logger(__name__)

# TODO: Clean up modules and create common base class


class MembershipTagging(Enum):
    """
    Controls how enriched property names are tagged when added to target DataFrames.
    
    In energy system modeling, objects often have relationships to other model components
    (e.g., generators belong to nodes, lines connect nodes). When enriching a DataFrame
    with properties from related objects, this enum controls naming conventions to avoid
    column name conflicts and maintain clarity about property origins.
    
    Values:
        - NONE: Property names remain unchanged (may cause conflicts with existing columns)
        - PREFIX: Property names get membership name as prefix (e.g., 'node_voltage' from 'voltage')
        - SUFFIX: Property names get membership name as suffix (e.g., 'voltage_node' from 'voltage')
    
    Examples:
        For a generator DataFrame (target_df) with 'node' membership, enriching with node properties:
        >>> MembershipPropertyEnricher().append_properties(target_df, dataset, MembershipTagging.NONE)
        >>> # Returns target_df with new columns ['voltage', 'load'] (original names from node DataFrame)
        >>>
        >>> MembershipPropertyEnricher().append_properties(target_df, dataset, MembershipTagging.PREFIX)
        >>> # Returns target_df with new columns ['node_voltage', 'node_load']
        >>>
        >>> MembershipPropertyEnricher().append_properties(target_df, dataset, MembershipTagging.SUFFIX)
        >>> # Returns target_df with new columns ['voltage_node', 'load_node']
    """
    NONE = "none"
    PREFIX = "prefix"
    SUFFIX = "suffix"


class MembershipPropertyEnricher:
    """
    Enriches energy system DataFrames with properties from related model objects.
    
    In energy system modeling, entities often have membership relationships to other
    model components. For example:
    - Generators belong to nodes and have fuel types
    - Lines connect between nodes
    - Storage units are located at nodes and have technology types
    
    This enricher automatically identifies membership columns in a target DataFrame
    and adds all properties from the corresponding model DataFrames. This enables
    comprehensive analysis by combining object properties with their relationships.
    
    Key Features:
    - Automatic identification of membership columns using MESCAL's flag index system
    - Support for multiple simultaneous memberships (node, fuel_type, company, etc.)
    - Preservation of NaN memberships in enriched data
    - Configurable property naming to avoid column conflicts
    - Integration with MESCAL Dataset architecture

    Args:
        membership_tag_separator: Separator used between membership name and property
                                  when PREFIX or SUFFIX tagging is applied
    
    Examples:
        >>> enricher = MembershipPropertyEnricher()
        >>> # Generator DataFrame with 'node' column linking to node objects
        >>> enriched_gen_df = enricher.append_properties(
        ...     generator_df, dataset, MembershipTagging.PREFIX
        ... )
        >>> # enriched_gen_df now includes 'node_voltage', 'node_area' columns from node properties
    """
    def __init__(self, membership_tag_separator: str = '_'):
        """
        Initialize the membership property enricher.
        
        Args:
            membership_tag_separator: Character(s) used to separate membership names
                                     from property names in PREFIX/SUFFIX modes
        """
        self._membership_tag_separator = membership_tag_separator

    def identify_membership_columns(self, column_names: list[str], dataset: Dataset) -> list[str]:
        """
        Identifies columns that represent memberships to other model objects.
        
        Uses MESCAL's flag index system to determine which columns in the target
        DataFrame represent relationships to other model components. This enables
        automatic discovery of enrichment opportunities without manual specification.
        
        Args:
            column_names: List of column names from the target DataFrame
            dataset: MESCAL Dataset containing model definitions and flag mappings
        
        Returns:
            List of column names that represent memberships to other model objects
            
        Examples:
            For a generator DataFrame with columns ['name', 'capacity', 'node', 'fuel_type']:
            >>> membership_cols = enricher.identify_membership_columns(
            ...     generator_df.columns, dataset
            ... )
            >>> print(membership_cols)  # ['node', 'fuel_type']
        """
        return [
            col for col in column_names
            if dataset.flag_index.column_name_in_model_describes_membership(col)
        ]

    def append_properties(
            self,
            target_df: pd.DataFrame,
            dataset: Dataset,
            membership_tagging: MembershipTagging = MembershipTagging.NONE
    ) -> pd.DataFrame:
        """
        Enriches target DataFrame with properties from all linked model objects.
        
        Performs comprehensive enrichment by automatically identifying all membership
        relationships and adding corresponding properties. This is the primary method
        for energy system DataFrame enrichment, enabling complex multi-dimensional
        analysis by combining object properties with their relationships.
        
        The method preserves all original data while adding new property columns.
        Missing relationships (NaN memberships) are handled gracefully by preserving
        NaN values in the enriched properties.
        
        Args:
            target_df: DataFrame to enrich (e.g., generator, line, storage data)
            dataset: MESCAL Dataset containing linked model DataFrames with properties
            membership_tagging: Strategy for naming enriched properties to avoid conflicts
        
        Returns:
            Enhanced DataFrame with all properties from linked model objects added.
            Original columns are preserved, new columns added based on memberships.
            
        Raises:
            Warning: Logged when membership objects are missing from source DataFrames
            
        Examples:
            Energy system use cases:
            
            >>> # Enrich generator data with node and fuel properties
            >>> enriched_generators = enricher.append_properties(
            ...     generators_df, dataset, MembershipTagging.PREFIX
            ... )
            >>> # Result: original columns + 'node_voltage', 'node_area', 'fuel_co2_rate', etc.
            
            >>> # Enrich transmission data with node characteristics
            >>> enriched_lines = enricher.append_properties(
            ...     transmission_df, dataset, MembershipTagging.SUFFIX
            ... )
            >>> # Result: line properties + node properties with '_node' suffix
        """
        membership_columns = self.identify_membership_columns(target_df.columns, dataset)
        result_df = target_df.copy()

        for column in membership_columns:
            result_df = self.append_single_membership_properties(
                result_df,
                dataset,
                column,
                membership_tagging
            )

        return result_df

    def append_single_membership_properties(
            self,
            target_df: pd.DataFrame,
            dataset: Dataset,
            membership_column: str,
            membership_tagging: MembershipTagging = MembershipTagging.NONE
    ) -> pd.DataFrame:
        """
        Enriches target DataFrame with properties from a specific membership relationship.
        
        This method provides fine-grained control over property enrichment by handling
        a single membership column. Useful when custom logic is needed for specific
        relationships or when processing memberships sequentially with different
        tagging strategies.
        
        The method uses MESCAL's flag index to determine the source model DataFrame
        for the membership column, then performs a left join to preserve all target
        records while adding available properties.
        
        Args:
            target_df: DataFrame to enrich (must contain the membership column)
            dataset: MESCAL Dataset with access to linked model DataFrames
            membership_column: Name of column containing object references (e.g., 'node', 'fuel_type')
            membership_tagging: Strategy for naming enriched properties
        
        Returns:
            DataFrame with properties from the linked model objects added.
            All original rows preserved; NaN memberships result in NaN properties.
            
        Raises:
            Warning: Logged when referenced objects are missing from the source DataFrame
            
        Examples:
            Targeted enrichment scenarios:
            
            >>> # Add only node properties to generators
            >>> gen_with_nodes = enricher.append_single_membership_properties(
            ...     generators_df, dataset, 'node', MembershipTagging.PREFIX
            ... )
            >>> # Result: generators + 'node_voltage', 'node_area', etc.
            
            >>> # Sequential enrichment with different tagging
            >>> result = generators_df.copy()
            >>> result = enricher.append_single_membership_properties(
            ...     result, dataset, 'node', MembershipTagging.PREFIX
            ... )
            >>> result = enricher.append_single_membership_properties(
            ...     result, dataset, 'fuel_type', MembershipTagging.SUFFIX
            ... )
        """
        source_flag = dataset.flag_index.get_linked_model_flag_for_membership_column(membership_column)
        source_df = dataset.fetch(source_flag)

        membership_objects = target_df[membership_column].dropna().unique()
        missing_objects = set(membership_objects) - set(source_df.index)

        if missing_objects:
            self._log_missing_objects_warning(missing_objects, membership_column)

        source_properties = source_df.copy()

        match membership_tagging:
            case MembershipTagging.PREFIX:
                source_properties = source_properties.add_prefix(f"{membership_column}{self._membership_tag_separator}")
            case MembershipTagging.SUFFIX:
                source_properties = source_properties.add_suffix(f"{self._membership_tag_separator}{membership_column}")

        result_df = target_df.merge(
            source_properties,
            left_on=membership_column,
            right_index=True,
            how="left"
        )

        return result_df

    def _log_missing_objects_warning(
            self,
            missing_objects: set,
            membership_column: str,
            max_show: int = 5
    ):
        """
        Logs warning about missing objects in the source DataFrame.
        
        In energy system modeling, missing references can indicate data quality
        issues, model inconsistencies, or incomplete datasets. This method provides
        informative warnings to help identify and resolve such issues.
        
        Args:
            missing_objects: Set of object identifiers missing from source DataFrame
            membership_column: Name of the membership column being processed
            max_show: Maximum number of missing objects to display in the warning
        """
        num_missing = len(missing_objects)
        warning_suffix = ", and more" if num_missing > max_show else ""
        logger.warning(
            f"{num_missing} objects missing in source dataframe for {membership_column}: "
            f"{list(missing_objects)[:max_show]}{warning_suffix}."
        )


class DirectionalMembershipPropertyEnricher:
    """
    Enriches energy system DataFrames with properties for directional relationships.
    
    Energy networks inherently contain directional relationships - transmission lines
    connect from one node to another, flows have origins and destinations, and trade
    occurs between regions. This enricher handles such bidirectional memberships by
    identifying from/to column pairs and enriching with appropriate directional tags.
    
    Common energy system applications:
    - Transmission lines: 'node_from' and 'node_to' linking to node properties
    - Inter-regional flows: 'region_from' and 'region_to' for trade analysis
    - Pipeline systems: 'hub_from' and 'hub_to' for gas network modeling
    - Market connections: 'market_from' and 'market_to' for price analysis
    
    The enricher automatically identifies directional column pairs using configurable
    identifiers (default: '_from' and '_to') and adds properties from the linked
    model objects with appropriate directional suffixes.
    
    Key Features:
        - Automatic identification of from/to column pairs
        - Flexible directional identifiers (customizable beyond '_from'/'_to')
        - Support for multiple directional relationships in one DataFrame
        - Preservation of NaN relationships
        - Integration with MESCAL's model flag system
    
    Args:
        from_identifier: Suffix/prefix identifying 'from' direction (default: '_from')
        to_identifier: Suffix/prefix identifying 'to' direction (default: '_to')
        membership_tag_separator: Separator for property name construction
    
    Examples:

        >>> enricher = DirectionalMembershipPropertyEnricher()
        >>> # Line DataFrame with 'node_from', 'node_to' columns
        >>> enriched_lines = enricher.append_properties(
        ...     line_df, dataset, MembershipTagging.NONE
        ... )
        >>> # Result includes node properties with '_from' and '_to' suffixes
    """
    def __init__(
            self,
            from_identifier: str = "_from",
            to_identifier: str = "_to",
            membership_tag_separator: str = '_',
    ):
        """
        Initialize the directional membership property enricher.
        
        Args:
            from_identifier: String identifying source/origin columns (e.g., '_from', 'source_')
            to_identifier: String identifying destination/target columns (e.g., '_to', 'dest_')
            membership_tag_separator: Character(s) separating membership names from properties
        """
        self._from_identifier = from_identifier
        self._to_identifier = to_identifier
        self._membership_tag_separator = membership_tag_separator
        self._tag_finder = CommonBaseKeyFinder(from_identifier, to_identifier)

    def identify_from_to_columns(self, column_names: list[str], dataset: Dataset) -> list[str]:
        """
        Identifies base names for directional membership column pairs.
        
        Analyzes column names to find base membership types that have both 'from'
        and 'to' variants. For example, identifies 'node' as a base when both
        'node_from' and 'node_to' columns exist and represent valid memberships.
        
        Args:
            column_names: List of column names from the target DataFrame
            dataset: MESCAL Dataset for membership validation
        
        Returns:
            List of base column names that have both from/to variants
            
        Examples:
            For a line DataFrame with ['name', 'capacity', 'node_from', 'node_to']:
            >>> base_columns = enricher.identify_from_to_columns(
            ...     line_df.columns, dataset
            ... )
            >>> print(base_columns)  # ['node']
            
            For inter-regional trade with ['flow', 'region_from', 'region_to', 'market_from']:
            >>> base_columns = enricher.identify_from_to_columns(
            ...     trade_df.columns, dataset  
            ... )
            >>> print(base_columns)  # ['region'] (market missing 'market_to')
        """
        potential_columns = self._tag_finder.get_keys_for_which_all_association_tags_appear(column_names)
        return [
            col for col in potential_columns
            if dataset.flag_index.column_name_in_model_describes_membership(col)
        ]

    def append_properties(
            self,
            target_df: pd.DataFrame,
            dataset: Dataset,
            membership_tagging: MembershipTagging = MembershipTagging.NONE
    ) -> pd.DataFrame:
        """
        Enriches DataFrame with properties from all directional relationships.
        
        Performs comprehensive directional enrichment by identifying all from/to
        column pairs and adding properties from both directions. Essential for
        network analysis where understanding characteristics of connected nodes,
        regions, or components is crucial for energy system modeling.
        
        Each directional relationship results in two sets of enriched properties:
        one for the 'from' direction and one for the 'to' direction, clearly
        distinguished by directional suffixes.
        
        Args:
            target_df: DataFrame with directional relationships (e.g., transmission lines)
            dataset: MESCAL Dataset containing model objects and their properties
            membership_tagging: Strategy for property naming (applied before directional tags)
        
        Returns:
            Enhanced DataFrame with directional properties added. Original data preserved,
            new columns follow pattern: [prefix_]property_name[_suffix]_direction
            
        Raises:
            Warning: Logged when referenced objects missing from source DataFrames
            
        Examples:
            Network transmission analysis:
            
            >>> # Transmission lines with node endpoints
            >>> enriched_lines = enricher.append_properties(
            ...     transmission_df, dataset, MembershipTagging.NONE
            ... )
            >>> # Result: original columns + 'voltage_from', 'voltage_to', 
            >>> #         'area_from', 'area_to', etc.
            
            >>> # Inter-regional trade flows
            >>> enriched_trade = enricher.append_properties(
            ...     trade_df, dataset, MembershipTagging.PREFIX
            ... )
            >>> # Result: trade data + 'region_gdp_from', 'region_gdp_to', etc.
        """
        membership_base_columns = self.identify_from_to_columns(target_df.columns, dataset)
        result_df = target_df.copy()

        for base_column in membership_base_columns:
            result_df = self.append_directional_properties(
                result_df,
                dataset,
                base_column,
                membership_tagging
            )

        return result_df

    def append_directional_properties(
            self,
            target_df: pd.DataFrame,
            dataset: Dataset,
            base_column: str,
            membership_tagging: MembershipTagging = MembershipTagging.NONE
    ) -> pd.DataFrame:
        source_flag = dataset.flag_index.get_linked_model_flag_for_membership_column(base_column)
        source_df = dataset.fetch(source_flag)
        return self.append_directional_properties_in_source_to_target_df(
            target_df,
            source_df,
            base_column,
            membership_tagging,
        )

    def append_directional_properties_in_source_to_target_df(
            self,
            target_df: pd.DataFrame,
            source_df: pd.DataFrame,
            base_column: str,
            membership_tagging: MembershipTagging = MembershipTagging.NONE
    ) -> pd.DataFrame:
        """
        Enriches DataFrame with properties from a directional relationship.

        Handles a single from/to membership pair by adding the corresponding model
        DataFrame's properties with directional tags.

        The method processes both directions (from/to) for the specified base column,
        adding properties with appropriate directional suffixes. Missing references
        are handled gracefully with NaN preservation.

        Args:
            target_df: DataFrame containing the directional columns
            source_df: DataFrame containing the properties
            base_column: Base membership name (e.g., 'node' for 'node_from'/'node_to')
            membership_tagging: Property naming strategy (applied before directional tags)

        Returns:
            DataFrame with directional properties added for the specified relationship.
            Properties follow naming pattern: [prefix_]property[_suffix]_direction

        Raises:
            Warning: Logged when referenced objects are missing from source DataFrame

        Examples:
            Targeted directional enrichment:

            >>> # Add only node properties to transmission lines
            >>> lines_with_nodes = enricher.append_directional_properties(
            ...     line_df, node_df, 'node', MembershipTagging.NONE
            ... )
            >>> # Result: lines + 'voltage_from', 'voltage_to', 'area_from', 'area_to'
        """
        result_df = target_df.copy()
        for tag in [self._from_identifier, self._to_identifier]:
            membership_column = self._get_full_column_name(base_column, tag, target_df.columns)

            if membership_column not in target_df.columns:
                continue

            membership_objects = target_df[membership_column].dropna().unique()
            missing_objects = set(membership_objects) - set(source_df.index)

            if missing_objects:
                self._log_missing_objects_warning(missing_objects, membership_column)

            source_properties = source_df.copy()

            match membership_tagging:
                case MembershipTagging.PREFIX:
                    source_properties = source_properties.add_prefix(f"{base_column}{self._membership_tag_separator}")
                case MembershipTagging.SUFFIX:
                    source_properties = source_properties.add_suffix(f"{self._membership_tag_separator}{base_column}")

            source_properties = source_properties.add_suffix(tag)

            result_df = result_df.merge(
                source_properties,
                left_on=membership_column,
                right_index=True,
                how="left"
            )
        return result_df

    def _get_full_column_name(self, base_column: str, tag: str, df_columns: list[str]) -> str:
        """
        Determines the actual column name for a directional membership.
        
        Handles flexibility in directional column naming by testing both
        suffix and prefix patterns. Supports various naming conventions
        used across different energy modeling platforms.
        
        Args:
            base_column: Base membership name (e.g., 'node')
            tag: Directional identifier (e.g., '_from', '_to')
            df_columns: List of actual column names in the DataFrame
        
        Returns:
            Actual column name found in the DataFrame
            
        Examples:
            >>> # For base_column='node', tag='_from'
            >>> # Tests 'node_from' first, then 'from_node'
            >>> name = enricher._get_full_column_name('node', '_from', df.columns)
        """
        test_suffix = f"{base_column}{tag}"
        return test_suffix if test_suffix in df_columns else f"{tag}{base_column}"

    def _log_missing_objects_warning(
            self,
            missing_objects: set,
            membership_column: str,
            max_show: int = 5
    ):
        num_missing = len(missing_objects)
        warning_suffix = ", and more" if num_missing > max_show else ""
        logger.warning(
            f"{num_missing} objects missing in source dataframe for {membership_column}: "
            f"{list(missing_objects)[:max_show]}{warning_suffix}."
        )


if __name__ == "__main__":
    node_model_df = pd.DataFrame({
        'voltage_kv': [380, 380, 220, 380, 220, 150, 400, 220, 380, 150],
        'asset_type': ['transmission', 'transmission', 'distribution', 'transmission',
                       'distribution', 'distribution', 'transmission', 'distribution',
                       'transmission', 'distribution'],
        'country': ['DE', 'DE', 'FR', 'FR', 'BE', 'BE', 'NL', 'NL', 'PL', 'PL'],
        'bidding_zone': ['DE_LU', 'DE_LU', 'FR', 'FR', 'BE', 'BE', 'NL', 'NL', 'PL', 'PL'],
        'capacity_mw': [2000, 1500, 800, 1200, 600, 400, 1800, 700, 1600, 500],
        'operator': ['TenneT', '50Hertz', 'RTE', 'RTE', 'Elia', 'Elia', 'TenneT', 'Stedin', 'PSE', 'PSE']
    }, index=['DE_T1', 'DE_T2', 'FR_D1', 'FR_T1', 'BE_D1', 'BE_D2', 'NL_T1', 'NL_D1', 'PL_T1', 'PL_D2'])

    line_model_df = pd.DataFrame({
        'node_from': ['DE_T1', 'DE_T2', 'FR_T1', 'FR_T1', 'BE_D1', 'BE_D2', 'NL_T1', 'DE_T1', 'PL_T1'],
        'node_to': ['FR_T1', 'BE_D1', 'BE_D1', 'DE_T1', 'NL_T1', 'FR_D1', 'DE_T2', 'PL_T1', 'DE_T2'],
        'length_km': [650, 320, 180, 650, 120, 280, 180, 580, 180],
        'technology': ['AC', 'AC', 'AC', 'AC', 'AC', 'AC', 'DC', 'AC', 'AC']
    }, index=['L_DE_FR_1', 'L_DE_BE_1', 'L_FR_BE_1', 'L_FR_DE_1', 'L_BE_NL_1', 'L_BE_FR_1', 'L_NL_DE_1', 'L_DE_PL_1',
              'L_PL_DE_1'])

    enricher = DirectionalMembershipPropertyEnricher(
        from_identifier='_from',
        to_identifier='_to',
        membership_tag_separator='_'
    )
    enriched_line_model_df = enricher.append_directional_properties_in_source_to_target_df(
        line_model_df,
        node_model_df,
        'node',
        MembershipTagging.SUFFIX
    )
    print(enriched_line_model_df)
