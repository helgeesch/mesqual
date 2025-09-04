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
        NONE: Property names remain unchanged (may cause conflicts with existing columns)
        PREFIX: Property names get membership name as prefix (e.g., 'node_voltage' from 'voltage')
        SUFFIX: Property names get membership name as suffix (e.g., 'voltage_node' from 'voltage')
    
    Examples:
        For a generator DataFrame with 'node' membership, enriching with node properties:
        - NONE: 'voltage', 'load' (original names from node DataFrame)
        - PREFIX: 'node_voltage', 'node_load' 
        - SUFFIX: 'voltage_node', 'load_node'
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
            
            Energy system insights enabled:
            - Voltage level compatibility analysis for transmission lines
            - Regional economic impact assessment for trade flows
            - Technology mix comparison between connected areas
            - Market price differential analysis across connections
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
        """
        Enriches DataFrame with properties from a specific directional relationship.
        
        Handles a single from/to membership pair by fetching the corresponding model
        DataFrame and adding its properties with directional tags. Provides precise
        control over individual directional relationships, useful for custom logic
        or selective enrichment scenarios.
        
        The method processes both directions (from/to) for the specified base column,
        adding properties with appropriate directional suffixes. Missing references
        are handled gracefully with NaN preservation.
        
        Args:
            target_df: DataFrame containing the directional columns
            dataset: MESCAL Dataset with access to the linked model DataFrame
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
            ...     transmission_df, dataset, 'node', MembershipTagging.NONE
            ... )
            >>> # Result: lines + 'voltage_from', 'voltage_to', 'area_from', 'area_to'
            
            >>> # Sequential processing with different strategies
            >>> result = transmission_df.copy()
            >>> # Add node properties with prefix
            >>> result = enricher.append_directional_properties(
            ...     result, dataset, 'node', MembershipTagging.PREFIX
            ... )
            >>> # Add region properties with suffix
            >>> result = enricher.append_directional_properties(
            ...     result, dataset, 'region', MembershipTagging.SUFFIX
            ... )
        """
        source_flag = dataset.flag_index.get_linked_model_flag_for_membership_column(base_column)
        source_df = dataset.fetch(source_flag)
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
    """
    Comprehensive examples demonstrating membership property enrichment for energy systems.
    
    This example section showcases both basic and advanced usage patterns for enriching
    energy system DataFrames with related object properties. Examples cover typical
    energy modeling scenarios including generators, transmission lines, and storage units.
    """
    # TODO: replace with pypsa example
    from mescal_mock import MockDataset
    import numpy as np

    print("=" * 80)
    print("MESCAL Energy Data Handling - Membership Property Enrichment Examples")
    print("=" * 80)

    # Initialize mock dataset and enrichers
    mock_ds = MockDataset()
    enricher = MembershipPropertyEnricher(membership_tag_separator='_')
    directional_enricher = DirectionalMembershipPropertyEnricher()

    # Fetch model DataFrames representing typical energy system objects
    generator_df = mock_ds.fetch('Generator.Model')
    line_df = mock_ds.fetch('Line.Model')
    
    # Simulate realistic data scenarios with missing relationships
    generator_df.loc[generator_df.index[0], 'node'] = pd.NA
    line_df.loc[line_df.index[0], 'node_from'] = pd.NA

    print("\n1. ORIGINAL DATA STRUCTURES")
    print("-" * 40)
    print("Generator DataFrame (partial):")
    print(generator_df.head(3))
    print(f"\nColumns: {list(generator_df.columns)}")
    
    print("\nTransmission Line DataFrame (partial):")
    print(line_df.head(3))
    print(f"\nColumns: {list(line_df.columns)}")

    # Example 1: Automatic enrichment with different tagging strategies
    print("\n\n2. AUTOMATIC MEMBERSHIP ENRICHMENT")
    print("-" * 40)
    
    # Demonstrate different tagging approaches
    tagging_examples = [
        (MembershipTagging.NONE, "No tagging (may cause column conflicts)"),
        (MembershipTagging.PREFIX, "Prefix tagging (membership_property)"),
        (MembershipTagging.SUFFIX, "Suffix tagging (property_membership)")
    ]
    
    for tagging, description in tagging_examples:
        print(f"\nTagging Strategy: {description}")
        enriched_gen = enricher.append_properties(
            generator_df.copy(),
            mock_ds,
            membership_tagging=tagging
        )
        print(f"Enriched columns: {len(enriched_gen.columns)} (was {len(generator_df.columns)})")
        new_columns = set(enriched_gen.columns) - set(generator_df.columns)
        print(f"New properties added: {sorted(list(new_columns))}")

    # Example 2: Manual membership identification and processing
    print("\n\n3. MANUAL MEMBERSHIP PROCESSING")
    print("-" * 40)
    
    membership_cols = enricher.identify_membership_columns(generator_df.columns, mock_ds)
    print(f"Identified membership columns: {membership_cols}")
    
    # Sequential processing with custom logic for each membership
    enriched_generator_df = generator_df.copy()
    for i, col in enumerate(membership_cols):
        print(f"\nProcessing membership '{col}'...")
        before_cols = len(enriched_generator_df.columns)
        
        # Demonstrate custom processing per membership type
        if 'node' in col.lower():
            # Use PREFIX for spatial relationships
            tagging = MembershipTagging.PREFIX
        elif 'fuel' in col.lower() or 'tech' in col.lower():
            # Use SUFFIX for technology/fuel classifications
            tagging = MembershipTagging.SUFFIX
        else:
            # Use NONE for other relationships
            tagging = MembershipTagging.NONE
            
        enriched_generator_df = enricher.append_single_membership_properties(
            enriched_generator_df,
            mock_ds,
            col,
            membership_tagging=tagging
        )
        after_cols = len(enriched_generator_df.columns)
        print(f"  Added {after_cols - before_cols} properties with {tagging.value} tagging")

    print(f"\nFinal enriched generator DataFrame shape: {enriched_generator_df.shape}")

    # Example 3: Directional relationship enrichment
    print("\n\n4. DIRECTIONAL RELATIONSHIP ENRICHMENT")
    print("-" * 40)
    
    # Identify directional relationships
    base_columns = directional_enricher.identify_from_to_columns(line_df.columns, mock_ds)
    print(f"Identified from/to base columns: {base_columns}")
    
    # Automatic directional enrichment
    enriched_line_df = directional_enricher.append_properties(
        line_df.copy(),
        mock_ds,
        membership_tagging=MembershipTagging.NONE
    )
    
    print(f"\nOriginal line DataFrame: {line_df.shape[1]} columns")
    print(f"Enriched line DataFrame: {enriched_line_df.shape[1]} columns")
    
    # Show directional properties
    directional_cols = [col for col in enriched_line_df.columns 
                       if col.endswith('_from') or col.endswith('_to')]
    print(f"Directional properties added: {len(directional_cols)}")
    print(f"Sample directional columns: {directional_cols[:6]}")

    # Example 4: Manual directional processing with custom logic
    print("\n\n5. MANUAL DIRECTIONAL PROCESSING")
    print("-" * 40)
    
    result_line_df = line_df.copy()
    for base_col in base_columns:
        print(f"\nProcessing directional relationship: {base_col}")
        before_cols = len(result_line_df.columns)
        
        # Custom tagging strategy based on relationship type
        if 'node' in base_col.lower():
            tagging = MembershipTagging.NONE  # Clean names for network analysis
        elif 'region' in base_col.lower():
            tagging = MembershipTagging.PREFIX  # Clear regional context
        else:
            tagging = MembershipTagging.SUFFIX  # Default strategy
            
        result_line_df = directional_enricher.append_directional_properties(
            result_line_df,
            mock_ds,
            base_col,
            membership_tagging=tagging
        )
        
        after_cols = len(result_line_df.columns)
        print(f"  Added {after_cols - before_cols} properties for '{base_col}' relationships")

    # Example 5: Practical energy analysis scenarios
    print("\n\n6. PRACTICAL ENERGY ANALYSIS SCENARIOS")
    print("-" * 40)
    
    print("\nScenario A: Generator Spatial Analysis")
    print("Adding node properties to understand generator geographical distribution:")
    spatial_generators = enricher.append_properties(
        generator_df.copy(), mock_ds, MembershipTagging.PREFIX
    )
    node_props = [col for col in spatial_generators.columns if col.startswith('node_')]
    print(f"Available node properties: {node_props}")
    
    print("\nScenario B: Transmission Network Analysis")
    print("Adding node properties to analyze transmission line characteristics:")
    network_lines = directional_enricher.append_properties(
        line_df.copy(), mock_ds, MembershipTagging.NONE
    )
    # Demonstrate voltage compatibility analysis
    from_voltage_cols = [col for col in network_lines.columns if 'voltage' in col and '_from' in col]
    to_voltage_cols = [col for col in network_lines.columns if 'voltage' in col and '_to' in col]
    if from_voltage_cols and to_voltage_cols:
        print(f"Can analyze voltage compatibility using: {from_voltage_cols[0]} vs {to_voltage_cols[0]}")
    
    print("\nScenario C: Missing Data Handling")
    print("Demonstrating graceful handling of incomplete relationships:")
    # Count NaN relationships before and after enrichment
    original_nans = generator_df.isna().sum().sum()
    enriched_nans = spatial_generators.isna().sum().sum()
    print(f"Original NaN values: {original_nans}")
    print(f"After enrichment NaN values: {enriched_nans} (preserved + new from missing refs)")
    
    # Example 6: Performance and memory considerations
    print("\n\n7. PERFORMANCE AND MEMORY CONSIDERATIONS")
    print("-" * 40)
    
    # Memory usage comparison
    import sys
    original_memory = sys.getsizeof(generator_df) + sys.getsizeof(line_df)
    enriched_memory = sys.getsizeof(spatial_generators) + sys.getsizeof(network_lines)
    
    print(f"Original DataFrames memory usage: {original_memory:,} bytes")
    print(f"Enriched DataFrames memory usage: {enriched_memory:,} bytes")
    print(f"Memory increase factor: {enriched_memory / original_memory:.2f}x")
    
    # Processing time consideration
    print("\nFor large datasets, consider:")
    print("- Processing in chunks if memory is constrained")
    print("- Using specific membership selection instead of full auto-enrichment")
    print("- Monitoring for missing reference warnings in production")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
