import pandas as pd

from mescal.energy_data_handling.area_accounting.area_variable_base import AreaVariableCalculatorBase


class AreaSumCalculator(AreaVariableCalculatorBase):
    """General calculator for summing node-level extensive quantities to area level.

    This calculator aggregates extensive quantities (values that scale with system size)
    from node-level to area-level using summation. Typical use cases include power
    generation, demand, energy volumes, reserves, and other additive quantities in
    energy systems analysis.

    Unlike intensive quantities (like prices), extensive quantities should be summed
    when aggregating to higher geographic levels, making this calculator appropriate
    for many physical quantities in energy modeling.

    Inherits from AreaVariableCalculatorBase and provides the MESCAL framework's
    standard approach for area-level aggregation of extensive variables.

    Args:
        node_model_df: DataFrame mapping nodes to areas
        area_column: Column name containing area identifiers

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Node model
        >>> node_model = pd.DataFrame({
        ...     'bidding_zone': ['DE_LU', 'DE_LU', 'FR', 'FR']
        ... }, index=['DE1', 'DE2', 'FR1', 'FR2'])
        >>> # Sum calculator
        >>> calc = AreaSumCalculator(node_model, 'bidding_zone')
        >>> # Node generation data
        >>> generation = pd.DataFrame({
        ...     'DE1': [800, 850], 'DE2': [750, 780],
        ...     'FR1': [900, 920], 'FR2': [850, 870]
        ... }, index=pd.date_range('2024-01-01', periods=2, freq='h'))
        >>> # Sum to areas
        >>> area_generation = calc.calculate(generation)
        >>> print(area_generation)
        bidding_zone  DE_LU   FR
        2024-01-01 00:00:00  1550  1750
        2024-01-01 01:00:00  1630  1790
    """

    def calculate(self, node_data_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate area sums from node-level extensive quantity data.

        Sums node-level values within each area to create area-level aggregates.
        This method is designed for extensive quantities where summation is the
        appropriate aggregation method (e.g., generation, demand, volumes).

        Missing nodes are handled gracefully - if a node exists in the node model
        but not in the data, it's simply ignored. Areas with no available nodes
        are omitted from the output.

        In case you want to exclude certain nodes from the aggregation (e.g. because
        they are virtual or synthetic nodes), you can simply remove them from the
        node_data_df before passing it to this method.

        Args:
            node_data_df: DataFrame with node-level time series data. Index should
                be datetime, columns should be node identifiers. Values represent
                extensive quantities (MW, MWh, etc.) that should be summed.

        Returns:
            DataFrame with area-level aggregated data. Index matches input time series,
            columns represent areas. Units are preserved from input data.

        Raises:
            ValueError: If node_data_df structure is invalid

        Example:
            >>> # Sum generation across nodes
            >>> area_generation = calc.calculate(node_generation_df)
            >>> # Sum demand across nodes  
            >>> area_demand = calc.calculate(node_demand_df)
        """

        self._validate_node_data(node_data_df, 'node_data_df')

        area_sums = {}
        for area in self.areas:
            area_nodes = self.get_area_nodes(area)
            area_nodes = [n for n in area_nodes if n in node_data_df.columns]
            if area_nodes:
                area_sums[area] = node_data_df[area_nodes].sum(axis=1)

        result = pd.DataFrame(area_sums)
        result.columns.name = self.area_column
        return result


if __name__ == '__main__':
    import numpy as np
    
    print("=== MESCAL Area Sum Calculator Examples ===\n")
    
    # Create sample node model with multiple area levels
    node_model_df = pd.DataFrame({
        'bidding_zone': ['DE_LU', 'DE_LU', 'FR', 'FR', 'BE', 'NL'],
        'country': ['DE', 'DE', 'FR', 'FR', 'BE', 'NL'],
        'region': ['CWE', 'CWE', 'CWE', 'CWE', 'CWE', 'CWE']
    }, index=['DE1', 'DE2', 'FR1', 'FR2', 'BE1', 'NL1'])
    
    print("Node model with multiple area hierarchies:")
    print(node_model_df)
    print()
    
    # Create sample time series data
    time_index = pd.date_range('2024-01-01', periods=6, freq='h')
    np.random.seed(42)
    
    # Example 1: Power generation data (MW)
    print("1. Power generation aggregation (MW):")
    node_generation = pd.DataFrame({
        'DE1': np.random.uniform(800, 1200, 6),
        'DE2': np.random.uniform(700, 1100, 6),
        'FR1': np.random.uniform(900, 1300, 6),
        'FR2': np.random.uniform(600, 1000, 6),
        'BE1': np.random.uniform(400, 800, 6),
        'NL1': np.random.uniform(500, 900, 6)
    }, index=time_index)
    
    print("Node generation (MW):")
    print(node_generation.round(1))
    print()
    
    # Sum to bidding zones
    sum_calc_bz = AreaSumCalculator(node_model_df, 'bidding_zone')
    bz_generation = sum_calc_bz.calculate(node_generation)
    print("Bidding zone generation (MW):")
    print(bz_generation.round(1))
    print()
    
    # Example 2: Energy demand data (MWh)
    print("2. Energy demand aggregation (MWh):")
    node_demand = pd.DataFrame({
        'DE1': np.random.uniform(600, 900, 6),
        'DE2': np.random.uniform(500, 800, 6),
        'FR1': np.random.uniform(700, 1000, 6),
        'FR2': np.random.uniform(450, 750, 6),
        'BE1': np.random.uniform(300, 600, 6),
        'NL1': np.random.uniform(400, 700, 6)
    }, index=time_index)
    
    print("Node demand (MWh):")
    print(node_demand.round(1))
    print()
    
    # Sum to countries
    sum_calc_country = AreaSumCalculator(node_model_df, 'country')
    country_demand = sum_calc_country.calculate(node_demand)
    print("Country demand (MWh):")
    print(country_demand.round(1))
    print()
    
    # Example 3: Multiple aggregation levels
    print("3. Multi-level aggregation comparison:")
    # Regional level
    sum_calc_region = AreaSumCalculator(node_model_df, 'region')
    regional_demand = sum_calc_region.calculate(node_demand)
    
    comparison = pd.DataFrame({
        'Node_Total': node_demand.sum(axis=1),
        'BZ_Total': bz_generation.sum(axis=1),  # Using generation for variety
        'Country_Total': country_demand.sum(axis=1),
        'Region_Total': regional_demand.sum(axis=1)
    })
    
    print("Aggregation level comparison (first 3 hours):")
    print(comparison.head(3).round(1))
    print()
    
    # Example 4: Handling missing nodes
    print("4. Handling missing nodes (excluding NL1):")
    partial_generation = node_generation.drop('NL1', axis=1)
    partial_bz_generation = sum_calc_bz.calculate(partial_generation)
    
    print("Bidding zone generation with NL1 missing:")
    print("(Note: NL bidding zone will be missing from output)")
    print(partial_bz_generation.round(1))
    print()
    
    # Example 5: Reserve capacity aggregation
    print("5. Reserve capacity aggregation (MW):")
    node_reserves = pd.DataFrame({
        'DE1': np.random.uniform(50, 100, 6),
        'DE2': np.random.uniform(40, 90, 6),
        'FR1': np.random.uniform(60, 120, 6),
        'FR2': np.random.uniform(35, 85, 6),
        'BE1': np.random.uniform(25, 60, 6),
        'NL1': np.random.uniform(30, 70, 6)
    }, index=time_index)
    
    country_reserves = sum_calc_country.calculate(node_reserves)
    print("Country reserve capacity (MW):")
    print(country_reserves.round(1))
    print()
    
    # Example 6: Energy balance calculation
    print("6. Energy balance by bidding zone (Generation - Demand):")
    # Convert generation to same time periods as demand for balance
    node_gen_for_balance = node_generation.iloc[:len(node_demand)]
    bz_gen_balance = sum_calc_bz.calculate(node_gen_for_balance)
    bz_dem_balance = sum_calc_bz.calculate(node_demand)
    
    balance = bz_gen_balance - bz_dem_balance
    balance.columns = [f"{col}_balance" for col in balance.columns]
    
    print("Bidding zone energy balance (Generation - Demand, MWh):")
    print("(Positive = surplus, Negative = deficit)")
    print(balance.round(1))
    print()
    print("Note: This demonstrates extensive quantity aggregation where")
    print("individual node contributions sum to area totals")
