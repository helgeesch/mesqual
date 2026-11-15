import logging
from enum import Enum

import pandas as pd

from mesqual.datasets import Dataset
from mesqual.utils.multi_key_utils.common_base_key_finder import CommonBaseKeyFinder

logger = logging.getLogger(__name__)

# TODO: Clean up modules and create common base class


class MembershipTagging(Enum):
    """How enriched property names are tagged when added to target DataFrames.

    - NONE: Property names remain unchanged (may cause conflicts with existing columns)
    - PREFIX: Property names get the membership name as prefix (e.g., 'node_voltage' from 'voltage')
    - SUFFIX: Property names get the membership name as suffix (e.g., 'voltage_node' from 'voltage')
    """

    NONE = 'none'
    PREFIX = 'prefix'
    SUFFIX = 'suffix'


class MembershipPropertyEnricher:
    """Adds the properties of the model objects that a frame refers to.

    The membership columns of the target DataFrame come from the
    `memberships` of its model flag spec (`FlagIndex.get_memberships`), for
    example the column `node` of `Generator.Model`. The enricher adds all
    properties of the referenced model objects. A NaN membership gives NaN
    properties.

    Args:
        membership_tag_separator: Separator used between membership name and property
            when PREFIX or SUFFIX tagging is applied

    Examples:
        >>> generators = dataset.fetch('Generator.Model')
        >>> enricher = MembershipPropertyEnricher()
        >>> enriched = enricher.append_properties(generators, dataset, 'Generator.Model', MembershipTagging.PREFIX)
        >>> list(enriched.columns)
        ['node', 'fuel', 'capacity', 'node_country', 'node_voltage', 'node_base_price', 'node_location']
        >>> enriched[['node', 'node_country', 'node_voltage']].iloc[:2]
                  node node_country  node_voltage
        Generator
        G1          N1           C1         380.0
        G2          N1           C1         380.0
    """

    def __init__(self, membership_tag_separator: str = '_'):
        """
        Initialize the membership property enricher.

        Args:
            membership_tag_separator: Character(s) used to separate membership names
                                     from property names in PREFIX/SUFFIX modes
        """
        self._membership_tag_separator = membership_tag_separator

    def identify_membership_columns(self, model_flag: str, column_names: list[str], dataset: Dataset) -> list[str]:
        """
        Return the columns of the frame that the model spec of `model_flag` declares as memberships.

        Args:
            model_flag: Model flag of the target DataFrame.
            column_names: Column names of the target DataFrame.
            dataset: Dataset whose flag index has the model spec.

        Returns:
            The membership columns that the frame has, in the order of `column_names`.

        Examples:
            >>> generators = dataset.fetch('Generator.Model')
            >>> MembershipPropertyEnricher().identify_membership_columns('Generator.Model', generators.columns, dataset)
            ['node']
        """
        memberships = dataset.flag_index.get_memberships(model_flag)
        return [col for col in column_names if col in memberships]

    def append_properties(
        self,
        target_df: pd.DataFrame,
        dataset: Dataset,
        model_flag: str,
        membership_tagging: MembershipTagging = MembershipTagging.NONE,
    ) -> pd.DataFrame:
        """
        Enriches target DataFrame with properties from all linked model objects.

        Identifies all membership columns and adds the properties of the
        referenced model objects.

        A NaN membership gets NaN properties. A referenced object that is
        missing from the source DataFrame gets NaN properties, and a warning is
        logged.

        Args:
            target_df: DataFrame to enrich (e.g., generator, line, storage data)
            dataset: Dataset containing linked model DataFrames with properties
            model_flag: Model flag of `target_df`. Its spec declares the memberships.
            membership_tagging: Strategy for naming enriched properties to avoid conflicts

        Returns:
            Enhanced DataFrame with all properties from linked model objects added.
                Original columns are preserved, new columns added based on memberships.
        """
        membership_columns = self.identify_membership_columns(model_flag, target_df.columns, dataset)
        result_df = target_df.copy()

        for column in membership_columns:
            result_df = self.append_single_membership_properties(
                result_df, dataset, model_flag, column, membership_tagging
            )

        return result_df

    def append_single_membership_properties(
        self,
        target_df: pd.DataFrame,
        dataset: Dataset,
        model_flag: str,
        membership_column: str,
        membership_tagging: MembershipTagging = MembershipTagging.NONE,
    ) -> pd.DataFrame:
        """
        Enriches target DataFrame with properties from a specific membership relationship.

        This method provides fine-grained control over property enrichment by handling
        a single membership column. Useful when custom logic is needed for specific
        relationships or when processing memberships sequentially with different
        tagging strategies.

        The memberships of the model spec of `model_flag` give the model flag
        that the column refers to. A left join keeps all target records and
        adds the available properties.

        A referenced object that is missing from the source DataFrame gets NaN
        properties, and a warning is logged.

        Args:
            target_df: DataFrame to enrich (must contain the membership column)
            dataset: Dataset with access to linked model DataFrames
            model_flag: Model flag of `target_df`.
            membership_column: Name of column containing object references (e.g., 'node', 'fuel_type')
            membership_tagging: Strategy for naming enriched properties

        Returns:
            DataFrame with properties from the linked model objects added.
                All original rows preserved; NaN memberships result in NaN properties.
        """
        source_flag = dataset.flag_index.get_memberships(model_flag)[membership_column]
        source_df = dataset.fetch(source_flag)

        membership_objects = target_df[membership_column].dropna().unique()
        missing_objects = set(membership_objects) - set(source_df.index)

        if missing_objects:
            self._log_missing_objects_warning(missing_objects, membership_column)

        source_properties = source_df.copy()

        match membership_tagging:
            case MembershipTagging.PREFIX:
                source_properties = source_properties.add_prefix(f'{membership_column}{self._membership_tag_separator}')
            case MembershipTagging.SUFFIX:
                source_properties = source_properties.add_suffix(f'{self._membership_tag_separator}{membership_column}')

        result_df = target_df.merge(source_properties, left_on=membership_column, right_index=True, how='left')

        return result_df

    def _log_missing_objects_warning(self, missing_objects: set, membership_column: str, max_show: int = 5):
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
        warning_suffix = ', and more' if num_missing > max_show else ''
        logger.warning(
            f'{num_missing} objects missing in source dataframe for {membership_column}: '
            f'{list(missing_objects)[:max_show]}{warning_suffix}.'
        )


class DirectionalMembershipPropertyEnricher:
    """Adds the properties of both ends of directional memberships, such as `node_from` and `node_to`.

    A pair is two membership columns with the same base name and the from
    and to identifiers (default: '_from' and '_to'), which refer to the same
    model flag. The enricher adds the properties of each end, tagged with the
    direction.

    Args:
        from_identifier: Suffix/prefix identifying 'from' direction (default: '_from')
        to_identifier: Suffix/prefix identifying 'to' direction (default: '_to')
        membership_tag_separator: Separator for property name construction

    Examples:
        >>> lines = dataset.fetch('Line.Model')
        >>> enricher = DirectionalMembershipPropertyEnricher()
        >>> enricher.identify_from_to_columns('Line.Model', lines.columns, dataset)
        ['node']
        >>> enriched = enricher.append_properties(lines, dataset, 'Line.Model', MembershipTagging.NONE)
        >>> enriched[['node_from', 'country_from', 'node_to', 'country_to']].iloc[:2]
             node_from country_from node_to country_to
        Line
        L1          N1           C1      N2         C2
        L2          N2           C2      N3         C1
    """

    def __init__(
        self,
        from_identifier: str = '_from',
        to_identifier: str = '_to',
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

    def identify_from_to_columns(self, model_flag: str, column_names: list[str], dataset: Dataset) -> list[str]:
        """
        Identifies base names for directional membership column pairs.

        Analyzes column names to find base membership types that have both 'from'
        and 'to' variants. For example, identifies 'node' as a base when both
        'node_from' and 'node_to' columns exist and the model spec of
        `model_flag` declares both as memberships to the same model flag.

        Args:
            model_flag: Model flag of the target DataFrame.
            column_names: List of column names from the target DataFrame
            dataset: Dataset whose flag index has the model spec

        Returns:
            List of base column names that have both from/to variants
        """
        memberships = dataset.flag_index.get_memberships(model_flag)
        potential_columns = self._tag_finder.get_keys_for_which_all_association_tags_appear(column_names)
        return [col for col in potential_columns if self._directional_target(col, memberships, column_names)]

    def _directional_target(self, base_column: str, memberships: dict[str, str], column_names) -> str | None:
        """The model flag that both directional columns of `base_column` refer to, else `None`."""
        targets = {
            memberships.get(self._get_full_column_name(base_column, tag, column_names))
            for tag in (self._from_identifier, self._to_identifier)
        }
        return targets.pop() if len(targets) == 1 else None

    def append_properties(
        self,
        target_df: pd.DataFrame,
        dataset: Dataset,
        model_flag: str,
        membership_tagging: MembershipTagging = MembershipTagging.NONE,
    ) -> pd.DataFrame:
        """
        Enriches DataFrame with properties from all directional relationships.

        Identifies all from/to column pairs and adds the properties of the
        objects at both ends, for example the countries of the nodes of a line.

        Each directional relationship results in two sets of enriched properties:
        one for the 'from' direction and one for the 'to' direction, clearly
        distinguished by directional suffixes.

        A referenced object that is missing from the source DataFrame gets NaN
        properties, and a warning is logged.

        Args:
            target_df: DataFrame with directional relationships (e.g., transmission lines)
            dataset: Dataset containing model objects and their properties
            model_flag: Model flag of `target_df`. Its spec declares the memberships.
            membership_tagging: Strategy for property naming (applied before directional tags)

        Returns:
            Enhanced DataFrame with directional properties added. Original data preserved,
                new columns follow pattern: [prefix_]property_name[_suffix]_direction
        """
        membership_base_columns = self.identify_from_to_columns(model_flag, target_df.columns, dataset)
        result_df = target_df.copy()

        for base_column in membership_base_columns:
            result_df = self.append_directional_properties(
                result_df, dataset, model_flag, base_column, membership_tagging
            )

        return result_df

    def append_directional_properties(
        self,
        target_df: pd.DataFrame,
        dataset: Dataset,
        model_flag: str,
        base_column: str,
        membership_tagging: MembershipTagging = MembershipTagging.NONE,
    ) -> pd.DataFrame:
        """
        Enriches the frame with the properties of the model flag that both directional columns refer to.

        Args:
            target_df: DataFrame containing the directional columns.
            dataset: Dataset with the referenced model flag.
            model_flag: Model flag of `target_df`. Its spec declares the memberships.
            base_column: Base membership name (e.g., 'node' for 'node_from'/'node_to').
            membership_tagging: Property naming strategy (applied before directional tags).

        Raises:
            ValueError: The two directional columns are not memberships to the same model flag.
        """
        memberships = dataset.flag_index.get_memberships(model_flag)
        source_flag = self._directional_target(base_column, memberships, target_df.columns)
        if source_flag is None:
            raise ValueError(
                f'{model_flag}: the directional columns of {base_column!r} are no memberships to one model flag.'
            )
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
        membership_tagging: MembershipTagging = MembershipTagging.NONE,
    ) -> pd.DataFrame:
        """
        Enriches DataFrame with properties from a directional relationship.

        Handles a single from/to membership pair by adding the corresponding model
        DataFrame's properties with directional tags.

        The method processes both directions (from/to) for the specified base column,
        adding properties with appropriate directional suffixes. A NaN
        membership gets NaN properties. A referenced object that is missing
        from the source DataFrame gets NaN properties, and a warning is logged.

        Args:
            target_df: DataFrame containing the directional columns
            source_df: DataFrame containing the properties
            base_column: Base membership name (e.g., 'node' for 'node_from'/'node_to')
            membership_tagging: Property naming strategy (applied before directional tags)

        Returns:
            DataFrame with directional properties added for the specified relationship.
                Properties follow naming pattern: [prefix_]property[_suffix]_direction
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
                    source_properties = source_properties.add_prefix(f'{base_column}{self._membership_tag_separator}')
                case MembershipTagging.SUFFIX:
                    source_properties = source_properties.add_suffix(f'{self._membership_tag_separator}{base_column}')

            source_properties = source_properties.add_suffix(tag)

            result_df = result_df.merge(source_properties, left_on=membership_column, right_index=True, how='left')
        return result_df

    def _get_full_column_name(self, base_column: str, tag: str, df_columns: list[str]) -> str:
        """
        Determines the actual column name for a directional membership.

        Returns `{base_column}{tag}` if it is a column, else `{tag}{base_column}`.

        Args:
            base_column: Base membership name (e.g., 'node')
            tag: Directional identifier (e.g., '_from', '_to')
            df_columns: List of actual column names in the DataFrame

        Returns:
            Actual column name found in the DataFrame
        """
        test_suffix = f'{base_column}{tag}'
        return test_suffix if test_suffix in df_columns else f'{tag}{base_column}'

    def _log_missing_objects_warning(self, missing_objects: set, membership_column: str, max_show: int = 5):
        num_missing = len(missing_objects)
        warning_suffix = ', and more' if num_missing > max_show else ''
        logger.warning(
            f'{num_missing} objects missing in source dataframe for {membership_column}: '
            f'{list(missing_objects)[:max_show]}{warning_suffix}.'
        )
