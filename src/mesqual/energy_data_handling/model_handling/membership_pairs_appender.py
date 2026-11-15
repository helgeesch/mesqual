from abc import ABC, abstractmethod
from typing import Literal
import pandas as pd

from mesqual.utils.pandas_utils.set_new_column import set_column
from mesqual.utils.multi_key_utils.common_base_key_finder import CommonBaseKeyFinder


class BaseMembershipPairsAppender(ABC):
    """Base class that adds one combination column per pair of from/to columns.

    A pair is two columns with the same base name and the from and to
    identifiers, for example `node_from` and `node_to`. There are three
    combinations:

    - Directional (`append_combo_columns`): keeps the direction, A to B.
    - Sorted (`append_sorted_combo_columns`): the values in sorted order, so
      A to B and B to A get the same value.
    - Opposite (`append_opposite_combo_columns`): the reverse direction, B to A.

    A subclass sets how two values combine, for example a string
    (`StringMembershipPairsAppender`) or a tuple (`TupleMembershipPairsAppender`).
    Either a suffix or a prefix must be set for each combination type.

    Args:
        from_identifier: Suffix of the source columns. Defaults to '_from'.
        to_identifier: Suffix of the target columns. Defaults to '_to'.
        combo_col_suffix: Suffix for directional combination column names. Defaults to '_combo'.
        combo_col_prefix: Prefix for directional combination column names. Defaults to None.
        sorted_combo_col_suffix: Suffix for sorted combination column names. Defaults to '_combo_sorted'.
        sorted_combo_col_prefix: Prefix for sorted combination column names. Defaults to None.
        opposite_combo_col_suffix: Suffix for opposite combination column names. Defaults to '_combo_opposite'.
        opposite_combo_col_prefix: Prefix for opposite combination column names. Defaults to None.

    Raises:
        ValueError: If neither suffix nor prefix is provided for any combination type.
    """

    def __init__(
        self,
        from_identifier: str = '_from',
        to_identifier: str = '_to',
        combo_col_suffix: str = '_combo',
        combo_col_prefix: str = None,
        sorted_combo_col_suffix: str = '_combo_sorted',
        sorted_combo_col_prefix: str = None,
        opposite_combo_col_suffix: str = '_combo_opposite',
        opposite_combo_col_prefix: str = None,
    ):
        self._from_identifier = from_identifier
        self._to_identifier = to_identifier

        self._combo_col_suffix = combo_col_suffix or ''
        self._combo_col_prefix = combo_col_prefix or ''
        if not any([self._combo_col_suffix, self._combo_col_prefix]):
            raise ValueError('Must specify either suffix or prefix for combo columns')

        self._sorted_combo_col_suffix = sorted_combo_col_suffix or ''
        self._sorted_combo_col_prefix = sorted_combo_col_prefix or ''
        if not any([self._sorted_combo_col_suffix, self._sorted_combo_col_prefix]):
            raise ValueError('Must specify either suffix or prefix for sorted combo columns')

        self._opposite_combo_col_suffix = opposite_combo_col_suffix or ''
        self._opposite_combo_col_prefix = opposite_combo_col_prefix or ''
        if not any([self._opposite_combo_col_suffix, self._opposite_combo_col_prefix]):
            raise ValueError('Must specify either suffix or prefix for opposite combo columns')

        self._common_base_key_finder = CommonBaseKeyFinder(from_identifier, to_identifier)

    def append_combo_columns(
        self,
        df_with_from_to_columns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Adds one column per from/to pair with the combination in the from/to direction.

        Args:
            df_with_from_to_columns: DataFrame containing paired relationship columns
                                   (e.g., 'node_from'/'node_to', 'region_from'/'region_to')

        Returns:
            Enhanced DataFrame with directional combination columns added.
                Original data preserved, new columns follow configured naming pattern.

        Examples:
            >>> appender = StringMembershipPairsAppender(separator=' - ')
            >>> appender.append_combo_columns(dataset.fetch('Line.Model'))[['node_from', 'node_to', 'node_combo']].iloc[:2]
                 node_from node_to node_combo
            Line
            L1          N1      N2    N1 - N2
            L2          N2      N3    N2 - N3
        """
        return self._append_combo_columns(df_with_from_to_columns, 'directional')

    def append_sorted_combo_columns(
        self,
        df_with_from_to_columns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Adds one column per from/to pair with the two values in sorted order.

        A to B and B to A get the same value, for example to find the lines of
        one corridor in both directions.

        Args:
            df_with_from_to_columns: DataFrame containing paired relationship columns
                                   with potential bidirectional connections

        Returns:
            Enhanced DataFrame with sorted combination columns added.
                Bidirectional relationships receive identical identifiers.

        Examples:
            >>> borders = pd.DataFrame({'country_from': ['C1', 'C2'], 'country_to': ['C2', 'C1']})
            >>> StringMembershipPairsAppender(separator=' - ').append_sorted_combo_columns(borders)['country_combo_sorted'].tolist()
            ['C1 - C2', 'C1 - C2']
        """
        return self._append_combo_columns(df_with_from_to_columns, 'sorted')

    def append_opposite_combo_columns(
        self,
        df_with_from_to_columns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Adds one column per from/to pair with the combination in the to/from direction.

        Args:
            df_with_from_to_columns: DataFrame containing directional relationships
                                   where reverse analysis is needed

        Returns:
            Enhanced DataFrame with opposite-direction combination columns added.
                Each relationship receives its reverse-direction identifier.

        Examples:
            >>> appender = StringMembershipPairsAppender(separator=' - ')
            >>> appender.append_opposite_combo_columns(dataset.fetch('Line.Model'))['node_combo_opposite'].iloc[:2].tolist()
            ['N2 - N1', 'N3 - N2']
        """
        return self._append_combo_columns(df_with_from_to_columns, 'opposite')

    def _append_combo_columns(
        self, df_with_from_to_columns: pd.DataFrame, which_combo: Literal['directional', 'sorted', 'opposite']
    ) -> pd.DataFrame:
        from_id = self._from_identifier
        to_id = self._to_identifier

        _col_names = df_with_from_to_columns.columns
        base_columns = self._common_base_key_finder.get_keys_for_which_all_association_tags_appear(_col_names)

        for base in base_columns:
            from_col = f'{base}{from_id}'
            to_col = f'{base}{to_id}'

            _from_values = df_with_from_to_columns[from_col]
            _to_values = df_with_from_to_columns[to_col]

            if which_combo == 'directional':
                new_col = f'{self._combo_col_prefix}{base}{self._combo_col_suffix}'
                new_values = self._combine_values(_from_values, _to_values)
            elif which_combo == 'sorted':
                new_col = f'{self._sorted_combo_col_prefix}{base}{self._sorted_combo_col_suffix}'
                new_values = self._combine_sorted_values(_from_values, _to_values)
            elif which_combo == 'opposite':
                new_col = f'{self._opposite_combo_col_prefix}{base}{self._opposite_combo_col_suffix}'
                new_values = self._combine_values(_to_values, _from_values)
            else:
                raise NotImplementedError

            df_with_from_to_columns = set_column(df_with_from_to_columns, new_col, new_values)

        return df_with_from_to_columns

    @abstractmethod
    def _combine_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        pass

    @abstractmethod
    def _combine_sorted_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        pass


class StringMembershipPairsAppender(BaseMembershipPairsAppender):
    """Combines a pair of values into one string, joined by `separator`."""

    def __init__(
        self,
        from_identifier: str = '_from',
        to_identifier: str = '_to',
        combo_col_suffix: str = '_combo',
        combo_col_prefix: str = None,
        sorted_combo_col_suffix: str = '_combo_sorted',
        sorted_combo_col_prefix: str = None,
        opposite_combo_col_suffix: str = '_combo_opposite',
        opposite_combo_col_prefix: str = None,
        separator: str = ' - ',
    ):
        """Initialize the string-based membership pairs appender.

        Args:
            from_identifier: Suffix of the source columns. Defaults to '_from'.
            to_identifier: Suffix of the target columns. Defaults to '_to'.
            combo_col_suffix: Suffix for directional combination column names. Defaults to '_combo'.
            combo_col_prefix: Prefix for directional combination column names. Defaults to None.
            sorted_combo_col_suffix: Suffix for sorted combination column names. Defaults to '_combo_sorted'.
            sorted_combo_col_prefix: Prefix for sorted combination column names. Defaults to None.
            opposite_combo_col_suffix: Suffix for opposite combination column names. Defaults to '_combo_opposite'.
            opposite_combo_col_prefix: Prefix for opposite combination column names. Defaults to None.
            separator: String separator for joining paired values. Defaults to ' - '.
                      Use ' → ' for directional flows, '-' for neutral connections,
                      ' <-> ' for bidirectional relationships.
        """
        super().__init__(
            from_identifier=from_identifier,
            to_identifier=to_identifier,
            combo_col_suffix=combo_col_suffix,
            combo_col_prefix=combo_col_prefix,
            sorted_combo_col_suffix=sorted_combo_col_suffix,
            sorted_combo_col_prefix=sorted_combo_col_prefix,
            opposite_combo_col_suffix=opposite_combo_col_suffix,
            opposite_combo_col_prefix=opposite_combo_col_prefix,
        )
        self._separator = separator

    def _combine_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Combines two series into directional string combinations.

        Args:
            a: Source/origin values (from column)
            b: Destination/target values (to column)

        Returns:
            Series with string combinations preserving a→b direction
        """
        return a.astype(str) + self._separator + b.astype(str)

    def _combine_sorted_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Combines two series into bidirectional string combinations with alphabetical ordering.

        Args:
            a: First set of relationship values
            b: Second set of relationship values

        Returns:
            Series with alphabetically sorted string combinations (bidirectional)
        """
        if a.empty and b.empty:
            return pd.Series()
        return pd.DataFrame({'a': a.astype(str), 'b': b.astype(str)}).apply(
            lambda x: self._separator.join(sorted([x['a'], x['b']])), axis=1
        )


class TupleMembershipPairsAppender(BaseMembershipPairsAppender):
    """Combines a pair of values into a tuple, for use as dictionary keys or index values.

    Examples:
        >>> appender = TupleMembershipPairsAppender()
        >>> appender.append_combo_columns(dataset.fetch('Line.Model'))['node_combo'].tolist()[:2]
        [('N1', 'N2'), ('N2', 'N3')]
    """

    def _combine_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Combines two series into directional tuple combinations.

        Args:
            a: Source/origin values (from column)
            b: Destination/target values (to column)

        Returns:
            Series with tuple combinations preserving (a, b) direction
        """
        return pd.Series([tuple(x) for x in zip(a, b)], index=a.index)

    def _combine_sorted_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Combines two series into bidirectional tuple combinations with alphabetical ordering.

        Args:
            a: First set of relationship values
            b: Second set of relationship values

        Returns:
            Series with alphabetically sorted tuple combinations (bidirectional)
        """
        return pd.Series([tuple(sorted([x, y])) for x, y in zip(a, b)], index=a.index)
