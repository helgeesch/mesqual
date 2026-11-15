import logging
from dataclasses import dataclass, fields
from numbers import Real

import pandas as pd

from mesqual.energy_data_handling.granularity_analyzer import GranularityError, TimeSeriesGranularityAnalyzer
from mesqual.energy_data_handling.network_lines_data import NetworkLineFlowsData


logger = logging.getLogger(__name__)


@dataclass
class CongestionRentCalculator:
    """Calculates congestion rents for electricity transmission lines.

    Congestion rent is the value that a line captures from the price
    difference between its two nodes: flow times price spread times the
    length of the time step.

    The calculator handles flows in both directions (up and down). It uses
    separate sent and received quantities, so transmission losses reduce
    the rent.

    Formulas:

    - Congestion rent (up) = granularity × (received_up × price_to - sent_up × price_from)
    - Congestion rent (down) = granularity × (received_down × price_from - sent_down × price_to)
    - Total congestion rent = congestion_rent_up + congestion_rent_down

    Input shapes:
        All six inputs are `Series` for one line, or all are `DataFrame`s
        with one column per line. The frames have the same columns. All
        inputs have the same index. A `Series` result keeps the name of
        the inputs when all inputs have the same name.

    Granularity:
        `granularity_hrs` is the length of each time step in hours. A
        `Series` has the index of the inputs and applies along the rows,
        so each line of a frame gets the same step length per row. A
        number applies to all rows. The values of an explicit
        `granularity_hrs` are not checked.

        Without `granularity_hrs`, a `DatetimeIndex` gives the step length
        per day from `TimeSeriesGranularityAnalyzer` in strict mode. Days
        may have different step lengths, for example a day of 60 min steps
        and a day of 15 min steps. Mixed step lengths or a missing row
        within a day raise `GranularityError`. An index that is not a
        `DatetimeIndex` gives 1 hour and logs a warning.

        A day is the calendar date in the timezone of the index. A caller
        that knows the MTU grid passes `granularity_hrs`. This applies
        also to an index in UTC or in naive local time where the MTU
        length changes or DST changes: the calendar dates of such an index
        do not match the delivery days.

    Attributes:
        sent_up: Power sent in up direction at node_from (MW).
        received_up: Power received in up direction at node_to, after losses (MW).
        sent_down: Power sent in down direction at node_to (MW).
        received_down: Power received in down direction at node_from, after losses (MW).
        price_node_from: Price at node_from (€/MWh).
        price_node_to: Price at node_to (€/MWh).
        granularity_hrs: Step length in hours. Detected from the index if None.

    Raises:
        TypeError: The inputs mix `Series` and `DataFrame`s, or
            `granularity_hrs` is neither a numeric `Series` nor a number.
        ValueError: The indices, the columns of the frames, or the index of
            a `granularity_hrs` Series do not match, or auto-detection gets
            an index with `NaT`.
        GranularityError: Auto-detection finds duplicate timestamps, mixed
            step lengths within a day, or a day with one row. The message
            asks to pass `granularity_hrs`.

    Examples:
        >>> index = pd.date_range('2024-01-01', periods=2, freq='h')
        >>> calculator = CongestionRentCalculator(
        ...     sent_up=pd.Series([100.0, 150.0], index=index),
        ...     received_up=pd.Series([95.0, 142.0], index=index),  # 5% losses
        ...     sent_down=pd.Series([0.0, 0.0], index=index),
        ...     received_down=pd.Series([0.0, 0.0], index=index),
        ...     price_node_from=pd.Series([45.0, 50.0], index=index),
        ...     price_node_to=pd.Series([65.0, 70.0], index=index),
        ... )
        >>> calculator.calculate().tolist()  # EUR per hour
        [1675.0, 2440.0]
    """

    sent_up: pd.Series | pd.DataFrame
    received_up: pd.Series | pd.DataFrame
    sent_down: pd.Series | pd.DataFrame
    received_down: pd.Series | pd.DataFrame
    price_node_from: pd.Series | pd.DataFrame
    price_node_to: pd.Series | pd.DataFrame
    granularity_hrs: pd.Series | float | None = None

    def __post_init__(self):
        """Validate the inputs and set the granularity."""
        self._check_inputs()
        if self.granularity_hrs is None:
            self.granularity_hrs = self._detect_granularity_hrs(self.sent_up.index)
        elif isinstance(self.granularity_hrs, Real) and not isinstance(self.granularity_hrs, bool):
            self.granularity_hrs = pd.Series(float(self.granularity_hrs), index=self.sent_up.index)
        elif not isinstance(self.granularity_hrs, pd.Series):
            raise TypeError('granularity_hrs must be a Series or a number')
        elif not self._is_number_dtype(self.granularity_hrs.dtype):
            raise TypeError(f'A granularity_hrs Series must be numeric, got dtype {self.granularity_hrs.dtype}.')
        if not self.granularity_hrs.index.equals(self.sent_up.index):
            raise ValueError('The index of granularity_hrs must equal the index of the inputs.')

    def _check_inputs(self):
        is_frame = isinstance(self.sent_up, pd.DataFrame)
        for name in (field.name for field in fields(self) if field.name != 'granularity_hrs'):
            values = getattr(self, name)
            if isinstance(values, pd.DataFrame) != is_frame:
                raise TypeError(f'All inputs must be Series or all must be DataFrames; {name} differs from sent_up.')
            if not self.sent_up.index.equals(values.index):
                raise ValueError(f'Index mismatch: sent_up vs {name}.')
            if is_frame and not self.sent_up.columns.equals(values.columns):
                raise ValueError(f'Columns mismatch: sent_up vs {name}.')

    @staticmethod
    def _is_number_dtype(dtype) -> bool:
        return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)

    @staticmethod
    def _detect_granularity_hrs(index: pd.Index) -> pd.Series:
        if not isinstance(index, pd.DatetimeIndex):
            logger.warning('Granularity for CongestionRentCalculator is defaulting back to 1 hrs.')
            return pd.Series(1.0, index=index)
        try:
            return TimeSeriesGranularityAnalyzer(strict_mode=True).get_granularity_as_series_of_hours(index)
        except GranularityError as error:
            raise GranularityError(f'{error} Pass granularity_hrs.') from error

    def _per_step(self, values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        return values.mul(self.granularity_hrs.to_numpy(), axis=0)

    @property
    def congestion_rent_up(self) -> pd.Series | pd.DataFrame:
        """Congestion rent of the flows in up direction (€).

        Returns:
            Series, or DataFrame with one column per line.
        """
        return self._per_step(self.received_up * self.price_node_to - self.sent_up * self.price_node_from)

    @property
    def congestion_rent_down(self) -> pd.Series | pd.DataFrame:
        """Congestion rent of the flows in down direction (€).

        Returns:
            Series, or DataFrame with one column per line.
        """
        return self._per_step(self.received_down * self.price_node_from - self.sent_down * self.price_node_to)

    @property
    def congestion_rent_total(self) -> pd.Series | pd.DataFrame:
        """Total congestion rent, the sum of up and down direction (€).

        Returns:
            Series, or DataFrame with one column per line.
        """
        return self.congestion_rent_up + self.congestion_rent_down

    def calculate(self) -> pd.Series | pd.DataFrame:
        """Calculate the total congestion rent (same as `congestion_rent_total`).

        Returns:
            Series, or DataFrame with one column per line, in €.
        """
        return self.congestion_rent_total

    @classmethod
    def from_node_prices(
        cls,
        flows: NetworkLineFlowsData,
        line_model: pd.DataFrame,
        node_prices: pd.DataFrame,
        granularity_hrs: pd.Series | float | None = None,
        node_from_col: str = 'node_from',
        node_to_col: str = 'node_to',
    ) -> 'CongestionRentCalculator':
        """Build the calculator for all lines from node prices and the line model.

        Maps the node prices to the two ends of each line with the columns
        `node_from_col` and `node_to_col` of the line model. The result has
        one column per line in `flows`.

        Args:
            flows: Flows of the lines, one column per line.
            line_model: Line table indexed by line, with the columns
                `node_from_col` and `node_to_col`.
            node_prices: Prices (€/MWh), one column per node, with the index of the flows.
            granularity_hrs: Step length in hours. Detected from the index if None.
            node_from_col: Column of the line model with the node at the start of each line.
            node_to_col: Column of the line model with the node at the end of each line.

        Returns:
            Calculator with one column per line.

        Raises:
            KeyError: A line of `flows` is not in `line_model`, or a node of
                these lines has no column in `node_prices`.

        Examples:
            >>> from mesqual.energy_data_handling.network_lines_data import NetworkLineFlowsData
            >>> flows = NetworkLineFlowsData.from_net_flow_without_losses(dataset.fetch('Line.Results.flow_sent_net'))
            >>> calculator = CongestionRentCalculator.from_node_prices(
            ...     flows, dataset.fetch('Line.Model'), dataset.fetch('Node.Results.price')
            ... )
            >>> calculator.calculate().columns.tolist()
            ['L1', 'L2', 'L3', 'L4', 'L5']
        """
        line_ends = line_model.loc[flows.sent_up.columns, [node_from_col, node_to_col]]
        return cls(
            sent_up=flows.sent_up,
            received_up=flows.received_up,
            sent_down=flows.sent_down,
            received_down=flows.received_down,
            price_node_from=node_prices[line_ends[node_from_col]].set_axis(line_ends.index, axis=1),
            price_node_to=node_prices[line_ends[node_to_col]].set_axis(line_ends.index, axis=1),
            granularity_hrs=granularity_hrs,
        )

    @classmethod
    def from_net_flow_without_losses(
        cls,
        net_flow: pd.Series | pd.DataFrame,
        price_node_from: pd.Series | pd.DataFrame,
        price_node_to: pd.Series | pd.DataFrame,
        granularity_hrs: pd.Series | float | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Calculate congestion rent from net flow data, without losses.

        Splits the net flow into up and down direction. Sent equals
        received in each direction.

        Args:
            net_flow: Net flow (MW). Positive is up direction, negative is down direction.
            price_node_from: Price at node_from (€/MWh).
            price_node_to: Price at node_to (€/MWh).
            granularity_hrs: Step length in hours. Detected from the index if None.

        Returns:
            Total congestion rent in €, with the shape of `net_flow`.

        Examples:
            >>> index = pd.date_range('2024-01-01', periods=2, freq='h')
            >>> net_flow = pd.Series([100.0, -50.0], index=index)
            >>> price_from, price_to = pd.Series([40.0, 40.0], index=index), pd.Series([50.0, 30.0], index=index)
            >>> CongestionRentCalculator.from_net_flow_without_losses(net_flow, price_from, price_to).tolist()
            [1000.0, 500.0]
        """
        sent_up = net_flow.clip(lower=0)
        sent_down = (-net_flow).clip(lower=0)
        return cls(
            sent_up=sent_up,
            received_up=sent_up,
            sent_down=sent_down,
            received_down=sent_down,
            price_node_from=price_node_from,
            price_node_to=price_node_to,
            granularity_hrs=granularity_hrs,
        ).calculate()

    @classmethod
    def from_up_and_down_flow_without_losses(
        cls,
        flow_up: pd.Series | pd.DataFrame,
        flow_down: pd.Series | pd.DataFrame,
        price_node_from: pd.Series | pd.DataFrame,
        price_node_to: pd.Series | pd.DataFrame,
        granularity_hrs: pd.Series | float | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Calculate congestion rent from up and down flows, without losses.

        Sent equals received in each direction.

        Args:
            flow_up: Flow in up direction (MW, non-negative).
            flow_down: Flow in down direction (MW, non-negative).
            price_node_from: Price at node_from (€/MWh).
            price_node_to: Price at node_to (€/MWh).
            granularity_hrs: Step length in hours. Detected from the index if None.

        Returns:
            Total congestion rent in €, with the shape of `flow_up`.

        Examples:
            >>> index = pd.date_range('2024-01-01', periods=2, freq='h')
            >>> flow_up, flow_down = pd.Series([100.0, 0.0], index=index), pd.Series([0.0, 50.0], index=index)
            >>> price_from, price_to = pd.Series([40.0, 40.0], index=index), pd.Series([50.0, 30.0], index=index)
            >>> CongestionRentCalculator.from_up_and_down_flow_without_losses(flow_up, flow_down, price_from, price_to).tolist()
            [1000.0, 500.0]
        """
        return cls(
            sent_up=flow_up,
            received_up=flow_up,
            sent_down=flow_down,
            received_down=flow_down,
            price_node_from=price_node_from,
            price_node_to=price_node_to,
            granularity_hrs=granularity_hrs,
        ).calculate()
