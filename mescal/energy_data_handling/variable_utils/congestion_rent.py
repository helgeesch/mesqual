from dataclasses import dataclass
import pandas as pd

from mescal.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CongestionRentCalculator:
    """Calculates congestion rents for electricity transmission lines.
    
    Congestion rent represents the economic value captured by transmission
    assets due to price differences between nodes. It's calculated as the
    product of power flow, price spread, and time granularity.
    
    The calculator handles bidirectional flows (up/down) and accounts for
    transmission losses by using separate sent and received quantities.
    This provides accurate congestion rent calculations that reflect actual
    market conditions and physical constraints.
    
    Mathematical formulation:
    - Congestion rent (up) = granularity × (received_up × price_to - sent_up × price_from)
    - Congestion rent (down) = granularity × (received_down × price_from - sent_down × price_to)
    - Total congestion rent = congestion_rent_up + congestion_rent_down
    
    Attributes:
        sent_up: Power sent in up direction (MW or MWh)
        received_up: Power received in up direction after losses (MW or MWh)  
        sent_down: Power sent in down direction (MW or MWh)
        received_down: Power received in down direction after losses (MW or MWh)
        price_node_from: Price at sending node (€/MWh)
        price_node_to: Price at receiving node (€/MWh)
        granularity_hrs: Time granularity in hours (auto-detected if None)
        
    Example:

        >>> import pandas as pd
        >>> # Time series data
        >>> index = pd.date_range('2024-01-01', periods=3, freq='h')
        >>> # Flow and price data
        >>> calc = CongestionRentCalculator(
        ...     sent_up=pd.Series([100, 150, 200], index=index),
        ...     received_up=pd.Series([95, 142, 190], index=index),  # 5% losses
        ...     sent_down=pd.Series([50, 75, 100], index=index),
        ...     received_down=pd.Series([48, 71, 95], index=index),  # 4% losses
        ...     price_node_from=pd.Series([45, 50, 55], index=index),
        ...     price_node_to=pd.Series([65, 70, 75], index=index)
        ... )
        >>> total_rent = calc.calculate()
        >>> print(total_rent)  # Congestion rent in €
    """
    sent_up: pd.Series
    received_up: pd.Series
    sent_down: pd.Series
    received_down: pd.Series
    price_node_from: pd.Series
    price_node_to: pd.Series
    granularity_hrs: pd.Series | float = None

    def __post_init__(self):
        """Initialize granularity and validate input consistency."""
        if self.granularity_hrs is None:
            if isinstance(self.sent_up.index, pd.DatetimeIndex):
                if len(self.sent_up.index) > 0:
                    from mescal.energy_data_handling.granularity_analyzer import TimeSeriesGranularityAnalyzer
                    analyzer = TimeSeriesGranularityAnalyzer(strict_mode=False)
                    self.granularity_hrs = analyzer.get_granularity_as_series_of_hours(self.sent_up.index)
                else:
                    self.granularity_hrs = 0
            else:
                logger.warning(f'Granularity for CongestionRentCalculator is defaulting back to 1 hrs.')
                self.granularity_hrs = 1
        if isinstance(self.granularity_hrs, (float, int)):
            self.granularity_hrs = pd.Series(self.granularity_hrs, index=self.sent_up.index)
        self.__check_indices()

    def __check_indices(self):
        """Validate that all input Series have matching indices.
        
        Raises:
            ValueError: If any Series has mismatched index with sent_up
        """
        ref_index = self.sent_up.index
        to_check = [
            self.received_up,
            self.sent_down,
            self.received_down,
            self.price_node_from,
            self.price_node_to
        ]
        for v in to_check:
            if not ref_index.equals(v.index):
                raise ValueError(f'All indices of provided series must be equal.')

    @property
    def congestion_rent_up(self) -> pd.Series:
        """Calculate congestion rent for up direction flows.
        
        Returns:
            Series with congestion rents in up direction (€)
        """
        return self.granularity_hrs * (
                self.received_up * self.price_node_to -
                self.sent_up * self.price_node_from
        )

    @property
    def congestion_rent_down(self) -> pd.Series:
        """Calculate congestion rent for down direction flows.
        
        Returns:
            Series with congestion rents in down direction (€)
        """
        return self.granularity_hrs * (
                self.received_down * self.price_node_from -
                self.sent_down * self.price_node_to
        )

    @property
    def congestion_rent_total(self) -> pd.Series:
        """Calculate total congestion rent (sum of up and down directions).
        
        Returns:
            Series with total congestion rents (€)
        """
        return self.congestion_rent_up + self.congestion_rent_down

    def calculate(self) -> pd.Series:
        """Calculate total congestion rent (convenience method).
        
        Returns:
            Series with total congestion rents in € (same as congestion_rent_total property)
        """
        return self.congestion_rent_total

    @classmethod
    def from_net_flow_without_losses(
            cls,
            net_flow: pd.Series,
            price_node_from: pd.Series,
            price_node_to: pd.Series,
            granularity_hrs: float = None
    ) -> pd.Series:
        """Calculate congestion rent from net flow data assuming no losses.
        
        Convenience method for cases where transmission losses are negligible
        or not available. Splits net flow into unidirectional components and
        assumes sent equals received for each direction.
        
        Args:
            net_flow: Net power flow (positive = up direction, negative = down direction) in MW or MWh
            price_node_from: Price at sending node (€/MWh)
            price_node_to: Price at receiving node (€/MWh)
            granularity_hrs: Time granularity in hours (auto-detected if None)
            
        Returns:
            Series with total congestion rents in €
            
        Example:
            
            >>> # Net flow with price data
            >>> net_flow = pd.Series([100, -50, 75], index=time_index)  
            >>> rent = CongestionRentCalculator.from_net_flow_without_losses(
            ...     net_flow, price_from, price_to
            ... )
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
            granularity_hrs=granularity_hrs
        ).calculate()

    @classmethod
    def from_up_and_down_flow_without_losses(
            cls,
            flow_up: pd.Series,
            flow_down: pd.Series,
            price_node_from: pd.Series,
            price_node_to: pd.Series,
            granularity_hrs: float = None
    ) -> pd.Series:
        """Calculate congestion rent from bidirectional flow data assuming no losses.
        
        Convenience method for cases where flows are already separated into up/down
        directions and transmission losses are negligible. Assumes sent equals
        received for each direction.
        
        Args:
            flow_up: Power flow in up direction (MW or MWh, non-negative)
            flow_down: Power flow in down direction (MW or MWh, non-negative)  
            price_node_from: Price at sending node (€/MWh)
            price_node_to: Price at receiving node (€/MWh)
            granularity_hrs: Time granularity in hours (auto-detected if None)
            
        Returns:
            Series with total congestion rents in €
            
        Example:
            
            >>> # Separate up/down flows
            >>> flow_up = pd.Series([100, 0, 75], index=time_index)
            >>> flow_down = pd.Series([0, 50, 0], index=time_index)
            >>> rent = CongestionRentCalculator.from_up_and_down_flow_without_losses(
            ...     flow_up, flow_down, price_from, price_to
            ... )
        """
        return cls(
            sent_up=flow_up,
            received_up=flow_up,
            sent_down=flow_down,
            received_down=flow_down,
            price_node_from=price_node_from,
            price_node_to=price_node_to,
            granularity_hrs=granularity_hrs
        ).calculate()


if __name__ == "__main__":
    index = pd.date_range("2024-01-01", periods=24, freq="H")
    dummy_data = {
        "sent_up": pd.Series([100, 150, 200], index=index[:3]),
        "received_up": pd.Series([95, 142, 190], index=index[:3]),
        "sent_down": pd.Series([50, 75, 100], index=index[:3]),
        "received_down": pd.Series([48, 71, 95], index=index[:3]),
        "price_node_from": pd.Series([45, 50, 55], index=index[:3]),
        "price_node_to": pd.Series([65, 70, 75], index=index[:3])
    }

    calculator = CongestionRentCalculator(**dummy_data)

    print("Direction-wise congestion rents:")
    print("Up direction:", calculator.congestion_rent_up)
    print("\nDown direction:", calculator.congestion_rent_down)
    print("\nTotal:", calculator.congestion_rent_total)

    net_flow = pd.Series([50, 75, 100], index=index[:3])
    simple_rent = CongestionRentCalculator.from_net_flow_without_losses(
        net_flow=net_flow,
        price_node_from=dummy_data["price_node_from"],
        price_node_to=dummy_data["price_node_to"]
    )
    print("\nSimplified calculation without losses:")
    print(simple_rent)
