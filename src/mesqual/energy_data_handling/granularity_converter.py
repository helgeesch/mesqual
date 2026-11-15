import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from mesqual.enums import QuantityTypeEnum


class TimeSeriesGranularityConverter:
    """Converts time series between granularities with the quantity type of the data.

    - Upsampling (coarser to finer) repeats each row `n` times, with
      `n = native step / target step`. `INTENSIVE` (prices, power) keeps the
      value. `EXTENSIVE` (volumes, welfare) divides it by `n`, so the sum stays.
    - Downsampling (finer to coarser) takes the mean (`INTENSIVE`) or the sum
      (`EXTENSIVE`) per target step. A target step without input rows gives
      no row.

    A gap is a missing timestamp. It stays a gap: the converter never fills
    forward. A NaN value is a value: upsampling repeats it as NaN, and a
    target step whose rows are all NaN gives NaN.

    The native step is known per calendar day. A day with two or more rows
    takes the most frequent step between its rows, the smallest of equally
    frequent steps. A day with one row (for example daily data) takes the
    step to the next input row. So weekly data spreads each row over 7
    days, and monthly data each row over its own month. The last row of the
    input, alone on its day, takes the step of the nearest day before it,
    else the most frequent step of the whole input: for monthly data the
    last row can be up to three days too long or too short, and it can
    reach into the next month. A single row before a gap looks the same as
    coarser data, so it spreads up to the next row. The day step picks the
    direction per day, so a change of step at a day boundary (for example
    hourly before a date, 15 minutes after it) converts each day with its
    own step.

    A step of whole days spans calendar days on tz-aware data: daily data
    upsamples to 23 or 25 hours on DST days. Other steps are absolute, so
    hourly data has a step of one hour also across the DST change. A
    repeated row never reaches past the next input row. An `EXTENSIVE`
    value splits evenly over its rows, also where the last of them is
    shorter (a target that does not divide the step).

    The native step comes from the row index, for all columns together. So
    a frame with an hourly and a 15-minute column on a common index converts
    on the 15-minute grid, and the NaN rows of the hourly column stay NaN.

    The input is sorted by time first. Duplicate timestamps and `NaT` raise.

    The target is a pandas offset (`parse_frequency`). Downsampling to `'D'`
    on tz-aware data gives calendar days, also on DST days (23 or 25 hours).
    """

    @staticmethod
    def parse_frequency(freq: str | pd.Timedelta | pd.offsets.BaseOffset) -> pd.offsets.BaseOffset:
        """Return `freq` as a pandas offset of a positive fixed frequency.

        Strings are pandas frequencies such as `'15min'`, `'1h'`, `'h'` or
        `'D'`, or Timedelta strings such as `'15 minutes'`. A step of whole
        days (`'24h'`, `'1 day'`, `pd.Timedelta('1D')`) is a calendar day,
        `pd.offsets.Day(n)`, the same as `'D'`: on tz-aware data it has 23 or
        25 hours on DST days.

        Raises:
            ValueError: `freq` is no `str`, `pd.Timedelta` or `pd.offsets.BaseOffset`,
                no fixed frequency (for example `'W'` or `'ME'`), or not positive.
        """
        error = ValueError(f"{freq!r} is not a positive fixed frequency such as '15min', '1h' or 'D'.")
        if not isinstance(freq, (str, pd.Timedelta, pd.offsets.BaseOffset)):
            raise error
        try:
            offset = to_offset(freq)
        except ValueError:
            try:
                offset = to_offset(pd.Timedelta(freq))
            except (ValueError, TypeError) as cause:
                raise error from cause
        try:
            step = pd.Timedelta(offset)
        except (ValueError, TypeError) as cause:
            raise error from cause
        if step <= pd.Timedelta(0):
            raise error
        if step % pd.Timedelta('1D') == pd.Timedelta(0):
            return pd.offsets.Day(step.days)
        return offset

    def convert(
        self,
        data: pd.DataFrame | pd.Series,
        target_freq: str | pd.Timedelta | pd.offsets.BaseOffset,
        quantity_type: QuantityTypeEnum,
        time_level: int | str | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Convert time series data to a target frequency.

        Args:
            data: Input time series with DatetimeIndex.
            target_freq: Target frequency, see `parse_frequency`
                (e.g., "15min", "1h", "D", pd.Timedelta(hours=1)).
            quantity_type: INTENSIVE (prices, power) or EXTENSIVE (volumes, welfare).
                UNKNOWN and AMBIGUOUS raise: the caller states the type.
            time_level: Name or position of the datetime level of a row
                `MultiIndex`. The data converts per combination of the other
                levels, which keep their order. `None`: the row index is a
                `DatetimeIndex`.

        Returns:
            Converted data, same type as input (Series or DataFrame). Empty
                input gives a copy, whatever its index type. A series of one row
                is returned unchanged: it has no step. With `time_level`, a group
                of one row takes the most frequent step of the whole input.

        Raises:
            ValueError: `quantity_type` is neither INTENSIVE nor EXTENSIVE,
                `target_freq` is no positive fixed frequency, or the index
                has duplicate timestamps (per group with `time_level`) or
                `NaT`.
            TypeError: The index of non-empty data is not a DatetimeIndex.
        """
        if quantity_type not in (QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE):
            raise ValueError(
                f'Granularity conversion needs an INTENSIVE or EXTENSIVE quantity type, got {quantity_type}. '
                f'State the quantity type of the data.'
            )
        if data.empty:
            return data.copy()
        offset = self.parse_frequency(target_freq)
        if time_level is not None:
            return self._convert_per_group(data, offset, quantity_type, time_level)
        return self._convert_time_index(data, offset, quantity_type)

    def _convert_time_index(
        self,
        data: pd.DataFrame | pd.Series,
        offset: pd.offsets.BaseOffset,
        quantity_type: QuantityTypeEnum,
        fallback_step: pd.Timedelta | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Convert data with a `DatetimeIndex`; the arguments are checked.

        `fallback_step` is the step of a day when no day has two rows;
        `None`: the most frequent step of `data`.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(f'Index must be DatetimeIndex, got {type(data.index)}')
        if data.index.hasnans:
            raise ValueError('Granularity conversion needs timestamps; the index has NaT.')
        if data.index.has_duplicates:
            raise ValueError('Granularity conversion needs unique timestamps; the index has duplicates.')
        data = data.sort_index()
        if isinstance(data, pd.Series):
            return (
                self._convert_frame(data.to_frame(), offset, quantity_type, fallback_step).iloc[:, 0].rename(data.name)
            )
        return self._convert_frame(data, offset, quantity_type, fallback_step)

    def _convert_per_group(
        self,
        data: pd.DataFrame | pd.Series,
        offset: pd.offsets.BaseOffset,
        quantity_type: QuantityTypeEnum,
        time_level: int | str,
    ) -> pd.DataFrame | pd.Series:
        """Convert the datetime level `time_level` of a row `MultiIndex` per combination of the other levels."""
        time_position = self._level_position(data.index, time_level)
        other_levels = [position for position in range(data.index.nlevels) if position != time_position]
        fallback_step = self._grouped_series_step(data.index, time_position)
        result = data.groupby(level=other_levels, sort=False, group_keys=True, dropna=False).apply(
            lambda group: self._convert_time_index(group.droplevel(other_levels), offset, quantity_type, fallback_step)
        )
        order = other_levels + [time_position]
        return result.reorder_levels([order.index(position) for position in range(data.index.nlevels)])

    @staticmethod
    def _level_position(index: pd.Index, time_level: int | str) -> int:
        """Position of `time_level` in a row `MultiIndex`; raise `TypeError` or `ValueError` if there is none."""
        if not isinstance(index, pd.MultiIndex):
            raise TypeError(f'time_level needs a row MultiIndex, got {type(index).__name__}.')
        names = list(index.names)
        if time_level in names:
            return names.index(time_level)
        if (
            isinstance(time_level, int)
            and not isinstance(time_level, bool)
            and -index.nlevels <= time_level < index.nlevels
        ):
            return time_level % index.nlevels
        raise ValueError(f'time_level {time_level!r} is no level of the row index; levels: {names}.')

    def convert_to_target_index(
        self,
        data: pd.DataFrame | pd.Series,
        target_index: pd.DatetimeIndex,
        quantity_type: QuantityTypeEnum,
    ) -> pd.DataFrame | pd.Series:
        """Convert data to match a specific target DatetimeIndex.

        The target frequency is the most frequent step of `target_index`, by
        the rule of the native step; a step of whole days is a calendar day. The result is reindexed to `target_index`.

        Raises:
            ValueError: `target_index` has fewer than two rows, so it has no step.
        """
        if len(target_index) < 2:
            raise ValueError(f'target_index needs at least two rows to give a step, got {len(target_index)}.')
        result = self.convert(data, self._series_step(target_index), quantity_type)
        return result.reindex(target_index)

    # ── internals ────────────────────────────────────────────────────────

    def _convert_frame(
        self,
        df: pd.DataFrame,
        offset: pd.offsets.BaseOffset,
        quantity_type: QuantityTypeEnum,
        fallback_step: pd.Timedelta | None,
    ) -> pd.DataFrame:
        """Convert a sorted frame, per row in the direction of its day."""
        target_step = pd.Timedelta(offset)
        day_step = self._day_steps(df.index, fallback_step)
        span_ends = pd.Series(self._span_ends(df.index, pd.TimedeltaIndex(day_step.to_numpy())), index=df.index)
        upsample, downsample = (day_step > target_step).to_numpy(), (day_step < target_step).to_numpy()
        # NaT (a series of one row) is neither: no step is known.
        keep = ~upsample & ~downsample
        parts = []
        if upsample.any():
            parts.append(self._upsample(df[upsample], target_step, quantity_type, span_ends[upsample]))
        if downsample.any():
            parts.append(self._downsample(df[downsample], offset, quantity_type))
        if keep.any():
            parts.append(df[keep])
        return pd.concat(parts).sort_index()

    @staticmethod
    def _downsample(
        df: pd.DataFrame,
        target: pd.offsets.BaseOffset,
        quantity_type: QuantityTypeEnum,
    ) -> pd.DataFrame:
        resampler = df.resample(target)
        result = resampler.sum(min_count=1) if quantity_type == QuantityTypeEnum.EXTENSIVE else resampler.mean()
        return result[resampler.size() > 0]

    def _upsample(
        self,
        df: pd.DataFrame,
        target_step: pd.Timedelta,
        quantity_type: QuantityTypeEnum,
        span_ends: pd.Series,
    ) -> pd.DataFrame:
        """Repeat each row from its timestamp up to its span end (`_span_ends`)."""
        index = df.index
        starts, ends = index, pd.DatetimeIndex(span_ends.to_numpy())
        on_calendar = index.tz is not None and target_step % pd.Timedelta('1D') == pd.Timedelta(0)
        if on_calendar:
            starts, ends = starts.tz_localize(None), ends.tz_localize(None)
        repeats = np.maximum(np.ceil((ends - starts) / target_step).to_numpy(dtype=float), 1).astype(int)
        source_rows = np.repeat(np.arange(len(index)), repeats)
        step_numbers = np.arange(len(source_rows)) - np.repeat(np.cumsum(repeats) - repeats, repeats)
        stamps = starts[source_rows] + step_numbers * target_step
        if on_calendar:
            stamps = self._localize(stamps, index.tz)
        result = df.iloc[source_rows].set_axis(stamps.rename(index.name), axis=0)
        if quantity_type == QuantityTypeEnum.EXTENSIVE:
            return result.div(repeats[source_rows], axis=0)
        return result

    @classmethod
    def _span_ends(cls, index: pd.DatetimeIndex, steps: pd.TimedeltaIndex) -> pd.DatetimeIndex:
        """Per row: the end of its native step, at most the next row.

        A step of whole days on tz-aware data ends on the wall clock, so a
        daily step spans one calendar day, also on DST days.
        """
        ends = index + steps
        whole_days = np.asarray(steps % pd.Timedelta('1D') == pd.Timedelta(0))
        if index.tz is not None and whole_days.any():
            on_wall_clock = cls._localize(index.tz_localize(None) + steps, index.tz)
            ends = pd.DatetimeIndex(np.where(whole_days, on_wall_clock, ends)).tz_convert(index.tz)
        next_rows = index[1:].append(ends[-1:])
        return ends.where(ends <= next_rows, next_rows)

    @staticmethod
    def _localize(wall_clock: pd.DatetimeIndex, tz) -> pd.DatetimeIndex:
        """Localize wall-clock stamps; an ambiguous stamp takes the DST side, a missing one moves forward."""
        return wall_clock.tz_localize(tz, ambiguous=np.ones(len(wall_clock), dtype=bool), nonexistent='shift_forward')

    @classmethod
    def _day_steps(cls, index: pd.DatetimeIndex, fallback_step: pd.Timedelta | None = None) -> pd.Series:
        """Per row: the native step of its day.

        A day with two or more rows takes its most frequent step
        (`_steps`). A day with one row takes the step to the next input
        row. The last row of the input, alone on its day, takes the step of
        the day before, else `fallback_step` (by default the most frequent
        step of `index`). `NaT` if no step is known.
        """
        dates = pd.Series(index.date, index=index)
        unique_dates = pd.unique(dates)
        steps = cls._steps(index)
        day_step = cls._most_frequent(steps.where(dates.eq(dates.shift())), dates).reindex(unique_dates)
        next_step = pd.Series(steps.shift(-1).to_numpy(), index=dates.to_numpy())
        single = dates.value_counts().reindex(unique_dates) == 1
        day_step = day_step.where(~single, next_step.groupby(level=0).first().reindex(unique_dates))
        fallback_step = cls._series_step(index) if fallback_step is None else fallback_step
        day_step = day_step.ffill().fillna(fallback_step)
        return pd.Series(day_step.reindex(dates.to_numpy()).to_numpy(), index=index)

    @classmethod
    def _grouped_series_step(cls, index: pd.MultiIndex, time_position: int) -> pd.Timedelta:
        """The most frequent step of the time level within the groups of the other levels."""
        times = index.get_level_values(time_position)
        groups = pd.factorize(index.droplevel(time_position), use_na_sentinel=False)[0]
        order = np.lexsort((times.asi8, groups))
        return cls._series_step(times[order], groups[order])

    @classmethod
    def _series_step(cls, index: pd.DatetimeIndex, groups: np.ndarray | None = None) -> pd.Timedelta:
        """The most frequent step (`_steps`) of `index`; `NaT` if it has no step."""
        steps = cls._steps(index, groups)
        return cls._most_frequent(steps, pd.Series(0, index=index)).get(0, pd.NaT)

    @staticmethod
    def _steps(index: pd.DatetimeIndex, groups: np.ndarray | None = None) -> pd.Series:
        """Per row: the positive step from the row before in the same group, else `NaT`.

        A step of whole days on the wall clock is a wall-clock step, so daily
        data on tz-aware data has a step of one day, also on DST days. Other
        steps are absolute: hourly data has a step of one hour across DST.
        """
        wall_clock = index.tz_localize(None) if index.tz is not None else index
        wall_clock_steps = pd.Series(wall_clock).diff()
        absolute_steps = pd.Series(index).diff()
        whole_days = (wall_clock_steps > pd.Timedelta(0)) & (wall_clock_steps % pd.Timedelta('1D') == pd.Timedelta(0))
        steps = wall_clock_steps.where(whole_days, absolute_steps)
        steps = steps.where(steps > pd.Timedelta(0))
        if groups is not None:
            groups = pd.Series(groups)
            steps = steps.where(groups.eq(groups.shift()))
        return pd.Series(steps.to_numpy(), index=index)

    @staticmethod
    def _most_frequent(steps: pd.Series, groups: pd.Series) -> pd.Series:
        """Per group: the most frequent step, the smallest of equally frequent steps."""
        valid = steps.notna().to_numpy()
        if not valid.any():
            return pd.Series(dtype='timedelta64[ns]')
        codes, keys = pd.factorize(np.asarray(groups)[valid])
        values = steps.to_numpy()[valid].astype('timedelta64[ns]').view('int64')
        order = np.lexsort((values, codes))
        codes, values = codes[order], values[order]
        run_starts = np.flatnonzero(np.r_[True, (codes[1:] != codes[:-1]) | (values[1:] != values[:-1])])
        counts = np.diff(np.r_[run_starts, len(codes)])
        run_codes, run_values = codes[run_starts], values[run_starts]
        best = np.lexsort((run_values, -counts, run_codes))
        best = best[np.r_[True, run_codes[best][1:] != run_codes[best][:-1]]]
        return pd.Series(pd.to_timedelta(run_values[best], unit='ns'), index=keys[run_codes[best]])
