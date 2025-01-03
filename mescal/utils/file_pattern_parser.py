from typing import Dict, Union, List, Type
import re

from mescal.utils.str_to_bool import str_to_bool

PARSED_DTYPES = Type[Union[str, int, float, bool, None]]

DTYPE_PATTERNS = {
    str: '.',
    int: '\\d',
    float: '.',
    bool: '.',
    None: '.',
}


class FilePatternParser:
    def __init__(
            self,
            pattern: str,
            dtypes: Dict[str, PARSED_DTYPES] = None,
    ):
        """
        Initializes a parser with a pattern and optionally with a dict of data types.
        The "pattern" is a string with attributes that shall be parsed.
        All attributes must be encapsulated by curly braces (e.g. {year}, {country}, ...).
        Example:
            FilePatternParser(
                pattern='MyFile_pattern_{year}--{country}_bingbong-{version}.csv',
                dtypes=dict(year=int, version=float)
            )
        """
        self.pattern = pattern
        self.dtypes = dtypes if dtypes else dict()
        for att in self._find_attributes_enclosed_in_curly_braces(pattern):
            if att not in self.dtypes.keys():
                self.dtypes[att] = None
            else:
                self.dtypes[att] = dtypes[att]

    @staticmethod
    def _find_attributes_enclosed_in_curly_braces(text: str) -> List[str]:
        pattern = r"\{(.*?)\}"
        attributes_matched = re.findall(pattern, text)
        return attributes_matched

    def get_attributes_for_filename(self, filename: str) -> Dict[str, PARSED_DTYPES]:
        return self._get_attributes_for_filename(filename)

    def _get_attributes_for_filename(self, filename: str) -> Dict[str, PARSED_DTYPES]:
        regex_pattern = self.regex_pattern
        match = re.match(regex_pattern, filename)

        if match:
            results = match.groupdict()
            for att, dtype in self.dtypes.items():
                if dtype is not None:
                    if dtype == bool:
                        results[att] = str_to_bool(results[att])
                    else:
                        results[att] = dtype(results[att])
            return results
        raise ValueError(f"Could not find match to attributes for filename {filename}")

    @property
    def regex_pattern(self) -> str:
        pattern = str(self.pattern)
        for att, dtype in self.dtypes.items():
            dtype_pattern = DTYPE_PATTERNS.get(dtype, '.')
            pattern = pattern.replace(f'{{{att}}}', f'(?P<{att}>{dtype_pattern}*)')
        return pattern

    @property
    def glob_pattern(self) -> str:
        pattern = str(self.pattern)
        for att, dtype in self.dtypes.items():
            pattern = pattern.replace(f'{{{att}}}', '*')
        return pattern

    @staticmethod
    def remove_replace(filename: str, remove_parts: List[str] = None, replace_parts: Dict[str, str] = None) -> str:
        remove_parts = remove_parts if remove_parts else []
        replace_parts = replace_parts if replace_parts else dict()

        for rm in remove_parts:
            filename = filename.replace(rm, '')
        for rp_key, rp_value in replace_parts.items():
            filename = filename.replace(rp_key, rp_value)

        return filename


if __name__ == '__main__':
    parser = FilePatternParser(
        pattern='MyFile_pattern_{year}--{bz}_bingbong-{version}.csv',
        dtypes=dict(year=int, version=float)
    )
    samples = [
        'MyFile_pattern_2024--DE_bingbong-01.2.csv',
        'MyFile_pattern_1990--NO2_bingbong-0123.csv',
    ]
    for s in samples:
        print(parser.get_attributes_for_filename(s))
