import difflib


def find_difference_and_join(text_a: str, text_b: str, join_differences_by: str = ' vs ') -> str:
    _J = join_differences_by
    differ = difflib.Differ()
    diff = differ.compare(text_a.split(), text_b.split())
    diff = list(diff)

    a_elements = []
    b_elements = []
    result = []
    for element in diff:
        if " " == element[0]:
            if a_elements or b_elements:
                result.append(f'({" ".join(a_elements)} {_J} {" ".join(b_elements)})')
                a_elements, b_elements = [], []
            result.append(element[2:])
        elif "+" == element[0]:
            b_elements.append(element[2:])
        elif "-" == element[0]:
            a_elements.append(element[2:])
    if a_elements or b_elements:
        result.append(f'({" ".join(a_elements)} {_J} {" ".join(b_elements)})')

    return ' '.join(result)


if __name__ == '__main__':
    print(find_difference_and_join(
        text_a='mean consumption ABC (MWh)',
        text_b='mean consumption DEF (MWh)'
    ))  # mean consumption (ABC vs DEF) (MWh)

    print(find_difference_and_join(
        text_a='sum generators (DE_LU) cost (EUR)',
        text_b='sum generators cost (EUR)'
    ))  # sum generators ((DE_LU) vs ) cost (EUR)

    print(find_difference_and_join(
        text_a='sum export (DE_LU) (GWh)',
        text_b='sum cross border exchange (GWh)'
    ))  # sum (export (DE_LU) vs cross border exchange) (GWh)

    print(find_difference_and_join(
        text_a='Hans consumption Peter (GWh)',
        text_b='Bing consumption Bong (GWh)'
    ))  # (Hans vs Bing) consumption (Peter vs Bong) (GWh)
