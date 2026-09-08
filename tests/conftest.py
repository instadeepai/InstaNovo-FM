import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--figshare-search-data",
        action="store",
        default=None,
        help="Path to the search_data.xlsx downloaded from Figshare, for the "
        "content-equality check against data/search_data.xlsx.",
    )
