"""Pytest configuration for the whole repo.

Adds the --impl flag so you can point the test suite at your own
implementation directory:

    pytest week1/test_week1.py -v --impl submissions/23b1234/week1

With no --impl flag, the tests run against the skeleton files in the week
folder itself (which will fail until you fill in the TODOs — that is the
correct result on day one).
"""


def pytest_addoption(parser):
    parser.addoption(
        "--impl",
        action="store",
        default=None,
        help="Directory containing your implementation for this week",
    )
