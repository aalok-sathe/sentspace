"""
MWE test case for getting syntax features on a single sentence using SentSpace
and the sentspace-syntax-server

setup:
    - install sentspace in the current environment using `poetry`
    - run `github.com/sentspace/sentspace-syntax-server` interactively
"""


import sentspace

server_url = "http://localhost"
server_port = "8000"


s = sentspace.Sentence("This is a sentence.", uid="TEST")


def test_dlt_features():
    syntax_features = sentspace.syntax.get_features(
        s,
        dlt=True,
        left_corner=False,
        syntax_server=server_url,
        syntax_port=server_port,
    )


if __name__ == "__main__":
    test_dlt_features()
