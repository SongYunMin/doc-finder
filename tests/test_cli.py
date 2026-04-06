from doc_finder import cli as cli_module


def test_cli_prints_search_result(capsys) -> None:
    cli_module.main(["--query", "hello cli"])

    assert capsys.readouterr().out.strip() == (
        "{'query': 'hello cli', 'answer': 'hello cli'}"
    )
