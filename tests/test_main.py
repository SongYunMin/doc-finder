from doc_finder import main as main_module


def test_main_starts_uvicorn_server(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(app: str, host: str, port: int, reload: bool) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["reload"] = reload

    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)

    main_module.main([])

    assert captured == {
        "app": "doc_finder.app:app",
        "host": "127.0.0.1",
        "port": 8000,
        "reload": False,
    }
