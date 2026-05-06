from integra_pose.utils import plugin_opt_in


def test_plugin_disabled_by_default(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "enabled_plugins.json"
    monkeypatch.setattr(plugin_opt_in, "_OPT_IN_STORE_PATH", store_path)

    plugin_dir = tmp_path / "plugin_example"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text("x = 1\n", encoding="utf-8")

    assert plugin_opt_in.is_plugin_enabled(plugin_dir) is False


def test_opt_in_round_trip_and_reprompt_on_change(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "enabled_plugins.json"
    monkeypatch.setattr(plugin_opt_in, "_OPT_IN_STORE_PATH", store_path)

    plugin_dir = tmp_path / "plugin_example"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
    (plugin_dir / "plugin.py").write_text("x = 1\n", encoding="utf-8")

    plugin_opt_in.set_plugin_enabled(plugin_dir, True, name="Example", version="1.0.0", source="user")
    assert plugin_opt_in.is_plugin_enabled(plugin_dir) is True

    # Changing the plugin entrypoints should invalidate the previous consent.
    (plugin_dir / "plugin.py").write_text("x = 2\n", encoding="utf-8")
    assert plugin_opt_in.is_plugin_enabled(plugin_dir) is False


def test_disable_sets_state(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "enabled_plugins.json"
    monkeypatch.setattr(plugin_opt_in, "_OPT_IN_STORE_PATH", store_path)

    plugin_dir = tmp_path / "plugin_example"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text("x = 1\n", encoding="utf-8")

    plugin_opt_in.set_plugin_enabled(plugin_dir, True, name="Example", version="1.0.0")
    plugin_opt_in.set_plugin_enabled(plugin_dir, False)

    record = plugin_opt_in.get_record(plugin_dir)
    assert record is not None
    assert record.state == "disabled"
    assert plugin_opt_in.is_plugin_enabled(plugin_dir) is False
