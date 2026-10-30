from oracle_builder.training.workflow import _best_effort_failure_record


def test_failure_recording_attempts_remaining_actions_when_log_fails(capsys):
    called = []

    def broken():
        raise OSError("too many open files")

    _best_effort_failure_record([
        ("write log", broken),
        ("update artifact", lambda: called.append("updated")),
    ])
    assert called == ["updated"]
    assert "too many open files" in capsys.readouterr().err
