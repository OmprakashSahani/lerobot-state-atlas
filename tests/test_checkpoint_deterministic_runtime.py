import os

import pytest

from lerobot_state_atlas.checkpoint_comparison import (
    DeterministicTorchExecutionError,
    deterministic_torch_execution,
)


class FakeTorch:
    def __init__(self):
        self.enabled = False
        self.warn_only = True
        self.precision = "medium"
        self.fail_apply = set()
        self.fail_restore = set()
        self.mutate_before_apply_failure = False
        self.calls = {}
        self.backends = _Backend(
            self,
            cudnn=_SettingGroup(
                self,
                "cudnn",
                deterministic=False,
                benchmark=True,
                allow_tf32=True,
            ),
            cuda=_Backend(
                self,
                matmul=_SettingGroup(self, "cuda.matmul", allow_tf32=True),
            ),
        )

    def _change(self, name, value, apply):
        call = self.calls.get(name, 0) + 1
        self.calls[name] = call
        if call == 1 and name in self.fail_apply:
            if self.mutate_before_apply_failure:
                apply()
            raise RuntimeError(f"apply {name} failed")
        if call > 1 and name in self.fail_restore:
            raise RuntimeError(f"restore {name} failed")
        apply()

    def are_deterministic_algorithms_enabled(self):
        return self.enabled

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self.warn_only

    def use_deterministic_algorithms(self, value, *, warn_only=False):
        self._change(
            "deterministic_algorithms",
            (value, warn_only),
            lambda: self._set_algorithms(value, warn_only),
        )

    def _set_algorithms(self, value, warn_only):
        self.enabled = value
        self.warn_only = warn_only

    def get_float32_matmul_precision(self):
        return self.precision

    def set_float32_matmul_precision(self, value):
        self._change(
            "float32_matmul_precision",
            value,
            lambda: setattr(self, "precision", value),
        )


class _Backend:
    def __init__(self, owner, **values):
        self.__dict__.update(values)


class _SettingGroup:
    def __init__(self, owner, prefix, **values):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_values", values)

    def __getattr__(self, name):
        return self._values[name]

    def __setattr__(self, name, value):
        token = f"{self._prefix}.{name}"
        self._owner._change(
            token,
            value,
            lambda: self._values.__setitem__(name, value),
        )


def snapshot(fake):
    return (
        fake.enabled,
        fake.warn_only,
        fake.backends.cudnn.deterministic,
        fake.backends.cudnn.benchmark,
        fake.backends.cuda.matmul.allow_tf32,
        fake.backends.cudnn.allow_tf32,
        fake.precision,
    )


def test_applies_records_and_restores_all_settings(monkeypatch) -> None:
    fake = FakeTorch()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "prior")
    with deterministic_torch_execution(
        device="cuda:0", model_dtype="bfloat16", torch_module=fake
    ) as settings:
        assert settings.cublas_workspace_config == ":4096:8"
        assert fake.enabled
        assert fake.backends.cudnn.deterministic
        assert not fake.backends.cudnn.benchmark
        assert not fake.backends.cuda.matmul.allow_tf32
        assert not fake.backends.cudnn.allow_tf32
        assert fake.precision == "highest"
    assert not fake.enabled
    assert fake.warn_only
    assert not fake.backends.cudnn.deterministic
    assert fake.backends.cudnn.benchmark
    assert fake.backends.cuda.matmul.allow_tf32
    assert fake.backends.cudnn.allow_tf32
    assert fake.precision == "medium"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "prior"


def test_rejects_device_dtype_and_apply_failure() -> None:
    with pytest.raises(DeterministicTorchExecutionError, match="CUDA"):
        with deterministic_torch_execution(
            device="cpu", model_dtype="float32", torch_module=FakeTorch()
        ):
            pass
    with pytest.raises(DeterministicTorchExecutionError, match="model_dtype"):
        with deterministic_torch_execution(
            device="cuda:0", model_dtype="float16", torch_module=FakeTorch()
        ):
            pass
    fake = FakeTorch()
    fake.fail_apply.add("deterministic_algorithms")
    with pytest.raises(
        DeterministicTorchExecutionError, match="deterministic_algorithms"
    ):
        with deterministic_torch_execution(
            device="cuda:0", model_dtype="float32", torch_module=fake
        ):
            pass


@pytest.mark.parametrize(
    "failed_setting",
    [
        "deterministic_algorithms",
        "cudnn.deterministic",
        "cudnn.benchmark",
        "cuda.matmul.allow_tf32",
        "cudnn.allow_tf32",
        "float32_matmul_precision",
    ],
)
def test_mid_setup_failure_restores_every_changed_setting(
    failed_setting, monkeypatch
) -> None:
    fake = FakeTorch()
    original = snapshot(fake)
    fake.fail_apply.add(failed_setting)
    fake.mutate_before_apply_failure = True
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    entered = False
    with pytest.raises(
        DeterministicTorchExecutionError, match=rf"apply {failed_setting} failed"
    ):
        with deterministic_torch_execution(
            device="cuda:0", model_dtype="float32", torch_module=fake
        ):
            entered = True
    assert not entered
    assert snapshot(fake) == original
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_setup_failure_restores_exact_existing_environment(monkeypatch) -> None:
    fake = FakeTorch()
    fake.fail_apply.add("cudnn.benchmark")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "original-value")
    with pytest.raises(DeterministicTorchExecutionError):
        with deterministic_torch_execution(
            device="cuda:0", model_dtype="float32", torch_module=fake
        ):
            pass
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "original-value"


def test_required_setting_capture_failure_never_enters_or_mutates_environment(
    monkeypatch,
) -> None:
    fake = FakeTorch()
    fake.get_float32_matmul_precision = lambda: (_ for _ in ()).throw(
        RuntimeError("precision getter unavailable")
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "original-value")
    entered = False
    with pytest.raises(
        DeterministicTorchExecutionError, match="precision getter unavailable"
    ):
        with deterministic_torch_execution(
            device="cuda:0", model_dtype="float32", torch_module=fake
        ):
            entered = True
    assert not entered
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "original-value"
    assert snapshot(fake) == (False, True, False, True, True, True, "medium")


def test_restoration_continues_and_reports_all_failures(monkeypatch) -> None:
    fake = FakeTorch()
    fake.fail_restore.update({"float32_matmul_precision", "cuda.matmul.allow_tf32"})
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(DeterministicTorchExecutionError) as caught:
        with deterministic_torch_execution(
            device="cuda:0", model_dtype="float32", torch_module=fake
        ):
            pass
    message = str(caught.value)
    assert "float32_matmul_precision" in message
    assert "cuda.matmul.allow_tf32" in message
    assert not fake.enabled
    assert fake.warn_only
    assert not fake.backends.cudnn.deterministic
    assert fake.backends.cudnn.benchmark
    assert fake.backends.cudnn.allow_tf32
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_setup_and_restoration_failures_preserve_both_diagnostics() -> None:
    fake = FakeTorch()
    fake.fail_apply.add("cudnn.benchmark")
    fake.fail_restore.add("deterministic_algorithms")
    with pytest.raises(DeterministicTorchExecutionError) as caught:
        with deterministic_torch_execution(
            device="cuda:0", model_dtype="float32", torch_module=fake
        ):
            pass
    message = str(caught.value)
    assert "apply cudnn.benchmark failed" in message
    assert "restore deterministic_algorithms failed" in message


def test_body_and_restoration_failures_preserve_both_diagnostics() -> None:
    fake = FakeTorch()
    fake.fail_restore.add("cudnn.deterministic")
    with pytest.raises(DeterministicTorchExecutionError) as caught:
        with deterministic_torch_execution(
            device="cuda:0", model_dtype="float32", torch_module=fake
        ):
            raise RuntimeError("body exploded")
    assert "body exploded" in str(caught.value)
    assert "restore cudnn.deterministic failed" in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_body_failure_is_preserved_when_restoration_succeeds() -> None:
    fake = FakeTorch()
    with pytest.raises(RuntimeError, match="body exploded"):
        with deterministic_torch_execution(
            device="cuda:0", model_dtype="float32", torch_module=fake
        ):
            raise RuntimeError("body exploded")
    assert snapshot(fake) == (False, True, False, True, True, True, "medium")


def test_nested_contexts_restore_immediate_outer_values(monkeypatch) -> None:
    fake = FakeTorch()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "process-original")
    original = snapshot(fake)
    with deterministic_torch_execution(
        device="cuda:0", model_dtype="float32", torch_module=fake
    ):
        outer = snapshot(fake)
        assert outer == (True, False, True, False, False, False, "highest")
        with deterministic_torch_execution(
            device="cuda:0", model_dtype="float32", torch_module=fake
        ):
            assert snapshot(fake) == outer
        assert snapshot(fake) == outer
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert snapshot(fake) == original
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "process-original"
