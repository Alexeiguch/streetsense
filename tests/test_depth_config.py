from streetsense.config import Config, DepthConfig


def test_depth_off_by_default():
    cfg = Config()
    assert cfg.depth.enabled is False
    assert cfg.depth.every_n_frames == 6
    assert cfg.depth.backend == "openvino"


def test_depth_config_fields():
    d = DepthConfig(enabled=True, every_n_frames=3, backend="torch")
    assert d.enabled and d.every_n_frames == 3 and d.backend == "torch"
