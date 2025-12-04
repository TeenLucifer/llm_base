from pathlib import Path
import yaml

# 默认项目名：取当前文件所在目录名，便于复制到其他项目直接复用
PROJECT_NAME = Path(__file__).resolve().parent.name


def _config_path() -> Path:
    """返回仓库根目录的 config.yaml 路径。"""
    root = Path(__file__).resolve().parents[1]
    cfg = root / "config.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"未找到配置文件: {cfg}")
    return cfg


def _load_project_config() -> dict:
    """读取并解析当前项目配置，路径转为绝对路径。"""
    cfg_path = _config_path()
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if PROJECT_NAME not in data:
        raise KeyError(f"config.yaml 中未找到项目配置: {PROJECT_NAME}")

    project_cfg = data[PROJECT_NAME] or {}
    base_dir = cfg_path.parent

    resolved = {}
    for key, value in project_cfg.items():
        p = Path(value)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        resolved[key] = str(p)

    resolved["config_path"] = str(cfg_path)
    return resolved


# 模块导入即加载配置，其他文件可直接使用 CONFIG
CONFIG = _load_project_config()

__all__ = ["CONFIG", "PROJECT_NAME"]
