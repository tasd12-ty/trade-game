"""Parameter loading for current trade_game experiments.

用法:
    from trade_game.params import load_io_params
    params_raw = load_io_params("CN2017", "US2017")  # H=中国, F=美国
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import ParamsSource
from .runtime.model import create_symmetric_parameters

PACKAGE_DIR = Path(__file__).resolve().parent
# 默认读取随包发布的数据；如需复用外部数据目录，可设置 TRADE_GAME_DATA_DIR 覆盖。
BASE_DIR = os.environ.get("TRADE_GAME_DATA_DIR", str(PACKAGE_DIR / "io-final"))


def _read_matrix_csv(filepath: str) -> np.ndarray:
    """读取 n×n 矩阵 CSV（第一列为行标签，第一行为列标签）。"""
    rows = []
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            rows.append([float(x) for x in row[1:]])
    return np.array(rows)


def _read_vector_csv(filepath: str) -> np.ndarray:
    """读取向量 CSV（两列：标签, 值）。"""
    vals = []
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            vals.append(float(row[1]))
    return np.array(vals)


def _load_country_block(data_dir: str) -> Dict[str, np.ndarray]:
    """从一个国家目录加载全部参数。"""
    alpha_ij = _read_matrix_csv(os.path.join(data_dir, "alpha_ij.csv"))
    gamma_ij = _read_matrix_csv(os.path.join(data_dir, "gamma_ij.csv"))
    rho_ij = _read_matrix_csv(os.path.join(data_dir, "rho_ij.csv"))
    beta_j = _read_vector_csv(os.path.join(data_dir, "beta_j.csv"))
    A_i = _read_vector_csv(os.path.join(data_dir, "A_i.csv"))
    E_j = _read_vector_csv(os.path.join(data_dir, "E_j.csv"))
    gamma_cj = _read_vector_csv(os.path.join(data_dir, "gamma_cj.csv"))
    rho_cj = _read_vector_csv(os.path.join(data_dir, "rho_cj.csv"))
    import_cost = _read_vector_csv(os.path.join(data_dir, "import_cost.csv"))

    return {
        "alpha_ij": alpha_ij,
        "gamma_ij": gamma_ij,
        "rho_ij": rho_ij,
        "beta_j": beta_j,
        "A_i": A_i,
        "Export_i": E_j,
        "gamma_cj": gamma_cj,
        "rho_cj": rho_cj,
        "import_cost": import_cost,
    }


def load_io_params(
    home_id: str = "CN2017",
    foreign_id: str = "US2017",
    tradable_sectors: Optional[List[int]] = None,
) -> Dict:
    """加载双国参数，返回与 bootstrap_simulator 兼容的 params_raw 字典。

    Args:
        home_id: H 国目录名，如 "CN2017", "CN2018"
        foreign_id: F 国目录名，如 "US2017", "US2018"
        tradable_sectors: 可贸易部门索引列表。默认全部5个部门均可贸易。
    """
    home_dir = os.path.join(BASE_DIR, home_id)
    foreign_dir = os.path.join(BASE_DIR, foreign_id)

    if not os.path.isdir(home_dir):
        raise FileNotFoundError(f"Home country dir not found: {home_dir}")
    if not os.path.isdir(foreign_dir):
        raise FileNotFoundError(f"Foreign country dir not found: {foreign_dir}")

    home_block = _load_country_block(home_dir)
    foreign_block = _load_country_block(foreign_dir)

    n = home_block["alpha_ij"].shape[0]
    if tradable_sectors is None:
        tradable_sectors = list(range(n))  # 全部可贸易

    return {
        "H": home_block,
        "F": foreign_block,
        "tradable_sectors": tradable_sectors,
    }


def load_params(source: ParamsSource) -> Dict:
    """Load parameters from a configured source."""
    if source.type == "symmetric":
        return create_symmetric_parameters()
    if source.type == "io_final":
        return load_io_params(
            home_id=source.home_id,
            foreign_id=source.foreign_id,
            tradable_sectors=list(source.tradable_sectors) if source.tradable_sectors else None,
        )
    raise ValueError(f"Unknown params_source type: {source.type}")


if __name__ == "__main__":
    # 测试加载
    params = load_io_params("CN2017", "US2017")
    print("Loaded params_raw:")
    print(f"  H alpha_ij shape: {params['H']['alpha_ij'].shape}")
    print(f"  F alpha_ij shape: {params['F']['alpha_ij'].shape}")
    print(f"  tradable_sectors: {params['tradable_sectors']}")
    print(f"  H beta_j: {params['H']['beta_j']}")
    print(f"  F beta_j: {params['F']['beta_j']}")
    print(f"  H Export_i: {params['H']['Export_i']}")
    print(f"  F Export_i: {params['F']['Export_i']}")


__all__ = ["load_io_params", "load_params"]
