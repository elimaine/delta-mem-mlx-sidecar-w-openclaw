from __future__ import annotations

import argparse

from delta_mem_sidecar.mlx_delta_adapter import convert_torch_adapter_to_mlx_npz


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Torch delta-Mem adapter checkpoint to an MLX-native NPZ artifact."
    )
    parser.add_argument("adapter_dir", help="Directory containing delta_mem_config.json and delta_mem_adapter.pt.")
    parser.add_argument(
        "--output",
        help="Output NPZ path. Defaults to <adapter_dir>/delta_mem_adapter_mlx.npz.",
    )
    args = parser.parse_args()

    output_path = convert_torch_adapter_to_mlx_npz(args.adapter_dir, output_path=args.output)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
