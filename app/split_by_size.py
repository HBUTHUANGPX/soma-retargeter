#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按文件夹实际磁盘大小将当前目录下的 6 位数字文件夹均匀划分为 8 组（01-08）
采用贪心算法（Largest-First + 分配至当前最小总大小组）实现大小平衡
支持 --dry-run 模拟执行
"""

import os
import subprocess
import shutil
import argparse
from typing import List, Tuple

def get_folder_size(folder: str) -> int:
    """使用 du 获取文件夹总大小（字节），失败返回 0"""
    try:
        result = subprocess.run(
            ['du', '-sb', folder],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.split()[0])
    except Exception as e:
        print(f"警告：无法获取文件夹 {folder} 的大小，跳过。错误信息：{e}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description="按实际大小将 6 位数字文件夹划分为 8 个大小均衡的组（01-08）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python split_by_size.py                # 直接执行移动
  python split_by_size.py --dry-run      # 仅模拟，打印分配方案
  python split_by_size.py -n             # 同上（简写）
        """
    )
    parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='模拟执行模式：仅打印分配方案和预计大小，不实际移动文件'
    )
    args = parser.parse_args()

    # 1. 收集所有符合条件的文件夹
    all_folders = [
        f for f in os.listdir('.')
        if os.path.isdir(f) and f.isdigit() and len(f) == 6
    ]
    all_folders.sort()

    if not all_folders:
        print("未找到符合条件的文件夹（6位数字目录）。")
        return

    print(f"共发现 {len(all_folders)} 个文件夹，正在计算磁盘占用大小...")

    # 2. 获取大小并按大小降序排序（贪心优化）
    folder_sizes: List[Tuple[str, int]] = [(f, get_folder_size(f)) for f in all_folders]
    folder_sizes.sort(key=lambda x: x[1], reverse=True)

    # 3. 初始化 8 个组并进行贪心分配
    groups: List[List[Tuple[str, int]]] = [[] for _ in range(8)]
    totals = [0] * 8

    for folder, size in folder_sizes:
        min_idx = min(range(8), key=lambda i: totals[i])
        groups[min_idx].append((folder, size))
        totals[min_idx] += size

    # 4. 创建目标目录（dry-run 时也创建以便预览结构，但不移动）
    for i in range(8):
        target_dir = f"{i+1:02d}"
        os.makedirs(target_dir, exist_ok=True)

    # 5. 执行移动或模拟输出
    action_word = "将移动" if args.dry_run else "已移动"
    moved_count = 0

    for i, grp in enumerate(groups):
        target_dir = f"{i+1:02d}"
        for folder, size in grp:
            size_mb = size / (1024 * 1024)
            print(f"{action_word} {folder} ({size_mb:.2f} MB) → {target_dir}/")
            if not args.dry_run:
                try:
                    shutil.move(folder, target_dir)
                    moved_count += 1
                except Exception as e:
                    print(f"  移动失败：{e}")

    # 6. 输出最终统计
    print("\n" + "=" * 65)
    print("划分结果统计（各组总大小已尽量均衡）：")
    total_all = sum(totals)
    for i in range(8):
        size_gb = totals[i] / (1024 ** 3)
        pct = (totals[i] / total_all * 100) if total_all > 0 else 0
        folder_count = len(groups[i])
        print(f"  {i+1:02d}/   {size_gb:7.2f} GB   ({folder_count:3d} 个文件夹，占比 {pct:5.1f}%)")

    if args.dry_run:
        print(f"\n【模拟模式】共规划移动 {len(all_folders)} 个文件夹，总大小 {total_all / (1024**3):.2f} GB")
        print("未执行实际移动操作。如确认无误，请去掉 --dry-run 参数重新运行。")
    else:
        print(f"\n实际移动完成！共移动 {moved_count} 个文件夹，总大小 {total_all / (1024**3):.2f} GB")

    print("=" * 65)

if __name__ == "__main__":
    main()