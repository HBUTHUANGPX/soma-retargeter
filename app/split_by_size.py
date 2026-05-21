#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按文件夹实际大小将当前目录下的 6 位数字文件夹均匀划分为 8 组（01-08）
使用贪心算法平衡各组总大小
"""

import os
import subprocess
import shutil
from typing import List, Tuple

def get_folder_size(folder: str) -> int:
    """使用 du 获取文件夹总大小（字节）"""
    try:
        result = subprocess.run(
            ['du', '-sb', folder],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.split()[0])
    except Exception as e:
        print(f"警告：无法获取 {folder} 的大小，跳过。错误：{e}")
        return 0

def main():
    # 1. 收集所有符合条件的文件夹（6位数字）
    all_folders = [
        f for f in os.listdir('.')
        if os.path.isdir(f) and f.isdigit() and len(f) == 6
    ]
    all_folders.sort()  # 按名称（时间）排序，便于追溯

    if not all_folders:
        print("未找到符合条件的文件夹（6位数字目录）。")
        return

    print(f"共发现 {len(all_folders)} 个文件夹，正在计算大小...")

    # 2. 获取每个文件夹的大小
    folder_sizes: List[Tuple[str, int]] = []
    for f in all_folders:
        size = get_folder_size(f)
        folder_sizes.append((f, size))

    # 3. 按大小降序排序（贪心算法优化）
    folder_sizes.sort(key=lambda x: x[1], reverse=True)

    # 4. 初始化 8 个组
    groups: List[List[Tuple[str, int]]] = [[] for _ in range(8)]
    totals = [0] * 8

    # 5. 贪心分配：始终分配给当前总大小最小的组
    for folder, size in folder_sizes:
        min_idx = min(range(8), key=lambda i: totals[i])
        groups[min_idx].append((folder, size))
        totals[min_idx] += size

    # 6. 创建目标目录并移动文件夹
    for i in range(8):
        target_dir = f"{i+1:02d}"
        os.makedirs(target_dir, exist_ok=True)

    moved_count = 0
    for i, grp in enumerate(groups):
        target_dir = f"{i+1:02d}"
        for folder, size in grp:
            try:
                shutil.move(folder, target_dir)
                moved_count += 1
                size_mb = size / (1024 * 1024)
                print(f"已移动 {folder} ({size_mb:.2f} MB) → {target_dir}/")
            except Exception as e:
                print(f"移动 {folder} 失败：{e}")

    # 7. 输出最终统计
    print("\n" + "=" * 60)
    print("划分完成！各组总大小统计：")
    total_all = sum(totals)
    for i in range(8):
        size_gb = totals[i] / (1024 ** 3)
        pct = (totals[i] / total_all * 100) if total_all > 0 else 0
        print(f"  {i+1:02d}/  ：{size_gb:7.2f} GB  ({len(groups[i]):3d} 个文件夹，占比 {pct:5.1f}%)")
    print(f"\n总计移动 {moved_count} 个文件夹，总大小 {total_all / (1024**3):.2f} GB")
    print("=" * 60)

if __name__ == "__main__":
    main()