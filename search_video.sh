#!/bin/bash
#
# search_video.sh — 视频场景分割 + 文本搜索 (一键脚本)
#
# 用法:
#   ./search_video.sh <视频文件> "搜索文本" [top_k]
#
# 示例:
#   ./search_video.sh demo.mp4 "一个人在走路"
#   ./search_video.sh demo.mp4 "汽车" 3
#
# 所有输出均保存在 test_run/<视频名>/ 下
#

set -e

# ── 参数 ──────────────────────────────────────────────
VIDEO="$1"
QUERY="$2"
TOP_K="${3:-5}"

if [ -z "$VIDEO" ] || [ -z "$QUERY" ]; then
    echo "用法: $0 <视频文件> \"搜索文本\" [top_k]"
    echo "示例: $0 demo.mp4 \"一个人在走路\""
    exit 1
fi

if [ ! -f "$VIDEO" ]; then
    echo "错误: 文件不存在: $VIDEO"
    exit 1
fi

# 视频文件名（不含扩展名）
VIDEO_STEM=$(basename "$VIDEO")
VIDEO_STEM="${VIDEO_STEM%.*}"

# ── 新建测试目录（不在已有项目目录里） ──────────────
TEST_DIR="./test_run/$VIDEO_STEM"
SEGMENT_DIR="$TEST_DIR/scenes"    # 分割后的帧存这里
RESULT_DIR="$TEST_DIR/results"    # 搜索结果缩略图存这里

mkdir -p "$SEGMENT_DIR" "$RESULT_DIR"

# ── Step 1: 分割场景 ─────────────────────────────────
echo ""
echo "========================================"
echo " Step 1: 场景分割"
echo "========================================"
echo " 视频: $VIDEO"
echo " 保存到: $SEGMENT_DIR/"
echo ""

python scripts/segment_video.py "$VIDEO" -o "$SEGMENT_DIR"

# ── Step 2: 搜索 ─────────────────────────────────────
echo ""
echo "========================================"
echo " Step 2: 文本搜索"
echo "========================================"
echo " 查询: \"$QUERY\""
echo ""

python run.py "$SEGMENT_DIR/$VIDEO_STEM" -q "$QUERY" -k "$TOP_K" -o "$RESULT_DIR"

echo ""
echo "========================================"
echo " 完成！"
echo " 场景帧: $SEGMENT_DIR/$VIDEO_STEM/"
echo " 搜索结果: $RESULT_DIR/"
echo "========================================"
echo ""
