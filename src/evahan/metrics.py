import json
import os
import argparse
from typing import Dict, List, Tuple

# ==========================================
# Part 1: 公共工具函数 (加载映射表 & DP算法)
# ==========================================

def load_variant_map(json_path: str) -> Dict[str, str]:
    """解析 variant_data.json 构建映射表"""
    if not os.path.exists(json_path):
        print(f"⚠️  [Variant] 未找到异体字对照表: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ [Variant] 读取失败: {e}")
        return {}
    
    variant_map = {}
    for char, info in data.items():
        if "standards" in info and info["standards"]:
            standard_char = info["standards"][0]
            if char != standard_char:
                variant_map[char] = standard_char
    return variant_map

def normalize_text(text: str, variant_map: Dict[str, str]) -> str:
    """文本归一化"""
    if not variant_map:
        return text
    return "".join([variant_map.get(c, c) for c in text])

def calculate_edit_operations(ref: str, hyp: str) -> Tuple[int, int, int, int]:
    """
    核心 DP 算法：计算编辑距离 (官方逻辑)
    返回: (edit_dist, del_num, ins_num, rep_num)
    """
    m, n = len(ref), len(hyp)
    dp = [[(0, 0, 0, 0) for _ in range(n + 1)] for __ in range(m + 1)]
    
    for j in range(n + 1): dp[0][j] = (j, 0, j, 0)
    for i in range(m + 1): dp[i][0] = (i, i, 0, 0)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                del_ops = dp[i-1][j]
                ins_ops = dp[i][j-1]
                rep_ops = dp[i-1][j-1]
                
                costs = [del_ops[0] + 1, ins_ops[0] + 1, rep_ops[0] + 1]
                min_cost = min(costs)
                
                if min_cost == costs[0]:   # Delete
                    dp[i][j] = (costs[0], del_ops[1] + 1, del_ops[2], del_ops[3])
                elif min_cost == costs[1]: # Insert
                    dp[i][j] = (costs[1], ins_ops[1], ins_ops[2] + 1, ins_ops[3])
                else:                      # Replace
                    dp[i][j] = (costs[2], rep_ops[1], rep_ops[2], rep_ops[3] + 1)
    return dp[m][n]

# ==========================================
# Part 2: 核心指标计算器 (不含文件IO)
# ==========================================

def compute_metrics(ref: str, hyp: str) -> Dict:
    """
    输入两个字符串（无论是原始的还是归一化的），计算指标
    """
    ref_len, hyp_len = len(ref), len(hyp)
    edit_dist, del_num, ins_num, rep_num = calculate_edit_operations(ref, hyp)
    
    # 1. CER
    cer = edit_dist / ref_len if ref_len > 0 else (1.0 if hyp_len > 0 else 0.0)
    
    # 2. Correct Chars & P/R/F1
    correct_chars = max(0, ref_len - del_num - rep_num)
    precision = correct_chars / hyp_len if hyp_len > 0 else (1.0 if ref_len == 0 else 0.0)
    recall = correct_chars / ref_len if ref_len > 0 else (1.0 if hyp_len == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 3. NED
    max_len = max(ref_len, hyp_len)
    ned = edit_dist / max_len if max_len > 0 else 0.0
    
    # 4. Comprehensive Score
    comp_score = (1 - cer) * 0.5 + f1 * 0.3 + (1 - ned) * 0.2
    
    return {
        "cer": cer, "p": precision, "r": recall, 
        "f1": f1, "ned": ned, "comp": comp_score
    }

# ==========================================
# Part 3: 两套评估流程 (Standard & Variant)
# ==========================================

def evaluate_standard(file_path: str):
    """【流程 A】标准评估：直接比对原始文本"""
    if not os.path.exists(file_path): return None

    totals = {"cer": 0, "p": 0, "r": 0, "f1": 0, "ned": 0, "comp": 0}
    count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                res = compute_metrics(data.get('text', ''), data.get('predicted', ''))
                
                for k in totals: totals[k] += res[k]
                count += 1
            except: continue

    if count == 0: return {"Error": "No Data"}
    
    return {
        "Mode": "Standard (Direct Match)",
        "Comp_Score": round(totals["comp"] / count, 4),
        "F1": round(totals["f1"] / count, 4),
        "CER": round(totals["cer"] / count, 4),
        "NED": round(totals["ned"] / count, 4),
        "Precision": round(totals["p"] / count, 4),
        "Recall": round(totals["r"] / count, 4),
        "Samples": count
    }

def evaluate_variant(file_path: str, variant_map_path: str):
    """【流程 B】归一化评估：先替换异体字，再比对"""
    if not os.path.exists(file_path): return None
    
    # 加载映射表
    variant_map = load_variant_map(variant_map_path)
    if not variant_map:
        return {"Error": "Variant map load failed or empty"}

    totals = {"cer": 0, "p": 0, "r": 0, "f1": 0, "ned": 0, "comp": 0}
    count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                # 关键步骤：先归一化
                ref_norm = normalize_text(data.get('text', ''), variant_map)
                hyp_norm = normalize_text(data.get('predicted', ''), variant_map)
                
                res = compute_metrics(ref_norm, hyp_norm)
                
                for k in totals: totals[k] += res[k]
                count += 1
            except: continue

    if count == 0: return {"Error": "No Data"}

    return {
        "Mode": "Variant Normalized (Glyph Mapping)",
        "Comp_Score": round(totals["comp"] / count, 4),
        "F1": round(totals["f1"] / count, 4),
        "CER": round(totals["cer"] / count, 4),
        "NED": round(totals["ned"] / count, 4),
        "Precision": round(totals["p"] / count, 4),
        "Recall": round(totals["r"] / count, 4),
        "Samples": count
    }

# ==========================================
# Part 4: Main 入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvaHan Double Evaluation System")
    parser.add_argument("--file", type=str, required=True, help="Prediction JSONL file")
    parser.add_argument("--variant", type=str, default="variant_data.json", help="Variant map JSON file")
    
    args = parser.parse_args()
    
    if os.path.exists(args.file):
        print(f"🚀 开始评估文件: {args.file}\n")
        
        # 1. 运行标准评估
        print(">>> 1. 正在进行 [标准评估] (Standard Evaluation)...")
        std_report = evaluate_standard(args.file)
        print(json.dumps(std_report, indent=4, ensure_ascii=False))
        print("-" * 60)
        
        # 2. 运行异体字归一化评估
        print(f">>> 2. 正在进行 [异体字归一化评估(variant_data.json)] (Variant Evaluation)...")
        var_report = evaluate_variant(args.file, args.variant)
        print(json.dumps(var_report, indent=4, ensure_ascii=False))
        
    else:
        print(f"❌ 文件不存在: {args.file}")