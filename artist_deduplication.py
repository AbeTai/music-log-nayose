#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
楽曲ログのアーティスト名名寄せシステム
"""

import os
import json
import re
import random
from typing import List, Dict, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.sparse import dok_matrix
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types

# ==================== パラメータ設定 ====================
SAMPLE_DATA_PATH = "sample_artists.csv"
CANDIDATE_PAIRS_PATH = "candidate_pairs.json"
ARTIST_MASTER_PATH = "artist_master.json"

# 閾値設定
LEVENSHTEIN_THRESHOLD = 5  # レーベンシュタイン距離の閾値
BM25_THRESHOLD = 0.5  # BM25スコアの閾値

# サンプルデータ件数
SAMPLE_SIZE = 100

# ==================== サンプルデータ生成 ====================

def generate_sample_data():
    """表記揺れのあるアーティスト名のサンプルデータを生成"""
    print("=== サンプルデータ生成中 ===")

    # アーティスト名の表記揺れパターン
    artist_variants = {
        "YOASOBI": ["YOASOBI", "YoASOBI", "YoOSBI", "YOASOBI ビデオ版あり", "Yoasobi"],
        "BTS": ["BTS", "B T S", "B.T.S.", "BTS (Official)", "防弾少年団"],
        "Ado": ["Ado", "Ado (Official)", "Ado【公式】", "ado", "ADO"],
        "BABYMETAL": ["BABYMETAL", "ＢＡＢＹＭＥＴＡＬ", "Baby Metal", "BabyMetal"],
        "Official髭男dism": ["Official髭男dism", "Official HIGE DANdism", "ヒゲダン", "髭男dism"],
        "米津玄師": ["米津玄師", "米津 玄師", "よねづけんし", "Kenshi Yonezu"],
        "あいみょん": ["あいみょん", "Aimyon", "aImyon", "あいみょん (Official)"],
        "King Gnu": ["King Gnu", "KingGnu", "King  Gnu", "king gnu", "King Gnu Official"],
        "RADWIMPS": ["RADWIMPS", "Radwimps", "RAD WIMPS", "RADWIMPS (Official)"],
        "ONE OK ROCK": ["ONE OK ROCK", "One Ok Rock", "ONE OK  ROCK", "ONEOKROCK"],
        "乃木坂46": ["乃木坂46", "乃木坂４６", "Nogizaka46", "nogizaka46"],
        "櫻坂46": ["櫻坂46", "櫻坂４６", "Sakurazaka46", "sakurazaka46"],
        "Mrs. GREEN APPLE": ["Mrs. GREEN APPLE", "Mrs.GREEN APPLE", "Mrs GREEN APPLE", "ミセス"],
        "Aimer": ["Aimer", "aimer", "Aimer (Official)", "エメ"],
        "LiSA": ["LiSA", "Lisa", "LISA", "LiSA (Official)", "LiSA ビデオ版あり"],
        "星野源": ["星野源", "星野 源", "Gen Hoshino", "Hoshino Gen"],
        "THE FIRST TAKE": ["THE FIRST TAKE", "The First Take", "THEFIRSTTAKE", "ファーストテイク"],
        "back number": ["back number", "backnumber", "Back Number", "back  number"],
        "Perfume": ["Perfume", "perfume", "PERFUME", "Perfume (Official)"],
        "BUMP OF CHICKEN": ["BUMP OF CHICKEN", "Bump Of Chicken", "BUMPOFCHICKEN", "バンプ"],
    }

    channels = ["Apple Music", "Spotify", "YouTube"]

    records = []
    record_id = 1

    # 各アーティストの表記揺れを複数回サンプリング
    for artist_canonical, variants in artist_variants.items():
        # 各バリアントを1-3回ランダムに追加
        for variant in variants:
            num_records = random.randint(1, 3)
            for _ in range(num_records):
                channel = random.choice(channels)
                records.append({
                    "id": record_id,
                    "artist_name": variant,
                    "channel": channel,
                    "play_count": random.randint(1, 100)
                })
                record_id += 1

    # データフレーム作成
    df = pd.DataFrame(records)

    # サンプルサイズに調整（足りない場合は重複、多い場合はサンプリング）
    if len(df) < SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, replace=True, random_state=42).reset_index(drop=True)
    else:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    # ID振り直し
    df['id'] = range(1, len(df) + 1)

    # CSV出力
    df.to_csv(SAMPLE_DATA_PATH, index=False, encoding='utf-8')
    print(f"サンプルデータを {SAMPLE_DATA_PATH} に出力しました ({len(df)}件)")
    print(f"ユニークなアーティスト名: {df['artist_name'].nunique()}件")

    return df


# ==================== 前処理 ====================

def preprocess_artist_name(name: str) -> str:
    """アーティスト名の前処理（最小限）"""
    # 前後の空白削除
    name = name.strip()

    # 「ビデオ版あり」などの明確な付加情報を削除
    patterns_to_remove = [
        r'\s*ビデオ版あり\s*$',
        r'\s*-\s*Topic\s*$',
    ]

    for pattern in patterns_to_remove:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)

    return name.strip()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """データフレーム全体の前処理"""
    print("\n=== 前処理実行中 ===")
    df = df.copy()
    df['artist_name_preprocessed'] = df['artist_name'].apply(preprocess_artist_name)
    print(f"前処理完了: {len(df)}件")
    return df


# ==================== ブロッキング ====================

def normalize_for_blocking(name: str) -> str:
    """ブロッキング用に名前を正規化"""
    # 全角→半角、小文字化、空白削除
    import unicodedata
    name = unicodedata.normalize('NFKC', name)
    name = name.lower()
    name = re.sub(r'\s+', '', name)
    return name


def create_blocks(unique_names: List[str]) -> Dict[str, List[Tuple[int, str]]]:
    """アーティスト名をブロックに分割（効率化のため）"""
    print("ブロッキング中...")
    blocks = defaultdict(list)

    for idx, name in enumerate(unique_names):
        # 複数のブロッキングキーを生成
        normalized = normalize_for_blocking(name)

        # ブロッキングキー1: 最初の2文字
        if len(normalized) >= 2:
            key1 = normalized[:2]
            blocks[key1].append((idx, name))

        # ブロッキングキー2: 最初の1文字
        if len(normalized) >= 1:
            key2 = normalized[0]
            blocks[key2].append((idx, name))

        # ブロッキングキー3: 長さの範囲（±3文字以内で比較）
        length = len(name)
        length_key = f"len_{length // 3}"
        blocks[length_key].append((idx, name))

    print(f"ブロック数: {len(blocks)}個")
    return blocks


# ==================== 距離計算 ====================

def tokenize_artist_name(name: str) -> List[str]:
    """アーティスト名を文字単位でトークン化"""
    # 空白、記号も含めて文字単位で分割
    return list(name)


def build_distance_matrix(blocks: Dict[str, List[Tuple[int, str]]], n: int) -> dok_matrix:
    """レーベンシュタイン距離行列を構築"""
    print(f"レーベンシュタイン距離行列を構築中... (サイズ: {n}x{n})")
    mat = dok_matrix((n, n), dtype="int16")

    total_comparisons = 0
    for block_key, block in blocks.items():
        idxs, strs = zip(*block)
        for i in range(len(strs)):
            for j in range(i + 1, len(strs)):
                d = Levenshtein.distance(strs[i], strs[j])
                mat[idxs[i], idxs[j]] = d
                mat[idxs[j], idxs[i]] = d
                total_comparisons += 1

    print(f"レーベンシュタイン距離計算完了: {total_comparisons}ペア")
    return mat


def build_bm25_matrix(unique_names: List[str], n: int) -> np.ndarray:
    """BM25スコア行列を構築"""
    print(f"BM25スコア行列を構築中... (サイズ: {n}x{n})")

    # 全アーティスト名をトークン化
    corpus_tokenized = [tokenize_artist_name(name) for name in unique_names]

    # BM25モデル構築
    bm25 = BM25Okapi(corpus_tokenized)

    # スコア行列を初期化
    bm25_matrix = np.zeros((n, n), dtype=np.float32)

    # 各アーティスト名をクエリとして全体にスコアリング
    for i, name in enumerate(unique_names):
        tokenized_query = tokenize_artist_name(name)
        scores = bm25.get_scores(tokenized_query)
        bm25_matrix[i, :] = scores

    print(f"BM25スコア計算完了")
    return bm25_matrix


# ==================== 候補ペア抽出 ====================

def extract_candidate_pairs(df: pd.DataFrame) -> List[Dict]:
    """距離メトリクスに基づいて候補ペアを抽出"""
    print("\n=== 候補ペア抽出中 ===")

    unique_names = df['artist_name_preprocessed'].unique().tolist()
    n = len(unique_names)
    print(f"ユニークなアーティスト名: {n}件")

    # 1. ブロッキング
    blocks = create_blocks(unique_names)

    # 2. レーベンシュタイン距離行列の構築
    lev_matrix = build_distance_matrix(blocks, n)

    # 3. BM25スコア行列の構築
    bm25_matrix = build_bm25_matrix(unique_names, n)

    # 4. 候補ペアの抽出
    print("\n候補ペアを抽出中...")
    candidate_pairs = []
    pair_set = set()  # 重複チェック用

    # 距離行列から閾値を満たすペアを抽出
    for i in range(n):
        for j in range(i + 1, n):
            # レーベンシュタイン距離を取得（計算されていない場合は大きな値）
            lev_dist = lev_matrix[i, j] if (i, j) in lev_matrix else 9999

            # BM25スコアを取得（双方向の最大値）
            bm25_score = max(bm25_matrix[i, j], bm25_matrix[j, i])

            # 閾値チェック
            if lev_dist <= LEVENSHTEIN_THRESHOLD or bm25_score >= BM25_THRESHOLD:
                name1 = unique_names[i]
                name2 = unique_names[j]

                pair_key = tuple(sorted([name1, name2]))
                if pair_key not in pair_set:
                    pair_set.add(pair_key)
                    candidate_pairs.append({
                        "name1": name1,
                        "name2": name2,
                        "levenshtein_distance": int(lev_dist),
                        "bm25_score": float(bm25_score)
                    })

    print(f"抽出された候補ペア: {len(candidate_pairs)}件")

    # JSON出力
    with open(CANDIDATE_PAIRS_PATH, 'w', encoding='utf-8') as f:
        json.dump(candidate_pairs, f, ensure_ascii=False, indent=2)
    print(f"候補ペアを {CANDIDATE_PAIRS_PATH} に出力しました")

    return candidate_pairs


# ==================== Gemini API統合 ====================

def call_gemini_for_deduplication(candidate_pairs: List[Dict]) -> str:
    """Gemini APIを呼び出してアーティスト名の名寄せを実行"""
    print("\n=== Gemini API呼び出し中 ===")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("環境変数 GEMINI_API_KEY が設定されていません")

    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash-exp"

    # 候補ペアを整形
    pairs_text = ""
    for idx, pair in enumerate(candidate_pairs, 1):
        pairs_text += f"{idx}. \"{pair['name1']}\" ↔ \"{pair['name2']}\" "
        pairs_text += f"(Levenshtein: {pair['levenshtein_distance']}, BM25: {pair['bm25_score']:.2f})\n"

    # プロンプト作成
    prompt = f"""以下のアーティスト名ペアのリストを確認し、同じアーティストの表記揺れと判断できるペアを抽出してください。

候補ペア ({len(candidate_pairs)}件):
{pairs_text}

同じアーティストと判断できるペアについて、以下のJSON形式で出力してください。
必ずJSONのみを出力し、説明文は含めないでください。

{{
  "matches": [
    {{
      "canonical_name": "正規化後の名前（最も一般的な表記を選択）",
      "variants": ["表記1", "表記2", "表記3", ...]
    }}
  ]
}}

注意事項:
- 明らかに同じアーティストのみをマッチングしてください
- 正規化後の名前は、最も公式に近い、または一般的な表記を選んでください
- variantsには正規化後の名前も含めてください
"""

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,  # 一貫性を高めるため低めに設定
    )

    print("Gemini APIにリクエスト送信中...")
    response_text = ""

    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response_text += chunk.text
                print(chunk.text, end="", flush=True)

        print("\n")
        return response_text

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        raise


def parse_gemini_response(response_text: str) -> Dict:
    """Gemini APIのレスポンスをパースしてJSON形式で返す"""
    print("\n=== Geminiレスポンスのパース中 ===")

    # JSONブロックを抽出（```json ... ```で囲まれている場合）
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # JSONブロックがない場合、レスポンス全体からJSONを抽出
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError("レスポンスからJSONを抽出できませんでした")

    try:
        result = json.loads(json_str)
        print(f"パース成功: {len(result.get('matches', []))}件のマッチを検出")
        return result
    except json.JSONDecodeError as e:
        print(f"JSONパースエラー: {e}")
        print(f"パース対象文字列: {json_str[:200]}...")
        raise


# ==================== マスタデータ生成 ====================

def generate_artist_master(gemini_result: Dict) -> Dict:
    """Geminiの結果からアーティストマスタを生成"""
    print("\n=== アーティストマスタ生成中 ===")

    master_data = {
        "version": "1.0",
        "generated_at": pd.Timestamp.now().isoformat(),
        "artist_mappings": []
    }

    matches = gemini_result.get("matches", [])

    for match in matches:
        canonical_name = match.get("canonical_name", "")
        variants = match.get("variants", [])

        if canonical_name and variants:
            mapping = {
                "canonical_name": canonical_name,
                "variants": variants,
                "variant_count": len(variants)
            }
            master_data["artist_mappings"].append(mapping)

    # JSON出力
    with open(ARTIST_MASTER_PATH, 'w', encoding='utf-8') as f:
        json.dump(master_data, f, ensure_ascii=False, indent=2)

    print(f"アーティストマスタを {ARTIST_MASTER_PATH} に出力しました")
    print(f"正規化されたアーティスト数: {len(master_data['artist_mappings'])}件")

    return master_data


# ==================== メイン処理 ====================

def main():
    """メイン処理"""
    print("=" * 60)
    print("楽曲ログのアーティスト名名寄せシステム")
    print("=" * 60)

    # 1. サンプルデータ生成
    df = generate_sample_data()

    # 2. データ読み込み（生成したデータをそのまま使用）
    print("\n=== データ読み込み ===")
    print(f"読み込んだデータ: {len(df)}件")
    print(df.head())

    # 3. 前処理
    df = preprocess_data(df)

    # 4. 候補ペア抽出
    candidate_pairs = extract_candidate_pairs(df)

    if len(candidate_pairs) == 0:
        print("\n候補ペアが見つかりませんでした。閾値を調整してください。")
        return

    # 5. Gemini API呼び出し
    try:
        response_text = call_gemini_for_deduplication(candidate_pairs)

        # 6. レスポンスのパース
        gemini_result = parse_gemini_response(response_text)

        # 7. マスタデータ生成
        master_data = generate_artist_master(gemini_result)

        print("\n" + "=" * 60)
        print("処理完了！")
        print("=" * 60)
        print(f"出力ファイル:")
        print(f"  - {SAMPLE_DATA_PATH}")
        print(f"  - {CANDIDATE_PAIRS_PATH}")
        print(f"  - {ARTIST_MASTER_PATH}")

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
