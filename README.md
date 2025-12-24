# 楽曲ログのアーティスト名名寄せシステム

Apple Music, Spotify, YouTubeなど複数チャネルから取得した楽曲ログにおいて、表記揺れのあるアーティスト名を統一するシステムです。

## 機能

- サンプルデータの自動生成（日本の人気アーティストの表記揺れを含む）
- 最小限の前処理（アーティスト名の重要な部分を保持）
- 複数の距離メトリクスによる候補ペア抽出
  - レーベンシュタイン距離
  - BM25スコア
- Gemini APIを使用した高精度な名寄せ判定
- アーティスト名マスタデータの自動生成

## セットアップ

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2. Gemini API キーの設定

```bash
export GEMINI_API_KEY="your_api_key_here"
```

## 実行方法

```bash
python artist_deduplication.py
```

## 出力ファイル

スクリプトを実行すると、以下の3つのファイルが生成されます：

1. **sample_artists.csv** - 表記揺れを含むサンプル楽曲ログデータ（100件）
2. **candidate_pairs.json** - 距離メトリクスで抽出された候補ペア
3. **artist_master.json** - Gemini APIで判定された正規化アーティスト名マスタ

## サンプルデータに含まれるアーティスト

- YOASOBI（表記揺れ例: YoOSBI, YOASOBI ビデオ版あり）
- BTS（表記揺れ例: B T S, B.T.S., 防弾少年団）
- Ado（表記揺れ例: Ado (Official), Ado【公式】）
- BABYMETAL（表記揺れ例: ＢＡＢＹＭＥＴＡＬ, Baby Metal）
- Official髭男dism（表記揺れ例: ヒゲダン, 髭男dism）
- 米津玄師（表記揺れ例: Kenshi Yonezu, よねづけんし）
- その他、日本の人気アーティスト多数

## パラメータ調整

`artist_deduplication.py` の冒頭で以下のパラメータを調整できます：

```python
LEVENSHTEIN_THRESHOLD = 5  # レーベンシュタイン距離の閾値
BM25_THRESHOLD = 0.5       # BM25スコアの閾値
SAMPLE_SIZE = 100          # サンプルデータ件数
```

## 処理フロー

1. サンプルデータ生成 → `sample_artists.csv` 出力
2. データ読み込み
3. 前処理適用（空白削除、付加情報削除）
4. 距離計算実行（レーベンシュタイン、BM25）
5. 候補ペア抽出 → `candidate_pairs.json` 出力
6. Gemini API呼び出し → 表記揺れ判定
7. マスタデータ生成 → `artist_master.json` 出力

## 注意事項

- Gemini API キーが必要です
- APIのレート制限に注意してください
- サンプルサイズが大きいと処理時間が長くなります

## ライセンス

MIT
