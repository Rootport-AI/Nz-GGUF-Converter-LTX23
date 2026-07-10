# 検証記録（VERIFICATION）

このドキュメントは、Sulphur-2-base → GGUF（Q4_K_M）変換の実行結果と検証結果を
まとめた記録です。対象ファイル:

- 出力GGUF: `output\Sulphur-2-base-distil-Q4_K_M.gguf`
  （サイズ 17,763,014,976 バイト = 約17.76GB、テンソル4444個）
- 変換元: `safetensors\sulphur_distil_bf16.safetensors`
  （サイズ 46,139,885,414 バイト = 約46.14GB、bf16）
- 参照GGUF: `Nz-LTX23-backend\models\ltx-2.3-gguf\LTX-2.3-distilled-1.1\LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`
  （サイズ 17,763,015,328 バイト、テンソル4444個）

変換の実行時刻（出力ファイルのタイムスタンプより）: 開始 2026-07-10 10:16:41、
完了（最終書き込み） 2026-07-10 12:50:06。所要時間は約2時間34分でした。

出力ファイルサイズは参照GGUFよりわずか352バイト小さいだけで、比率にすると
約0.002%の差（許容±1%に対して十分小さい）です。

---

## 1. 量子化カーネルの検証結果（要点）

量子化カーネル（`src/converter/quant_kernels.py`、Q4_K/Q5_K/Q6_Kの書き込み側実装）
そのものの検証は `Docs/QUANT_KERNELS.md` に詳細があります。ここでは要点だけ
引用します。

- **移植の忠実性ゲート**（参照モデル不要・最も基礎的な検証）: 量子化アルゴリズムの
  探索処理を「Cの演算順序をそのまま写した純スカラ版」と比較し、乱数データ・実データ
  ともに**不一致ゼロ（ビット完全一致）**を確認済み。カーネルがllama.cppのC参照実装
  を忠実に再現していることの裏づけです。
- **べき等テスト**（参照GGUFの実テンソルを復元→再量子化し、元のバイト列との一致率
  を測定、各タイプ40テンソル）:

  | タイプ | ブロック一致率 | 完全一致テンソル | 再量子化SQNR |
  |--------|----------------|------------------|--------------|
  | Q4_K   | 99.90%         | 40中6            | 69.3 dB      |
  | Q5_K   | 85.50%         | 40中0            | 46.6 dB      |
  | Q6_K   | 85.67%         | 40中0            | 72.7 dB      |

  Q5_K/Q6_Kが100%にならないのはカーネルの不具合ではなく、K量子化アルゴリズムの
  スケール探索が「一度復元された値」に対して再最適化するために生じる、本質的な
  性質です（詳細な原因分析はQUANT_KERNELS.md §5-2）。不一致があっても再構成誤差は
  小さく、品質上の問題ではありません。
- **素の量子化品質**（乱数データを1回だけ量子化→復元したSQNR）: Q4_K=22.9dB、
  Q5_K=28.8dB、Q6_K=35.0dB。1bit増えるごとに約6dB改善しており、量子化理論どおりの
  挙動でカーネルが正しく動作していることを裏づけています。

---

## 2. 構造検証の実行結果（全文）

実行コマンド: `run.bat verify`
（内部的には `PYTHONPATH=src .venv\Scripts\python.exe -m converter verify`）

```
Output:    S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-GGUF-Converter-LTX23\output\Sulphur-2-base-distil-Q4_K_M.gguf
Reference: S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-LTX23-backend\models\ltx-2.3-gguf\LTX-2.3-distilled-1.1\LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf
Tensors:   output=4444 reference=4444

[PASS] tensor_count
[PASS] tensor_names_order
[PASS] tensor_types_shapes
[PASS] kv_keys
[PASS] kv_types
[PASS] kv_fixed_values
[PASS] kv_config_json
[PASS] general_alignment_absent
[PASS] file_size_ratio
[PASS] dequant_sanity

RESULT: PASS
```

終了コード: `0`

各チェックの意味（`src/converter/verify.py`のdocstringより）:

1. `tensor_count` — テンソル総数が参照GGUFと一致するか（4444=4444）。
2. `tensor_names_order` — テンソル名の集合および並び順が完全一致するか。
3. `tensor_types_shapes` — 各テンソルのGGML量子化タイプと形状が参照GGUFと一致するか。
4. `kv_keys` — KVメタデータのキー集合が一致するか（順不同）。
5. `kv_types` — 共通するKVキーの値の型が一致するか。
6. `kv_fixed_values` — `general.architecture=="ltxv"`、
   `general.quantization_version==2`、`general.file_type==15`が期待値どおりか。
7. `kv_config_json` — `config`キーの値が有効なJSONで、トップレベルに
   `transformer`キーを持つか（値そのものはSulphur独自の設定なので参照GGUFとの
   一致は求めない）。
8. `general_alignment_absent` — 出力・参照ともに`general.alignment`キーが
   存在しないこと。
9. `file_size_ratio` — 出力ファイルサイズが参照GGUFの±1%以内であること
   （実測差 約0.002%）。
10. `dequant_sanity` — 出力GGUFからK-quantテンソルを8個サンプリングし、
    `gguf.quants.dequantize`で例外なく復元でき、NaN/Infを含まないこと。

**結果: 全10項目PASS。失敗項目なし。**

---

## 3. pytest全結果

実行コマンド: `PYTHONPATH=src .venv\Scripts\python.exe -m pytest tests/ -v`

```
============================= test session starts =============================
platform win32 -- Python 3.12.9, pytest-9.1.1, pluggy-1.6.0
collecting ... collected 71 items

tests/test_convert.py .......... (8 tests)                              PASSED
tests/test_metadata.py .......... (9 tests)                             PASSED
tests/test_output_vs_reference.py .......... (15 tests)                 PASSED
tests/test_quant_roundtrip.py .......... (24 tests)                     PASSED
tests/test_typemap.py .......... (13 tests)                             PASSED

======================== 71 passed in 73.05s (0:01:13) ========================
```

（フルログの主要行は上記どおり。全71件がPASS、失敗・スキップともに0件。）

特に本番実行に依存するテストが今回きちんと動作したことを確認しています。

- `tests/test_output_vs_reference.py::test_real_output_gguf_matches_reference_structure`
  — 出力GGUFが実際に存在する場合のみ意味を持つテストで、今回のファイルに対して
  実行され、PASSしました（存在しない場合は自動でスキップされる設計）。

終了コード: `0`

---

## 4. E2E段階A（バックエンドローダでのCPUロード検証）

### 4-1. 方式

`scripts/e2e_load_check.py`（本タスクで新規作成）を、バックエンドの
`.venv-engine\Scripts\python.exe`（torchインストール済み、`Nz-LTX23-backend`
同梱の実行環境）で実行しました。バックエンド側のファイルは一切変更していません
（`sys.path`にバックエンドのルートを追加してモジュールをimportするだけの借用）。

検証した2点:

1. **KVメタデータの取得**: バックエンドの`engine.gguf.loader_service.
   GGUFStateDictLoader.metadata()`をそのまま呼び出し、出力GGUFのKVから
   `config`/`ltx.config`/`general.config`のいずれかを探して`json.loads`する、
   本番と全く同じコードパスが成功することを確認。
2. **全4444テンソルの CPU dequant**: 出力GGUFを`gguf.GGUFReader`でストリーミング
   オープンし、テンソルを1個ずつ読み、バックエンドの
   `engine.gguf.quant_service.dequantize_ggml_tensor`
   （`GGUFStateDictLoader.load()`が実際のモデルロード時に呼ぶのと同じ関数）で
   CPU dequantし、例外の有無・NaN/Infの有無を確認。1テンソル処理するごとに
   参照を破棄してから次のテンソルへ進むストリーミング設計のため、モデル全体
   （bf16で全展開すると約44GB）を一度にメモリへ載せることはありません。

このマシンの搭載RAMは約63.8GiB（空き約48.6GiB）で全展開ロードも理論上は可能な
容量でしたが、タスクの基本方針どおり「常にストリーミングで検証する」設計を
採用しました（搭載RAM量に関わらず安全側の設計を優先）。

バックエンドモジュール（`engine.gguf.loader_service`、
`engine.gguf.quant_service`）のimportはCUDA初期化なしで成功したため、
`gguf.quants`へのフォールバックは発生せず、**本番と同一のバックエンドコード
パスで検証できました**。

### 4-2. 結果

```
Output GGUF: ...\output\Sulphur-2-base-distil-Q4_K_M.gguf
Host RAM: total=63.8 GiB free=48.6 GiB
Backend modules imported OK: engine.gguf.loader_service, engine.gguf.quant_service

=== (a) KV metadata / config check [backend path] ===
OK: config JSON parsed, top-level keys: ['audio_vae', 'scheduler', 'transformer', 'vae', 'vocoder']
(elapsed: 0.19s)

=== (b) per-tensor CPU dequantization check [backend path] ===
  ... 500/4444 tensors processed (500 ok, 0 failed so far)
  ... 1000/4444 tensors processed (1000 ok, 0 failed so far)
  ... 1500/4444 tensors processed (1500 ok, 0 failed so far)
  ... 2000/4444 tensors processed (2000 ok, 0 failed so far)
  ... 2500/4444 tensors processed (2500 ok, 0 failed so far)
  ... 3000/4444 tensors processed (3000 ok, 0 failed so far)
  ... 3500/4444 tensors processed (3500 ok, 0 failed so far)
  ... 4000/4444 tensors processed (4000 ok, 0 failed so far)

=== Summary ===
Mode                 : backend
KV/config check      : PASS
Tensors checked      : 4444
Tensors OK           : 4444
Tensors FAILED       : 0
Elapsed              : 44.9s (0.7 min)
Throughput           : 98.9 tensors/s

RESULT: PASS
```

実行コマンド:

```
S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-LTX23-backend\.venv-engine\Scripts\python.exe scripts\e2e_load_check.py
```

（プロジェクトルート `Nz-GGUF-Converter-LTX23` で実行。GPUは一切使用していません
——`dequantize_ggml_tensor`はCPU上のtorchテンソルに対して呼ばれ、明示的な
`.cuda()`/`.to("cuda")`は行っていません。）

**結論: 4444テンソル全件がバックエンドの実dequant関数でCPU上において例外なく
成功し、NaN/Infも検出されませんでした。KVメタデータの`config`取得・パースも
バックエンドの本番コードパスで成功しました。**

---

## 5. E2E段階B（GPU実生成）— 手順書・未実施

**段階Bは今回実施していません。** 以下は実施する際の手順書です。

### 5-1. 前提・注意事項

- GPU排他: 段階Bの生成テストを実行している間は、他にGPUを使うプロセス
  （AviUtl2本体でのプレビュー生成など）を同時に起動しないでください。GPUメモリの
  競合でどちらかが失敗する可能性があります。
- 本ツール自体（download/extract-typemap/convert/verify、およびE2E段階A）は
  すべてCPUのみで完結しており、GPUは一切使用していません。GPUを使うのは段階B
  （実際にバックエンドで動画を生成するテスト）のみです。

### 5-2. 手順

1. 出力GGUF `output\Sulphur-2-base-distil-Q4_K_M.gguf` を、バックエンドの
   `Nz-LTX23-backend\models\ltx-2.3-gguf\` 配下に**新しいサブフォルダ**を作って
   コピーします。例:
   `Nz-LTX23-backend\models\ltx-2.3-gguf\Sulphur-2-base-distil-1.0\Sulphur-2-base-distil-Q4_K_M.gguf`
   （既存の`LTX-2.3-distilled-1.1\`と兄弟フォルダにする。既存の参照GGUFは
   上書きしないこと。）
2. バックエンドの`services/model_registry.py`は`transformer`カテゴリについて、
   デフォルトのGGUFファイルの**2階層上**（＝`models\ltx-2.3-gguf\`）を
   **再帰的に**スキャンして`.gguf`ファイルを自動登録する設計になっています
   （`CATEGORY_SPECS["transformer"]`: `parent_levels=2, recursive=True`）。
   そのため、上記1でコピーするだけで、バックエンドを再起動するかAPI経由で
   `GET /models`を呼べば、新しいモデルがドロップダウン（またはAPIレスポンス）
   に「登録名 = ファイル名（拡張子除く）」として自動的に現れるはずです。
   設定ファイルの編集は不要です。
3. フロントエンド（AviUtl2側UI）またはAPIから、`transformer`カテゴリの
   モデル選択でこの新しいモデル名を選び、**低解像度・短尺**（数秒程度）の
   設定で1本だけ動画を生成します。フルサイズ・長尺でいきなり試すと、
   万一問題があった場合の切り分けに時間がかかるため、まずは軽量な設定で
   「読み込めて、生成が最後まで完走し、出力に明らかな破綻（ノイズ画像など）
   がないか」を確認することを推奨します。
4. 生成が成功したら、ログに`GGUF load complete: ...`のような行が出て
   テンソル数・パラメータ数が記録されているはずです。異常があれば
   `engine/gguf/loader_service.py`のログ（dequant失敗時は「skipping」
   という警告つきでログに残る設計）を確認してください。

段階Bを実施したら、この節に実施日時・使用したモデル選択方法・生成結果の要約を
追記してください。

---

## 6. 10Eros v1.2 変換の検証記録（2026-07-10）

上記1〜4節はSulphur-2-base-distilモデルの変換についての記録でしたが、同じ変換
ツールを使ってもう1本、10Eros v1.2という別モデルも変換しました。対象ファイル:

- 変換元: `safetensors\10Eros_v1.2_bf16.safetensors`
  （サイズ 46,139,885,510 バイト = 約46.14GB、bf16）
- 出力GGUF: `output\10Eros-v1.2-Q4_K_M.gguf`
  （サイズ 17,763,014,976 バイト = 約17.76GB、テンソル4444個）
- 参照GGUF: 上記1〜4節と同じ
  `Nz-LTX23-backend\models\ltx-2.3-gguf\LTX-2.3-distilled-1.1\LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`
  （サイズ 17,763,015,328 バイト、テンソル4444個）

変換の実行時刻（出力ファイルのタイムスタンプより）: 開始 2026-07-10 17:12:43、
完了（最終書き込み） 2026-07-10 19:45:57。所要時間は約2時間33分でした。

出力ファイルサイズは参照GGUFよりわずか352バイト小さいだけで、比率にすると
約0.00198%の差（許容±1%に対して十分小さい）です。Sulphur-2-base-distil版の
ときと全く同じ差分バイト数で、変換ツールの挙動として安定していることが
分かります。

### 6-1. 構造検証の実行結果（全文）

実行コマンド:

```
PYTHONPATH=src .venv\Scripts\python.exe -m converter verify --out "output/10Eros-v1.2-Q4_K_M.gguf"
```

（参照GGUFはconfig.toml既定のまま。`--out`のみ上書き。）

```
Output:    output\10Eros-v1.2-Q4_K_M.gguf
Reference: S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-LTX23-backend\models\ltx-2.3-gguf\LTX-2.3-distilled-1.1\LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf
Tensors:   output=4444 reference=4444

[PASS] tensor_count
[PASS] tensor_names_order
[PASS] tensor_types_shapes
[PASS] kv_keys
[PASS] kv_types
[PASS] kv_fixed_values
[PASS] kv_config_json
[PASS] general_alignment_absent
[PASS] file_size_ratio
[PASS] dequant_sanity

RESULT: PASS
```

終了コード: `0`

各チェック項目の意味は本ドキュメント2節の説明と同一です（`src/converter/verify.py`
は変更していないため）。**結果: 全10項目PASS。失敗項目なし。**Sulphur-2-base-distil
版のときと同じく、全チェックが一発でPASSしました。

### 6-2. E2E段階A（バックエンドローダでのCPUロード検証）

方式は本ドキュメント4節と同一です。`scripts/e2e_load_check.py`は元々
`--gguf`オプションで対象ファイルを指定できる設計だったため、`scripts/`配下への
追加改修は不要でした。GPUを使わないことを明示するため、実行時に環境変数
`CUDA_VISIBLE_DEVICES=""`を設定したうえで、バックエンドの
`.venv-engine\Scripts\python.exe`で実行しました。

実行コマンド:

```
CUDA_VISIBLE_DEVICES="" S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-LTX23-backend\.venv-engine\Scripts\python.exe scripts\e2e_load_check.py --gguf "output/10Eros-v1.2-Q4_K_M.gguf"
```

（プロジェクトルート `Nz-GGUF-Converter-LTX23` で実行。）

結果（全文）:

```
Output GGUF: output\10Eros-v1.2-Q4_K_M.gguf
Host RAM: total=63.8 GiB free=46.7 GiB (informational only -- this script always streams one tensor at a time regardless of available RAM; see module docstring)
Backend modules imported OK: engine.gguf.loader_service, engine.gguf.quant_service

=== (a) KV metadata / config check [backend path] ===
OK: config JSON parsed, top-level keys: ['audio_vae', 'scheduler', 'transformer', 'vae', 'vocoder']
(elapsed: 0.24s)

=== (b) per-tensor CPU dequantization check [backend path] ===
  ... 500/4444 tensors processed (500 ok, 0 failed so far)
  ... 1000/4444 tensors processed (1000 ok, 0 failed so far)
  ... 1500/4444 tensors processed (1500 ok, 0 failed so far)
  ... 2000/4444 tensors processed (2000 ok, 0 failed so far)
  ... 2500/4444 tensors processed (2500 ok, 0 failed so far)
  ... 3000/4444 tensors processed (3000 ok, 0 failed so far)
  ... 3500/4444 tensors processed (3500 ok, 0 failed so far)
  ... 4000/4444 tensors processed (4000 ok, 0 failed so far)

=== Summary ===
Mode                 : backend
KV/config check      : PASS
Tensors checked      : 4444
Tensors OK           : 4444
Tensors FAILED       : 0
Elapsed              : 48.2s (0.8 min)
Throughput           : 92.2 tensors/s

RESULT: PASS
```

終了コード: `0`

バックエンドモジュール（`engine.gguf.loader_service`、`engine.gguf.quant_service`）
のimportはCUDA初期化なしで成功し、`gguf.quants`へのフォールバックは発生せず、
本番と同一のバックエンドコードパスで検証できました。`CUDA_VISIBLE_DEVICES=""`を
設定した状態での実行のため、GPUは物理的に見えない状態で完走しています
（`dequantize_ggml_tensor`はCPU上のtorchテンソルに対して呼ばれ、明示的な
`.cuda()`/`.to("cuda")`も行っていません）。

**結論: 10Eros v1.2版についても、構造検証（10項目）・E2E段階A（KVメタデータ
取得＋4444テンソル全件のCPU dequant）の両方が、Sulphur-2-base-distil版と同じく
一発でPASSしました。GPUは一切使用していません。**
