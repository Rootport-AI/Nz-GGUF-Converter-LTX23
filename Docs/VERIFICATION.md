# 検証記録（VERIFICATION）

（注記: 本書の実行ログ・パス表記は2026-08-19のバックエンドリポジトリ改名より前のもの
です。文中に出てくる`Nz-LTX23-backend`は現在の`Nz-Videomni`を指し、当時のディレクトリ
構成（`models\ltx-2.3-gguf\`等）も現在は`models\LTX23\Weights\`等へ再設計されています。
実行結果を当時のまま正確に記録するため、本文中の表記はそのまま残しています。）

このドキュメントは、本リポジトリの変換ツールで実際に行った変換の実行結果と検証
結果をまとめた記録です。1〜5節がSulphur-2-base、6節が10Eros v1.2（いずれもGGUF
変換）、7節がPrunaVAEDデコーダ変換、8節がコミュニティ製の量子化済みLTX 2.5
（`ltx25-comfyquant`）の記録です。

以下1〜5節は、Sulphur-2-base → GGUF（Q4_K_M）変換についての記録です。対象ファイル:

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

---

## 7. PrunaVAEDデコーダ変換の検証記録（2026-08-05）

`convert-vae` サブコマンド（新設）で、枝刈り版の映像VAEデコーダ「PrunaVAED v2」を
バックエンドがそのまま読み込める形へ変換しました。仕様の正本はバックエンド側の
`Docs/PRUNAVAED_WORKORDER.md` で、本節はその §9 の**ゲートG1（変換ツールの機械
検証）**の記録です。設計判断の説明は `Docs/DESIGN.md` の6節にあります。

対象ファイル:

- 変換元: `safetensors\prunavaed\vae\diffusion_pytorch_model.safetensors`
  （サイズ 1,327,909,418 バイト = 約1.33GB、bf16、diffusers形式）
  - 取得元: Hugging Face `PrunaAI/PrunaVAED`
  - revision（コミットハッシュ）: `4baacd7ef66a6131439542c1f05872afe042e128`
  - SHA-256: `4cbf0cbe6c185514d62c6c58c35dc42d7ea15924f34391e08be12f44bfccdf1d`
- 出力: `output\PrunaVAED-decoder-bf16.safetensors`
  （サイズ **690,047,968 バイト** = 約690MB、テンソル **102本**）
  - SHA-256: `48453517849dd8c0de0d56177d8643c9fb783228d2c7ffc4286a20dfaf7dde40`
  - 内訳: 8 バイト（ヘッダー長）＋ ヘッダー 34,936 バイト ＋ 本体 690,013,024 バイト
  - 本体の内訳: デコーダ100本 690,012,512 バイト ＋ 統計量2本 512 バイト
  - パラメータ総数: **345,006,256**（統計量を除く。ワークオーダー §2.3 が事前に
    積算した値と1個の違いも無く一致）
- 照合に使った既存ファイル（自己検証の項目6）:
  `Nz-LTX23-backend\models\ltx-2.3-components\vae\LTX23_video_vae_bf16.safetensors`

変換の実行時刻（出力ファイルのタイムスタンプより）: 2026-08-05 11:46:36。
ダウンロード（約1.33GB）を含めて約5分、変換処理そのものは十数秒でした。

### 7-1. pytest全結果

実行コマンド: `PYTHONPATH=src .venv\Scripts\python.exe -m pytest tests/ -v`

```
======================= 75 passed, 14 skipped in 29.08s =======================
```

内訳（新設した `tests/test_convert_vae.py` は18件すべてPASS）:

| ファイル | 件数 |
|---|---|
| tests/test_convert.py | 8 |
| tests/test_convert_vae.py | **18（新設）** |
| tests/test_metadata.py | 9 |
| tests/test_output_vs_reference.py | 15 |
| tests/test_quant_roundtrip.py | 24 |
| tests/test_typemap.py | 13 |

本テーマ着手前の総数は 71件（57 passed / 14 skipped）でした。18件増えて 89件
（75 passed / 14 skipped）になり、**既存テストの退行はゼロ**です。skipped の14件は
すべて従来どおりで、参照GGUF（約17.8GB）が手元にある場合だけ意味を持つテストです。

新設テストが押さえている内容:

- キー対応表が、ワークオーダー §4.1 の表を**手で書き写した102行の対照表**と
  完全一致すること（表を生成したのと同じ方法で期待値を作っては検証にならないため、
  対照表は独立に手書きしています）。実際にこのテストが、実装時のキー名の誤り
  （アップサンプラの階層を1段浅く書いていた）を捕まえました。
- 縮小版（各チャンネル幅を1/64にした合成モデル。射影resnet・アップサンプラ・
  統計量という構造上の特徴はそのまま）を使った変換の通し。キー名は幅に依存しない
  ので、**実物とまったく同じ102本のキー**が出ることまで検証できます。
- 同じ形状のテンソルを入れ替えたら、MD5の突き合わせが確実に捕まえること。
  併せて「本数・キー集合・ファイルサイズの各検査はこの誤りを見逃す」ことも
  同時に確認しています（見逃すからこそ全件突き合わせが要る、という主張の裏づけ）。
- 同じ元テンソルを2箇所へ書く対応表（単射でない対応表）を捕まえること。
- テンソルの欠落・想定外のテンソルの混入で、それぞれ専用の例外が上がること。
- `__metadata__` の `config` に `decoder_blocks` が**入っていない**こと。
- 同じ入力から2回変換すると、バイト単位で同一のファイルができること。

### 7-2. 実ファイルに対する `convert-vae` の実行（全文）

実行コマンド:

```
PYTHONPATH=src .venv\Scripts\python.exe -m converter convert-vae
```

（引数なし。すべて `config.toml` の `[prunavaed]` セクションの既定値で実行。
ダウンロードもこのコマンドが行います。）

出力全文（進捗バーの行だけ、同じ行の書き換えが繰り返されるため最終状態のみに省略）:

```
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed.
Falling back to regular HTTP download. For better performance, install the package
with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
Repo    : PrunaAI/PrunaVAED
Revision: 4baacd7ef66a6131439542c1f05872afe042e128
Filename: vae/diffusion_pytorch_model.safetensors
Source  : S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-GGUF-Converter-LTX23\safetensors\prunavaed\vae\diffusion_pytorch_model.safetensors
Output  : S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-GGUF-Converter-LTX23\output\PrunaVAED-decoder-bf16.safetensors
Ref. VAE: S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-LTX23-backend\models\ltx-2.3-components\vae\LTX23_video_vae_bf16.safetensors
(ダウンロード進捗バー ... 1.33G/1.33G)
Verifying SHA-256 of the downloaded file...
SHA-256 OK: 4cbf0cbe6c185514d62c6c58c35dc42d7ea15924f34391e08be12f44bfccdf1d
Verifying the source SHA-256 (pinned revision 4baacd7ef66a6131439542c1f05872afe042e128)...
Source SHA-256 OK: 4cbf0cbe6c185514d62c6c58c35dc42d7ea15924f34391e08be12f44bfccdf1d
Writing 102 tensors to S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-GGUF-Converter-LTX23\output\PrunaVAED-decoder-bf16.safetensors
writing tensors: 100%|##########| 102/102 [00:00<00:00, 295.83tensor/s]
md5 cross-check: 100%|##########| 102/102 [00:01<00:00, 57.98tensor/s]

Source : S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-GGUF-Converter-LTX23\safetensors\prunavaed\vae\diffusion_pytorch_model.safetensors
Output : S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-GGUF-Converter-LTX23\output\PrunaVAED-decoder-bf16.safetensors
SHA-256 (source): 4cbf0cbe6c185514d62c6c58c35dc42d7ea15924f34391e08be12f44bfccdf1d
Tensors: 102
Params : 345,006,256 (per_channel_statistics excluded)
Bytes  : header=34936 payload=690013024 file=690047968

[PASS] 1_tensor_count
[PASS] 2_key_set
[PASS] 3_tensor_shapes
[PASS] 4_parameter_total
[PASS] 5_md5_passthrough
[PASS] 6_latent_statistics_vs_reference
[PASS] 7_output_header_roundtrip
[PASS] 8_total_size

RESULT: PASS
```

終了コード: `0`

**結果: 自己検証8項目すべてPASS。失敗項目・スキップ項目ともにゼロ。**
今回は照合用の既存ファイル（`LTX23_video_vae_bf16.safetensors`）が手元にあったため、
任意項目である6番も実際に実行されてPASSしています。

各項目の意味:

1. `1_tensor_count` — テンソル本数が102本であること（デコーダ100本＋統計量2本）。
2. `2_key_set` — 出力したキー名の集合が、ワークオーダー §4.1 の表から機械的に
   生成した期待集合と**過不足なく完全一致**すること。並び順まで一致します。
3. `3_tensor_shapes` — 全102本の形状が、事前に確定した形状表と一致すること。
4. `4_parameter_total` — パラメータ総数が 345,006,256 であること（統計量を除く）。
5. `5_md5_passthrough` — (a) 対応表が単射であること（同じ元テンソルを2箇所へ
   書いていないこと）と、(b) **全102本**について「出力ファイル上のバイト範囲」と
   「元ファイル上の該当バイト範囲」のMD5が一致すること。標本抽出ではなく全件です。
   理由は、項目1〜4・6〜8がすべて「集合」と「形」しか見ておらず、**同じ形状の
   テンソルどうしの取り違えを全項目が素通しする**ためです（たとえば `up_blocks.5` の
   `conv1` と `conv2` はどちらも `[384,384,3,3,3]` で、入れ替えても本数・形状・
   パラメータ総数・ファイルサイズは1バイトも変わりません）。
6. `6_latent_statistics_vs_reference` — `per_channel_statistics.mean-of-means` と
   `std-of-means` が、既存の `LTX23_video_vae_bf16.safetensors` の同名テンソルと
   **バイト単位で一致**すること。PrunaVAEDはデコーダだけを作り直したもので統計量は
   元のままのはずだ、という事前調査の裏づけになります。
7. `7_output_header_roundtrip` — 書いたファイルを読み直して、キーの並び・
   バイト範囲が隙間も重なりも無く連続していること、`__metadata__` に `config` が
   あり `decoder_blocks` を含まないこと、`_class_name` が `PrunaVAEDDecoder` で
   あること、そして**safetensorsライブラリ本体がこのファイルを受け付ける**こと。
8. `8_total_size` — ファイルサイズが「8バイト＋ヘッダー＋本体」に一致し、かつ
   本体が 690,012,512（デコーダ）＋512（統計量）バイトであること。

### 7-3. 出力ファイルの中身（独立に読み直したもの）

```
n keys: 102
first 3: ['conv_in.conv.weight', 'conv_in.conv.bias', 'up_blocks.0.res_blocks.0.conv1.conv.weight']
last 4 : ['conv_out.conv.weight', 'conv_out.conv.bias',
          'per_channel_statistics.mean-of-means', 'per_channel_statistics.std-of-means']
dtypes : ['BF16']
conv_in  shape: [1024, 128, 3, 3, 3]
conv_out shape: [48, 64, 3, 3, 3]
射影resnetのキー: ['up_blocks.3.norm3.weight', 'up_blocks.3.norm3.bias',
                   'up_blocks.3.conv_shortcut.weight', 'up_blocks.3.conv_shortcut.bias',
                   'up_blocks.6.norm3.weight', 'up_blocks.6.norm3.bias',
                   'up_blocks.6.conv_shortcut.weight', 'up_blocks.6.conv_shortcut.bias']
meta keys: ['config', 'license', 'model_version', 'provenance']
config  : {"vae": {"_class_name": "PrunaVAEDDecoder", "latent_channels": 128,
           "patch_size": 4, "norm_layer": "pixel_norm", "causal_decoder": false,
           "timestep_conditioning": false, "decoder_base_channels": 128}}
model_version: PrunaVAED-v2
license : 21,393文字（LTX-2 Community License 全文。既存の
          LTX23_video_vae_bf16.safetensors から取ったものと同一）
provenance: {"source_repo": "PrunaAI/PrunaVAED",
             "source_revision": "4baacd7ef66a6131439542c1f05872afe042e128",
             "source_filename": "vae/diffusion_pytorch_model.safetensors",
             "source_sha256": "4cbf0cbe6c185514d62c6c58c35dc42d7ea15924f34391e08be12f44bfccdf1d",
             "converted_by": "Nz-GGUF-Converter-LTX23 convert-vae",
             "tool_version": "1.1.0"}
```

確認できること: キーに `decoder.` も `vae.` も付いていないこと（バックエンドは
このファイルをキー変換なしで読むため、ここに書かれた名前がそのまま
`load_state_dict` に渡ります）、全テンソルがBF16のままであること、`config` に
`decoder_blocks` が無いこと、そして射影resnet（PrunaVAEDが新設した、チャンネル数を
変えるブロック）の `norm3` と `conv_shortcut` が平坦インデックス3番と6番に正しく
配置されていること。

### 7-4. 再現性の確認

同じ入力から2回変換し、出力がバイト単位で同一になることを確認しました。

実行コマンド（2回目。ダウンロードを飛ばし、出力先だけ変えたもの）:

```
PYTHONPATH=src .venv\Scripts\python.exe -m converter convert-vae
  --st-path "safetensors/prunavaed/vae/diffusion_pytorch_model.safetensors"
  --out "<一時ディレクトリ>/rerun.safetensors"
```

```
run1 48453517849dd8c0de0d56177d8643c9fb783228d2c7ffc4286a20dfaf7dde40
run2 48453517849dd8c0de0d56177d8643c9fb783228d2c7ffc4286a20dfaf7dde40
identical: True
```

2回目も自己検証8項目すべてPASS（終了コード `0`）でした。出力ファイルには時刻を
一切書き込んでいないため、再ホストしたファイルが正しいかどうかを、後から誰でも
再変換して確かめられます。

### 7-5. G1の結論

ワークオーダー §9 のゲートG1が要求する4点はすべて満たしました。

- `tests/test_convert_vae.py` のpytestが全PASS（18件）。既存テストの退行なし。
- 実ファイル（revision固定・SHA-256照合済み）に対する `convert-vae` の実行が成功し、
  §5.3 の自己検証8項目がすべてPASS。
- 実行コマンドと出力全文を本節に記録。
- 出力ファイルのサイズ（690,047,968バイト）・テンソル本数（102本）・
  パラメータ総数（345,006,256）を明記。

なお、後続のゲート（G2以降＝バックエンド側の読み込み・数値の健全性・速度の採否
判定）は本リポジトリの範囲外で、バックエンド側の `Docs/VERIFICATION_LOG.md` §52 に
記録されます。

（2026-09-07追記: バックエンド（Nz-Videomni、旧Nz-LTX23-backend）が公開された
ため、この記録は
`https://github.com/Rootport-AI/Nz-Videomni/blob/main/Docs/VERIFICATION_LOG.md`
の §52 として参照できます。）

---

## 8. ltx25-comfyquant（コミュニティ製の量子化済みLTX 2.5）変換の検証記録（2026-09-13）

ComfyUI向けにあらかじめ量子化されたLTX 2.5のファインチューンモデルを、逆量子化して
公式版と同じ構造のGGUFへ作り直す第4のプロファイル `ltx25-comfyquant` を追加しました。
契約（受け入れ条件・型の決め方・出力の約束事）の正本は
`Docs/LTX25_CONVERSION_MODE_SPEC.md` の「ltx25-comfyquant profile」節です。本節は
その節が要求する検証を実際に行った結果の記録です。**実物の変換は完全版のファイルに
対して実施済み**で（変換と自己検証の結果は8-5-1、公式版との層ごとの照合は8-3′）、
残るのはGPUを使うバックエンド実機ゲートだけです（8-5-3）。

### 8-1. 対象と背景

対象ファイル:

- `Nz-Videomni\models\LTX25\Weights\redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors`
- ヘッダの `__metadata__` に `quant_format = "mixed:w4a8+int8"`、
  `quant_mixed_hi_layers = "831"` が入っています。作成者側の道具立てが、出力への影響が
  大きい層を8bit（int8）へ、残りを4bit（w4a8）へ振り分けたことを示す記録です。
- テンソルは全部で8,447本。公式と同じ論理的な重みが4,349本あり、残る4,098本はすべて
  量子化の補助データ（倍率・符号帳・形式マーカー）です。論理4,349本の内訳は、
  量子化されていないものが2,909本、int8 ConvRotが831本、非対称w4a8が609本です。

**手元のファイルは不完全なダウンロードでした。** ヘッダは本体17,024,750,304バイトを
宣言しているのに対し、実ファイルは16,150,757,376バイト（ヘッダ1,272,912バイト＋本体
16,149,484,464バイト）しかなく、末尾875,265,840バイト（約835MiB）が欠けています。
欠けた領域には、1,440本の形式マーカー（`comfy_quant`）の全数と、重み54本が含まれます。
完全版のファイルサイズはヘッダの宣言から逆算して17,026,023,216バイトのはずで、
再取得時に照合するSHA-256は
`ab59bb5e74e76937b55a6876fb23c4b58261e798227eb544f7d8a2934728c882` です
（`--expect-sha256` に渡せます）。この値の出典は配布元の公開API
（`https://civitai.com/api/v1/model-versions/3250230` の `files[].hashes.SHA256`、
2026-09-13取得）で、同APIの掲載サイズ16,626,975.80KB（=17,026,023,216バイト）は
上記の逆算値と一致します。

その後、同日中に完全版を取得しました。実測のサイズとSHA-256は上の値と一致しています
（変換の入口で `--expect-sha256` により照合。8-5-1）。以降の8-3′・8-5-1・8-5-2は、
すべてこの完全版に対する結果です。8-3は、不完全版しか無かった時点の記録としてその
まま残してあります。

公式ファイルとの関係については、次を確認しました。

- ヘッダの `config`・`license`・`model_version`・`gemma_source_checkpoint` の4つが、
  公式の `ltx-2.5-22b-distilled-transformer-bf16.safetensors` と**バイト単位で同一**。
- `config` のSHA-256は `13be9edf16635af90dfb02881d87383256e20583ccdd825dd424a2d7b6923855` で、
  変換ツールが持つ公式ビルダーオラクル `typemap/ltx25_builder_oracle.json` の
  `config_bytes_sha256` と一致。つまり構造は公式のLTX 2.5そのものです。

承認済みの公式マップ（`typemap/ltx25_conversion_map.json`）と突き合わせた内訳:

| 公式マップの型 | 元ファイルでの持ち方 | 本数 |
|---|---|---|
| Q4_K | int8 ConvRot | 831 |
| Q4_K | 非対称w4a8 | 513 |
| Q4_K | 量子化されていない（BF16 218・F32 96） | 314 |
| BF16 | 非対称w4a8（connector） | 96 |
| BF16 | 量子化されていない | 2,305 |
| F32 | 量子化されていない（元ファイルはBF16） | 290 |

### 8-2. pytest全結果

実行コマンド: `PYTHONPATH=src .venv\Scripts\python.exe -m pytest tests/ -v`

```
============================= test session starts =============================
platform win32 -- Python 3.12.9, pytest-9.1.1, pluggy-1.6.0
collecting ... collected 280 items

（中略・全件PASSED）

======================= 280 passed in 90.38s (0:01:30) ========================
```

内訳（新設2ファイルは太字）:

| ファイル | 件数 |
|---|---|
| tests/test_comfy_dequant.py | **66（新設）** |
| tests/test_convert.py | 12 |
| tests/test_convert_vae.py | 18 |
| tests/test_ltx25.py | 53 |
| tests/test_ltx25_comfyquant.py | **35（新設）** |
| tests/test_ltx25_gemma.py | 21 |
| tests/test_metadata.py | 12 |
| tests/test_output_vs_reference.py | 17 |
| tests/test_quant_roundtrip.py | 33 |
| tests/test_typemap.py | 13 |

本テーマ着手前の総数は179件でした。101件増えて280件になり、**既存テストの退行は
ゼロ**です。既存ファイルへの変更は `tests/test_ltx25.py` の1行だけで、これは既存の
テスト関数にワーカー数の確認を1つ足したものなので件数は変わっていません。失敗・
スキップともに0件です。

新設テストが押さえている内容のうち、特に効いているもの:

- 逆量子化の検算は、テスト側に**独立に書いた順方向の量子化器**を通してから復元し、
  一致を見る形にしています。さらに「ニブル（1バイトに詰めた4bitの上下）の順序を
  わざと逆にすると相関がほぼ0まで崩れる」ことも確認しました。往復テストが素通し
  （何を入れても通る形）になっていないことの裏づけです。
- FP8 E4M3（8bitの浮動小数点形式）の復号表256通りを、テスト側でビットから組み立てた
  別実装と全数照合しています。
- `typemap/` のJSON 9本のSHA-256を固定値で確認しています。この系統は `typemap/` を
  読むだけで書かない、という約束の機械的な裏づけです。

### 8-3. 現物照合（不完全版での先行実行・公式bf16との層ごとの数値比較）

`scripts/compare_comfyquant_vs_official.py` で、コミュニティ版を逆量子化した結果と、
手元にある公式bf16原本を層ごとに比較しました。GPUは使わず、両ファイルを1テンソルずつ
読む方式です。レポートの全文は `output\compare-redgraft-partial.json`（git管理外）に
あり、以下の数値はそこから転記しています。

- 実行時刻: 2026-09-13、所要692.7秒。
- 量子化された1,440層のうち、**1,386層を照合**できました。残る54層は欠落部にあって
  読めません（エラーは0件）。
- 形式マーカーは全数が欠落部にあるため読めず、1,386層すべてで補助テンソルの
  組み合わせから形式を推定しています（レポートの `marker_assumed` が真）。

| ゲート | 対象 | 合格条件 | 結果 |
|---|---|---|---|
| G-0 | 網羅性 | 読めない層0・逆量子化エラー0・形式マーカーの推定0（全マーカーを実読） | **不合格（想定どおり）** 読めない層54・マーカー推定1,386。完全版でのやり直し（8-5-2）で合格が必要 |
| G-A | connectorの非量子化BF16 162本 | 公式とバイト一致 | **合格** 162/162一致 |
| G-B | connectorのw4a8 48本 | cos ≥ 0.99 | **合格** 最小0.99731・中央値0.99733・最大0.99735 |
| G-C | 量子化層 1,386本 | 全層 cos ≥ 0.50 | **合格** 最小0.80525・第1百分位0.93179・中央値0.99801・最大0.99996 |
| G-D | transformerの量子化層 1,338本 | cos中央値 ≥ 0.90 かつ第1百分位 ≥ 0.60 | **合格** 中央値0.99826・第1百分位0.93037・最小0.80525 |
| G-E | 量子化層 1,386本 | 0.50 ≤ 標準偏差比 ≤ 2.00 | **合格** 最小0.98517・中央値1.00050・最大1.22209 |
| G-F | 量子化層 1,386本 | 行ノルム相関 ≥ 0.80 | **合格** 最小0.92817・第1百分位0.98027・中央値0.99987 |

（cos＝コサイン類似度。2本のベクトルの向きがどれだけ揃っているかを −1〜1で表す指標で、
1に近いほど一致。行ノルム相関＝出力チャンネルごとの重みの大きさの並びが、公式と
どれだけ同じ順序・同じ比率かを見る指標。倍率を掛ける向きを取り違えた実装は、ここで
崩れます。）

形式ごとの分布:

| 区分 | 本数 | cos 最小 | cos 中央値 | 相対RMSE 中央値 |
|---|---|---|---|---|
| int8 ConvRot（transformer） | 829 | 0.97393 | 0.99946 | 0.033 |
| 非対称w4a8（transformer） | 509 | 0.80525 | 0.98668 | 0.163 |
| 非対称w4a8（connector） | 48 | 0.99731 | 0.99733 | 0.073 |

読み取れること:

- G-Aが成立した（connectorの非量子化テンソルが公式と1バイトも違わない）ため、
  connectorは移植のときに一切手を加えられていないと言えます。したがってG-Bの48本で
  観測された0.9973前後のずれは**ファインチューンによる違いではなく、4bit量子化の
  誤差そのもの**です。逆量子化の実装が正しいことの、最も強い裏づけになります。
- 8bitのint8層は誤差が小さく（相対RMSEの中央値3.3%）、4bitのw4a8層は大きい
  （同16.3%）という並びで、ビット数どおりの素直な結果です。
- cosが低い層は `transformer_blocks.{9,14,15,18,19,20}.attn2.to_q` /
  `.to_k` に集中しています（最小は15番ブロックの `attn2.to_k` で0.80525）。
  クロスアテンションのqとkはもともと値の分布が偏りやすく、4bit量子化の誤差が
  出やすい場所です。

### 8-3′. 完全版での現物照合（2026-09-13・G-0〜G-F全合格）

完全版のファイルに対して、8-3と同じ照合をやり直しました。実行したコマンドは
`scripts/compare_comfyquant_vs_official.py` に `--community`（完全版）・`--official`
（公式bf16原本）・`--report output/compare-redgraft-full.json` を渡したものです。
レポートの全文は `output\compare-redgraft-full.json`、実行ログは
`output\redgraft-compare-full.log` にあります（どちらもgit管理外）。以下の数値は
そこから転記しています。

- 実行時刻: 2026-09-13 14:19:32〜14:31:33、所要720.0秒。終了コードは0。
- 量子化された1,440層を**全数照合**しました（読めない層0・逆量子化エラー0）。
- 形式マーカーは1,440本すべてを**実際に読んで**判定しています（`markers_parsed`
  1,440・`markers_assumed` 0）。推定は1本も使っていません。本文の中身は8-4。

| ゲート | 対象 | 合格条件 | 結果 |
|---|---|---|---|
| G-0 | 網羅性 | 読めない層0・逆量子化エラー0・形式マーカーの推定0 | **合格** 1,440/1,440 |
| G-A | connectorの非量子化BF16 162本 | 公式とバイト一致 | **合格** 162/162一致 |
| G-B | connectorのw4a8 96本 | cos ≥ 0.99 | **合格** 最小0.99729・第1百分位0.99729・中央値0.99732・最大0.99737 |
| G-C | 量子化層 1,440本 | 全層 cos ≥ 0.50 | **合格** 最小0.80525・第1百分位0.93512・中央値0.99767・最大0.99996 |
| G-D | transformerの量子化層 1,344本 | cos中央値 ≥ 0.90 かつ第1百分位 ≥ 0.60 | **合格** 中央値0.99825・第1百分位0.93055・最小0.80525 |
| G-E | 量子化層 1,440本 | 0.50 ≤ 標準偏差比 ≤ 2.00 | **合格** 最小0.98517・中央値1.00045・最大1.22209 |
| G-F | 量子化層 1,440本 | 行ノルム相関 ≥ 0.80 | **合格** 最小0.92817・第1百分位0.98059・中央値0.99986・最大0.999999 |

形式ごとの分布:

| 区分 | 本数 | cos 最小 | cos 中央値 | 相対RMSE 中央値 |
|---|---|---|---|---|
| int8 ConvRot（transformer） | 831 | 0.97393 | 0.99946 | 0.033 |
| 非対称w4a8（transformer） | 513 | 0.80525 | 0.98669 | 0.163 |
| 非対称w4a8（connector） | 96 | 0.99729 | 0.99732 | 0.073 |

8-3で読めなかった54層について（新しく照合できた分）:

- 内訳は、transformerの6本（`transformer_blocks.9` の後半。int8が2本・w4a8が4本）と、
  `video_embeddings_connector` のw4a8 48本です。
- cosは最小0.98444・中央値0.99732・最大0.99993で、8-3で得ていた分布の内側に収まりました。
  最小値の層は `transformer_blocks.9.ff.net.2`（w4a8）です。
- 54本が加わっても全体の統計はほとんど動いていません（全層のcos中央値は0.99801→0.99767）。
  8-3の結論（逆量子化の実装は正しい）は、完全版でもそのまま成り立ちます。

### 8-4. 副次的な発見

照合のついでに、量子化されていない2,909本についても公式と突き合わせました。

- **2,523本は公式とバイト単位で完全一致**でした。
- 290本の `*scale_shift_table*` は、公式がF32で持っているものをコミュニティ版が
  BF16で持っています。dtypeが違うのでバイト比較はできませんが、公式のF32をBF16へ
  丸めた結果と**290本すべてがビット単位で一致**しました（本節の執筆時に確認）。
  値としては同じもので、幅が狭いだけです。
- 残る96本の `*.to_gate_logits.weight` だけが、値として公式と異なります
  （コミュニティ版がF32・公式がBF16で、float32に揃えたときの最大絶対差は
  0.0076〜0.0776）。量子化されていないテンソルのうち、中身が書き換わっているのは
  この96本だけ、ということになります。

形式マーカーの長さについて:

- 1,440本の `comfy_quant` はすべて1次元のU8で、長さは67バイトが1,248本、
  72バイトが192本の2種類です。長さは量子化形式と対応していません
  （67バイト: int8 744本・w4a8 504本／72バイト: int8 87本・w4a8 105本）。
- 想定される2種類のマーカーJSONを、Pythonの `json.dumps` の既定書式（`, ` と `: `）で
  書くとどちらも72文字、区切りを詰めて書くとどちらも67文字になります。観測された
  2種類の長さは、この書式の違いで説明がつきます。
- **完全版で本文を実際に読み、この説明が正しいことを確かめました**（2026-09-13）。
  実文は次の4種類で、意味の上では2種類、残る違いは区切り文字の書き方だけです。

| 実文 | 長さ | 本数 |
|---|---|---|
| `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}` | 67 | 744 |
| `{"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256}` | 67 | 504 |
| `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}` | 72 | 87 |
| `{"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}` | 72 | 105 |

- int8の831本（744＋87）は、**全数が `"convrot": true` を明示**していました。逆量子化の
  実装は、マーカーに `convrot` の記載が無ければ「回転なし」と解釈するので、ここは結果を
  左右する要点でした。8-3が回転ありを仮定して高いcosを出していたことと、文面は矛盾
  しません。w4a8の609本（504＋105）は形式上つねに回転ありで、本文に `convrot` は
  現れません。

### 8-5. 完全版に対する作業（8-5-1・8-5-2は実施済み／8-5-3は未実施）

8-5-1と8-5-2は2026-09-13に実施しました。残るのは、GPUを使う8-5-3だけです。

#### 8-5-1. 実物の受け入れ検査と変換（実施済み・2026-09-13）

受け入れ検査の内容は、変換が書き出した `～.gguf.inventory.json` にそのまま残って
います。期待どおりの値でした。

- raw tensors 8,447（`{BF16: 2813, F32: 2145, F8_E4M3: 609, I8: 1440, U8: 1440}`）
- logical tensors 4,349（`{asym_w4a8_int8: 609, int8_tensorwise: 831, plain: 2909}`）
- marker variants 2（int8とw4a8の2種類。内容を正規化してから数えるため、8-4の書式の
  違いでは増えません）
- 出力の type_counts は `{BF16: 2401, F32: 290, Q6_K: 1658}`

変換のコマンドと結果:

```
run.bat convert --model ltx25-comfyquant ^
    --st-path safetensors\redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors ^
    --expect-sha256 ab59bb5e74e76937b55a6876fb23c4b58261e798227eb544f7d8a2934728c882
```

- 実行時刻: 2026-09-13 13:27:02〜14:15:54、所要2,932秒（約49分）。GPUは使わず、
  量子化ワーカーは既定の4本。実行ログは `output\redgraft-convert-Q6_K.log`。
- `--expect-sha256` を付けたので、元ファイルの同一性（17,026,023,216バイト・
  SHA-256 `ab59bb5e…`）は変換の入口で照合されています。
- 出力: `output\redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K.gguf`
- サイズ: **19,631,143,808バイト**（READMEの表の値と1バイトも違いません）
- SHA-256: `66ed7bf11d87e0493706fec7272189e4f2217ec8927935ae409bda7d723b1a61`
- テンソル4,349本・型の内訳 `{BF16: 2401, F32: 290, Q6_K: 1658}`
- `～.gguf.manifest.json` の要点: `quant_type` は `Q6_K`、`official_map_sha256` は
  `6d41db41…`、`builder_oracle_sha256` は `fa3b8518…`、`inventory_sha256` は
  `b2aa108a…`、`source_quant_format` は `mixed:w4a8+int8`、`quant_kind_counts` は
  `{asym_w4a8_int8: 609, int8_tensorwise: 831, plain: 2909}`、`tool_version` は 1.2.0。
- `output\` にある既存の成果物（公式版のGGUFなど）は書き換わっていません。作業用の
  一時ファイル（`*.tmp`）も残っていません。

自己検証（`self-verify`）は、変換とは別のプロセスで単独実行し、合格しました。

```
run.bat self-verify --model ltx25-comfyquant ^
    --st-path safetensors\redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors ^
    --out output\redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K.gguf
```

- 実行時刻: 14:16:13〜14:19:23、所要190秒。ログは `output\redgraft-self-verify-Q6_K.log`。
  報告されたサイズ・SHA-256・型の内訳は、上の変換の値と同じです。

出力GGUFの追加確認（GPU不要・読むだけ。2026-09-13）:

- KVのキーは7つちょうどでした（`general.architecture`・`general.quantization_version`・
  `general.file_type`・`config`・`license`・`model_version`・`gemma_source_checkpoint`）。
  元ファイルの `quant_format` などは混入していません。
- `general.architecture` は `ltxv`、`model_version` は `2.5.0`。
- テンソルは4,349本、型の内訳は上のとおりで、名前は昇順に並んでいます。
- 公式版のGGUF（`output\LTX-2.5-22B-distilled-transformer.gguf`）と、テンソルの名前・
  並び順・shapeが**4,349行すべて一致**しました。違うのは型だけで、Q4_K→Q6_Kの1,658行
  です。ほかの型の差はありません。
- 数層を抜き出し、出力GGUFから復元した値と、元ファイルから逆量子化した値を比べました。

| 区分 | テンソル | 出力の型 | 結果 |
|---|---|---|---|
| int8（transformer） | `transformer_blocks.0.audio_attn2.to_k.weight` | Q6_K | 相対RMSE 0.0213・cos 0.99977 |
| 非対称w4a8（transformer） | `transformer_blocks.0.attn1.to_k.weight` | Q6_K | 相対RMSE 0.0246・cos 0.99970 |
| 非対称w4a8（connector） | `audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_k.weight` | BF16 | バイト一致 |
| 量子化されていない | `audio_embeddings_connector.learnable_registers` | BF16 | バイト一致 |

ここでの相対RMSEは、Q6_Kへ量子化し直したことで**新たに加わった誤差**そのものです
（比べている相手が公式版ではなく、元ファイルを逆量子化した値のため）。2%台という値は、
BF16で出す行がバイト一致したことと合わせて、書き出しの経路に取り違えが無いことを示します。

#### 8-5-2. 現物照合のやり直し（実施済み・2026-09-13）

8-3の照合を完全版に対して実行し直し、**G-0〜G-Fの全ゲートに合格**しました（終了コード0）。
数値と読み取れることは8-3′にまとめてあります。懸案だった2点——読めなかった54層と、
形式マーカーの実文——は、どちらも解消しています（54層は8-3′、マーカーの中身は8-4）。

#### 8-5-3. バックエンド実機ゲート（GPU使用）— 未実施・手順書

**BE-1〜BE-6はまだ実施していません。** GPUを使うため、実行前に必ず了承を得てください。
実行中は他のGPU利用プロセス（AviUtl2本体でのプレビュー生成など）を同時に動かさないこと。

| # | 手順 | 合格条件 |
|---|---|---|
| BE-1 | 出力GGUFを `Nz-Videomni\models\LTX25\Weights\` の直下へ置く | — |
| BE-2 | `GET /models` を呼ぶ | 登録名（拡張子を除いたファイル名）が列挙される |
| BE-3 | `POST /pipeline/load` でこのtransformerを選ぶ | ロード完了。Gemmaの版の照合を通過し、未初期化のテンソルが残らない |
| BE-4 | `python -m engine25.gguf_transformer --selftest <gguf>` | 解放と再構築の往復が通る。VRAMのピークを記録する |
| BE-5 | 短尺のT2V（文章から動画）を1本生成 | 完走し、黒画面やノイズにならない。同じseedでの公式GGUFの出力と見比べる |
| BE-6（任意） | `--quant-type Q4_K` の出力と同じseedで比較 | Q6_Kにした効果と、速度差を把握する |

実施したら、この節に実施日時と結果の要約を追記してください。

### 8-6. 申し送り

- **（解消済み・2026-09-13）** int8側のマーカーが `convrot` を真で持っているかは、完全版で
  本文を読んで確認しました。831本すべてが `"convrot": true` を明示しており、8-3の照合が
  置いていた仮定と食い違いはありません（8-4）。
- `general.file_type` はすべてのプロファイル共通の固定値15（Q4_K_Mの意味）のままです。
  `--quant-type Q6_K` で出力した場合、第三者のGGUF閲覧ツールにはこの表示が実態と
  違って見えます。実際の型は `～.gguf.manifest.json` の `quant_type` が正本で、
  同じファイルの `general_file_type_note` にもその旨を書いてあります。
- **（解消済み・2026-09-13）** `comfy_quant` の本文は完全版で読みました。マーカー長の
  67/72の違いがJSONの書式の差であることも、実文の突き合わせで確かめています（8-4）。
