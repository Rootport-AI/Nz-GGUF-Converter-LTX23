# Nz-GGUF-Converter-LTX23

## LTX 2.5 conversion (opt-in, source-locked)

The existing commands remain LTX 2.3 by default. The LTX 2.5 transformer is
selected only with `--model ltx25`, and the LTX-fine-tuned Gemma 4 text
encoder only with `--model gemma4-ltx25`; the two are independent profiles
with separate source locks and separate evidence. E1-E3 are pinned from the
authenticated official source and the current 4,349-row E3 bundle inventory
has an independently approved E4 map. The previous 4,091-row approved map is
deliberately stale and is rejected by inventory hash. E5/E6 are complete for
both profiles as of 2026-08-21 (see the Open gates section of
`Docs/LTX25_CONVERSION_MODE_SPEC.md` for the output digests). This does not
authorize publication or backend use. It never derives a type map from a
third-party GGUF.

The approved replacement E4 policy uses the following explicit staging
sequence. The resulting artifact remains separate from publication and backend
acceptance:

```
run.bat inspect --model ltx25
run.bat convert --model ltx25
run.bat self-verify --model ltx25

run.bat inspect --model gemma4-ltx25
run.bat convert --model gemma4-ltx25
run.bat self-verify --model gemma4-ltx25
```

`all` and `extract-typemap` are intentionally LTX 2.3-only. LTX 2.5 emits
all 4,349 source rows after one `model.diffusion_model.` strip: 4,091
transformer rows plus 129 audio and 129 video embedding connectors. The
connectors retain their bare GGUF names and BF16 type so the existing LTX
bundle loader can extract them; they are never Q4_K. Each source dtype must
match the E3 per-key profile (4,059 BF16 and 290 F32). `build-map` is a
maintainer review operation, not a normal conversion step: it always writes
`typemap/ltx25_conversion_map.draft.json`, and refuses the approved E4 path.
The independently reviewed 4,349-row E4 map is accepted by conversion. See
`Docs/LTX25_CONVERSION_MODE_SPEC.md` for the source gate and output protocol.

Q4_K uses bounded 1024-block tasks. LTX 2.3 keeps one worker by default;
LTX 2.5 uses the reviewed profile default of four workers (1--8 accepted via
`--quant-workers`). This changes neither the float32 kernel nor output bytes.

このリポジトリは、Nz-Videomni（AviUtl2向けの動画生成システム）で使う**重みファイルの変換ツールをまとめて置く場所**です。名前にGGUFと入っていますが、扱うのはGGUF変換だけではありません。現在は次の2つの変換を収めています。

1. **GGUF変換**（このツールの出発点）— safetensors（PyTorchの重みファイル形式）を、GGUF（GPT-Generated Unified Format。llama.cpp系のツールで使われる量子化モデル形式）のQ4_K_M（4bit量子化の一種。重みを4bitに圧縮しつつ精度劣化を抑える方式）へ変換します。変換の系統は4つあり、`--model` で選びます。
   - `ltx23`（既定）— LTX 2.3のファインチューンモデル「Sulphur 2 base」（約43GB）を約17.8GBへ。
   - `ltx25` — LTX 2.5本体の映像生成モデル（約42GB）を約14.7GBへ。
   - `gemma4-ltx25` — LTX 2.5が使う文章理解モデル「Gemma 4」（約26GB）を約9.2GBへ。
   - `ltx25-comfyquant` — コミュニティ製の「すでに量子化されたLTX 2.5」を逆量子化して、公式版と同じ構造のGGUFへ作り直します（下記「コミュニティ製の量子化済みLTX 2.5モデルの変換」を参照）。
2. **PrunaVAED変換**（`convert-vae`）— 枝刈り（pruning。寄与の小さいチャンネルを削ること）を施した映像VAEデコーダ「PrunaVAED」の配布ファイルから、デコーダ部分だけを取り出して約690MBのsafetensorsに作り直します。バックエンドがそのまま読み込める形（キー名の付け替え済み）で出力します。

変換後のGGUFは、既存のバックエンド（Nz-Videomni）が読み込んでいるLTX-2.3-22B-distilled-1.1-Q4_K_M.ggufと同じ構造・型マップに揃えることで、バックエンド側の推論コードを変更せずに差し替えられるようにします。

### コミュニティ製の量子化済みLTX 2.5モデルの変換（`--model ltx25-comfyquant`）

有志が公開しているLTX 2.5のファインチューンモデルの中には、ComfyUI（ノードをつないで画像や動画を生成するツール）向けに**あらかじめ量子化された形でしか配布されていない**ものがあります。この系統は、そうしたファイルの重みを一度もとの精度へ復元（逆量子化）したうえで、公式版とまったく同じ構造のGGUFへ作り直します。量子化前のbf16版がどこにも公開されていない場合の、唯一の取り込み手段です。

受け入れられる量子化形式は次の2種類だけです。それ以外の形式（FP8、NVFP4など）や、見覚えのない指定が1つでも混じっていれば、その場で停止します。

- **int8 ConvRot** — 重みを8bit整数に落とし、行ごとの倍率で戻す方式。これにConvRot（アダマール変換による回転。入力チャンネルを混ぜて外れ値をならし、量子化誤差を小さくする処理）を併用します。
- **非対称w4a8** — 重みを4bitの符号帳（16段階の値の一覧表）で表し、16チャンネルごとの相対倍率と行ごとの倍率を掛け合わせて戻す方式。ConvRotは常に有効です。

#### 前提

- 対象にできるのは**公式のLTX 2.5から派生したモデルだけ**です。ファイル先頭のヘッダに入っている`config`（モデル構造の定義）が公式と一致すること、テンソル（重みの配列）の名前と形状が公式の4,349本と過不足なく一致すること、対になる文章理解モデルの版を示す`gemma_source_checkpoint`があることを確認し、1つでも違えば変換を始めません。配布元が固定されていないため、ファイルそのもののSHA-256（内容から計算される指紋のような値）で縛ることはしません。代わりにこの構造検査で守っています。
- **ダウンロードが途中で切れたファイルは受け付けません。** ヘッダが宣言している本体のバイト数と、実ファイルの大きさが一致するかを最初に検査し、食い違えば「不完全なダウンロードの可能性」と明示して止まります。数時間かけて変換したあとで気づく、という事態を避けるためです。

#### 使い方

元のsafetensorsは自分で入手し、`--st-path` で渡してください（`run.bat download` はこの系統では使えません）。

```
run.bat inspect     --model ltx25-comfyquant --st-path <元のsafetensors>
run.bat convert     --model ltx25-comfyquant --st-path <元のsafetensors>
run.bat self-verify --model ltx25-comfyquant --st-path <元のsafetensors>
```

- `inspect` は変換せずに受け入れ検査と棚卸しだけを行う下見です。`convert` は同じ検査を自分で行うので、いきなり `convert` から始めても構いません。
- 出力先の既定は `output\<元のファイル名>-<型>.gguf` です。同じ場所に次の2つのファイルが並びます。
  - `～.gguf.inventory.json` — 元ファイルの棚卸し表（テンソルの一覧と量子化形式の内訳）。
  - `～.gguf.manifest.json` — 変換の来歴（元ファイルのSHA-256とサイズ、使った型、出力のSHA-256、逆量子化の細かい約束事）。**来歴はこのファイルにだけ書き、GGUF本体には書きません。**
- `--quant-type` で出力の型を選べます。既定の `Q6_K` は**誤差を小さくすることを優先**した設定、`Q4_K` は公式GGUFと同一の構造・同一のサイズになる設定です。
- `--expect-sha256 <値>` を付けると、元ファイルのSHA-256が指定値と一致することも併せて確かめます（任意）。

出力サイズの目安（公式と同じ4,349本を出力する場合）:

| `--quant-type` | 出力サイズ |
|---|---|
| `Q6_K`（既定） | 19,631,143,808バイト（約19.6GB） |
| `Q4_K` | 14,738,670,464バイト（約14.7GB。公式GGUFと同じ） |

#### できあがったGGUFの置き場所

バックエンド（Nz-Videomni）の `models\LTX25\Weights\` の直下へコピーするだけで、モデル選択のドロップダウンに現れます（考え方は後述の「変換したGGUFの配置と選択」と同じで、LTX 2.3ではなくLTX 2.5の置き場所を使う点だけが違います）。LTX 2.5用の配置についてはNz-Videomni側のREADMEの該当節も参照してください。

#### 利用条件

元のファイルは第三者が配布しているLTX 2.5の派生物で、LTX-2.xコミュニティライセンスの下にあります。**この系統の出力は自分で使うためだけのものです。再配布しないでください。** 入手元（Civitaiなど）の配布条件と上流ライセンスの遵守は、利用者の責任になります。

## セットアップ

1. `setup.bat` をダブルクリック、またはコマンドプロンプトから実行してください。
   - このディレクトリ内に `.venv` という名前のPython仮想環境を作成します（ホスト側のPython環境は変更しません）。
   - 既に `.venv` がある場合は作成をスキップし、依存パッケージのインストールのみ行います。
   - `requirements.txt` に記載された依存パッケージ（gguf, numpy, safetensors, huggingface_hub, tqdm, pytest）をインストールします。torch（PyTorch）はインストールしません。変換処理はCPU上でsafetensors/numpy/ggufライブラリのみを使って行うためです。
2. 基となるPythonインタープリタ（Python 3.11以上が必要）は次の優先順位で自動選択されます: 環境変数 `NZKONV_PYTHON`（Python 3.11以上の実行ファイルのパスを指定）→ `py -3.12` → `py -3.11` → `py -3.13` → バックエンド（Nz-Videomni）に同梱されているCPython 3.12（読み取り専用で借用）。
3. 注意: バックエンド同梱のCPythonが選ばれた場合、作成される `.venv` はバックエンドの `.python` ディレクトリを参照します（`.venv\pyvenv.cfg` に記録されます）。バックエンドのフォルダを移動・削除するとこの仮想環境は壊れるため、その場合は `.venv` を削除して `setup.bat` を再実行してください。

## 使い方

すべてのコマンドは `run.bat` 経由で実行します。`run.bat` は `.venv` が無ければ「run setup.bat first」と表示して終了します。

```
run.bat download          元のsafetensorsファイルをHugging Face Hubからダウンロードする
run.bat extract-typemap   参照GGUF（config.tomlのreference.gguf_path）からテンソルの型・形状の対応表（typemap）を抽出する
run.bat convert           safetensorsを読み込み、typemapに従ってQ4_K_M GGUFへ変換する
run.bat verify            生成したGGUFの構造・テンソル形状が参照GGUFと一致するか検証する
run.bat all               上の4つ（download → extract-typemap → convert → verify）を順に実行する。
                          途中で失敗したらそこで止まる。typemapが既にある場合はextract-typemapを飛ばす
                          （--force-typemapを付けると飛ばさずに作り直す）
run.bat convert-vae       PrunaVAED（枝刈り版の映像VAE）をダウンロードし、デコーダ部だけを
                          取り出した約690MBのsafetensorsを作る（下記「PrunaVAEDの変換」参照）
```

設定は `config.toml` にまとめてあります（ダウンロード元のrepo_id、参照GGUFのパス、出力先など）。

## PrunaVAEDの変換（`run.bat convert-vae`）

PrunaVAEDは、Pruna AIが公開しているLTX-2.3専用の映像VAEデコーダです。VAE（Variational Autoencoder。潜在表現と実際の映像を相互に変換する部分）のうち「潜在表現から映像を復元する側」だけを軽くしたもので、バックエンドの設定画面から任意で選べるようにするために変換します。

```
run.bat convert-vae
```

このコマンドは次を一気に行います。

1. Hugging Faceの `PrunaAI/PrunaVAED` から、**コミットハッシュで固定した版**の重みファイル（約1.33GB）を取得します。取得後にファイルのSHA-256（内容から計算される指紋のような値）を照合し、少しでも違えばそこで停止します。v1とv2はファイルの構造が同一で中身の値だけが違うため、サイズだけでは取り違えを防げないからです。
2. 1.33GBのうち**デコーダ部分の100本のテンソルだけ**を抜き出し、バックエンドがそのまま読み込めるキー名へ付け替えます。エンコーダ部分は元のモデルと完全に同一なので配布しません。
3. 潜在表現の統計量（`latents_mean` / `latents_std`）を2本加えて、合計102本・約690MBのsafetensorsとして書き出します。**値は1ビットも変換していません**（BF16のまま生バイトをコピーしています）。
4. 書き出したファイルを自分で読み直し、8項目の自己検証を行います。テンソル本数・キー名の集合・形状・パラメータ総数に加えて、**全テンソルについて元ファイルの該当領域とMD5を突き合わせ**ます。本数や形状だけの検査では「同じ形のテンソルどうしの取り違え」を見逃すためです。1項目でも失敗すれば異常終了し、そのファイルは使いません。

出力は `output\PrunaVAED-decoder-bf16.safetensors` です。ファイル名に `video` という文字列が入っていないのは意図的で、バックエンドが `video` を含むファイルを「差し替え可能な映像VAE本体」として自動登録してしまうためです（デコーダ単体を本体として選ばれると壊れます）。

実行結果の全文は `Docs/VERIFICATION.md` の「7. PrunaVAEDデコーダ変換の検証記録」を参照してください。

## GPU排他ルール（重要）

- 変換処理（download / extract-typemap / convert / verify）はすべてCPU上で行われ、GPUを使用しません。
- ただし、変換後のGGUFを実際にバックエンドで読み込んで生成テスト（E2Eテスト）を行う際は、GPUメモリを他のプロセスと共有できません。E2E生成テストを実行している間は、他のGPUを使うプロセス（AviUtl2本体でのプレビュー生成など）を同時に起動しないでください。

## ディスク使用量の目安

GGUF変換の場合:

- ダウンロードする元のsafetensors（Sulphur 2 base, bf16）: 約43GB
- 変換後の出力GGUF（Q4_K_M）: 約16.5GB
- 合計で60GB程度の空き容量を確保してから作業を始めてください。

PrunaVAED変換（`convert-vae`）の場合:

- ダウンロードする元のsafetensors（PrunaVAED v2, bf16）: 約1.33GB
- 変換後の出力safetensors: 約690MB
- 合計で2GB程度あれば足ります。

## ディレクトリ構成

```
safetensors/   ダウンロードした元のsafetensorsを置く場所（.gitignore対象）
output/        変換後のGGUF・safetensorsの出力先（.gitignore対象）
typemap/       参照GGUFから抽出した型マップJSON（コミット対象）
src/converter/ 変換ツール本体のPythonパッケージ
src/converter/data/
               変換後ファイルに埋め込むライセンス全文（コミット対象）
tests/         テストコード
Docs/          設計メモ等のドキュメント
```

## 変換したGGUFの配置と選択

`run.bat convert` の出力（`output\*.gguf`）は、このツールのフォルダに置いてあるだけでは
バックエンド（Nz-Videomni）から認識されません。バックエンドに反映するには以下の
手順を行ってください。

1. `output\*.gguf` を、バックエンドの `Nz-Videomni\models\LTX23\Weights\` の
   **直下**へコピーします（例: `models\LTX23\Weights\Sulphur-2-base-distil-Q4_K_M.gguf`）。
   既存の参照GGUF `LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf` と同じ階層に、別のファイル名で
   並べる形になります。**同名でコピーして上書きしないでください。**
   ※かつては `LTX-2.3-distilled-1.1\` のようなサブフォルダを作って置く手順でしたが、
   この入れ子構造は2026-07-26に廃止しました（再ホスト先のリポジトリが最初から直下の構造で
   持つようになり、インストーラ側の平坦化処理も削除済み）。現在の正しい配置は直下です。
2. バックエンドは `models\LTX23\Weights\` 配下を**再帰的に**スキャンして`.gguf`ファイルを
   自動登録する設計なので、サブフォルダに入れても認識自体はされます。ただし既定パスと
   ドキュメントは直下を前提にしているため、直下に置いてください。登録名はファイル名
   （拡張子を除いた部分）がそのまま使われ、`config.yaml`の編集は不要です。
3. バックエンドを起動（または再起動）した状態で、UIの「Models」設定、またはAPI
   `POST /pipeline/load` でこの登録名を選べば、そのGGUFが読み込まれます。

このツールで取得した蒸留LoRA（`safetensors\distill_loras\` 配下）も、同様に
`Nz-Videomni\models\LTX23\StyleLoRA\` へコピーするだけでバックエンドが自動認識します。
生成時はAPIの `loras:[{name, strength}]`、またはGradio UIのプロンプト内
`<lora:名前:強度>` という記法で指定します（こちらも設定ファイルの編集は不要です）。

実際に配置・選択して動画生成まで確認した手順とログは、`Docs/VERIFICATION.md`の
「5. E2E段階B」を参照してください。

## 変換実績

このツールを使って変換したモデルと、そこまで確認できている範囲の記録です。
Sulphur 2 baseと10Erosは、実機で動画生成まで確認済みです。

- Sulphur 2 base: 変換したGGUFは `https://huggingface.co/Rootport/Nz-Sulphur2` から
  ダウンロードできます。
- 10Eros: 変換に成功しています。
- REDGraft（コミュニティ製の量子化済みLTX 2.5・`--model ltx25-comfyquant`）: `Q6_K` で
  変換済みです。出力の大きさは上の「出力サイズの目安」の表にある `Q6_K` の値ちょうどで、
  `self-verify` にも合格しました（出力のSHA-256は `66ed7bf1` で始まる値）。逆量子化した
  結果を公式版と層ごとに数値照合するゲートにも、全件合格しています。**ただし実機での
  動画生成はまだ確認していません。** 数値と手順は `Docs/VERIFICATION.md` の
  「8. ltx25-comfyquant」を参照してください。

## ライセンス

- 本リポジトリ自体は Apache License 2.0 で公開しています。全文は `LICENSE` を
  参照してください。
- `src/converter/quant_kernels.py` の一部（K-quant量子化アルゴリズムの探索処理）は、
  llama.cpp（MIT License）のC実装を移植したものです。詳細はファイル冒頭のコメントを
  参照してください。
- `src/converter/data/LTX-2-Community-License.txt` は、変換後のGGUFファイルの
  メタデータへ埋め込むためのライセンス全文データであり、**本ツール自体のライセンス
  ではありません**（本ツール自体のライセンスはApache License 2.0です）。
