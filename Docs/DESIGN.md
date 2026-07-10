# 設計解説（DESIGN）

このドキュメントは、Sulphur-2-base（LTX-2.3のファインチューンモデル）のsafetensors
（PyTorchの重みファイル形式）を GGUF（GPT-Generated Unified Format。llama.cpp系の
ツールで使われる量子化モデル形式）の Q4_K_M（4bit量子化を中心に、テンソルごとに
異なるビット幅を混在させる「mixed」形式）へ変換するツールについて、なぜこの設計に
なっているかを説明します。実装そのものの読み方は各モジュールのdocstringに譲り、
ここでは「なぜこの方式を選んだか」という設計判断を中心にまとめます。

## 1. 背景と目的

既存のバックエンド（Nz-LTX23-backend）は、LTX-2.3の公式配布モデルを変換した
`LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`（以下「参照GGUF」）を読み込んで動画生成
（DiT、動画生成の本体となるトランスフォーマー、を使った拡散モデルの推論）を行って
います。オーナーが使いたいのはSulphur-2-baseというファインチューン版で、こちらは
Hugging Face上でsafetensors形式（bf16、約43GB）でのみ配布されています。

バックエンドの推論コード（`engine/gguf/loader_service.py` /
`engine/gguf/quant_service.py`）は「参照GGUFと同じテンソル名・同じ量子化タイプ・
同じKVメタデータ構造を持つGGUFファイル」を読み込む前提で書かれています。そのため
このツールのゴールは、**Sulphur-2-baseの重みの値を保ちながら、参照GGUFと構造的に
瓜二つのGGUFを作ること**です。構造さえ揃えれば、バックエンド側のコードは一切
変更せずにモデルを差し替えられます。

## 2. なぜこの方式か（設計判断の理由）

### 2-1. なぜ独自の量子化カーネルを実装したか

GGUFを扱うPythonライブラリ`gguf-py`（本ツールが依存している`gguf`パッケージ）は、
Q4_K / Q5_K / Q6_K（いずれもK-quantと呼ばれる系列の4〜6bit量子化）について
「復元（dequantize、圧縮データから元の浮動小数点値に戻すこと）」は実装している
ものの、「量子化（quantize、圧縮データを新たに書き込むこと）」は実装していません。
gguf-pyだけでは変換ツールを完成させられないということです。

そこで、llama.cpp（GGUF形式を生み出したC言語プロジェクト）のC言語による参照実装
（`ggml/src/ggml-quants.c`）をNumPyへ移植しました。移植の経緯・対応表・検証結果は
`Docs/QUANT_KERNELS.md` に詳しくまとめてあります。要点だけ言うと、単に「それらしい
量子化」を実装したのではなく、C実装の演算順序（浮動小数の丸め誤差まで含む）を
ビット単位で再現するところまでこだわっています。これは、量子化アルゴリズムの
探索処理（最適なスケール値を掃引して選ぶ処理）が丸め誤差に敏感で、実装が少しでも
ずれると「見た目は近いが微妙に違う」量子化結果になってしまうためです。

### 2-2. なぜ「参照GGUF由来のtypemap」を唯一の正としたか

各テンソルをQ4_K/Q5_K/Q6_K/F32/BF16のどれに量子化するかは、LTX-2.3のアーキテクチャ
上のテンソルの役割（AdaLN層か、通常のLinear層か、ブロックの先頭/末尾かなど）に
よって決まっており、その判断ロジックは公開されていません。しかし参照GGUF自体が
「どのテンソルがどの型で量子化されているか」の答えそのものを持っています。

そこで`src/converter/typemap.py`が参照GGUFを読み、テンソル名の並び順を保ったまま
「テンソル名 → GGML量子化タイプ」の対応表（typemap）をJSONとして抽出し、これを
変換処理の唯一の正（single source of truth）として使います。`classify_by_rule`と
いうルールベースの分類器も同モジュールに実装していますが、これは変換処理そのもの
では使われません。あくまで「typemapの決定がどんな規則で説明できるか」を監査する
ための補助ツールです（`converter extract-typemap`実行時に自動で監査ログが出ます）。
Sulphur-2-baseは参照GGUFと同じLTX-2.3アーキテクチャのファインチューンなので、
テンソル名の集合が完全一致し、この転用が成立します。

### 2-3. なぜ2パス・ストリーミング書き込みなのか

元のsafetensorsファイルは約43GB、出力のGGUFも約17.8GBあります。これを愚直に
「全部読み込んでから書き出す」実装にすると、数十GB単位のメモリを一時的に必要と
し、非力なマシンでは失敗しかねません。そこで`src/converter/convert.py`は次の
2パス構成にしています。

1. **パス1（メタデータのみ）**: typemapの順番どおりに、各テンソルの名前・形状・
   バイトサイズだけを`gguf.GGUFWriter`に登録します。テンソルの中身（実データ）は
   一切触りません。登録が終わったらヘッダー・KVメタデータ・テンソル情報ブロックを
   ファイルへ書き出します。
2. **パス2（データストリーミング）**: 同じ順番でtypemapをもう一度なぞり、
   safetensorsから該当テンソルの生バイト列を読み込み、必要なら量子化して、
   GGUFへ書き込みます。1テンソル分を処理したらすぐに参照を破棄し、次のテンソルの
   処理に移ります。常にメモリ上に「1テンソル分＋αの定数」しか保持しません。

この設計により、変換処理に必要なメモリはテンソル1個分（最大でも数百MB程度）で
済み、43GBのファイルサイズやマシンの搭載RAMに関わらず安定して動作します。

### 2-4. なぜsafetensorsライブラリの標準APIを使わず生バイト読み込みにしたか

safetensors 0.7.0の`framework="numpy"`はBF16（bfloat16）テンソルを読み込めません
（NumPyにbfloat16型が存在せず、`ml_dtypes`もインストールしていないため）。かと
いって`framework="pt"`（PyTorch経由）を使うにはtorchのインストールが必要になり、
「変換処理はCPU・軽量な依存関係だけで完結させる」という方針（後述2-5）に反します。

そこで`convert.py`内の`_SafetensorsRaw`クラスが、safetensorsのファイルヘッダー
（JSON形式、小さい）だけを読み、各テンソルの生バイト列は`data_offsets`を使って
`seek`/`read`で直接読み出します。BF16の値は「float32の上位16bit」そのものなので、
`uint16 << 16`をfloat32として解釈するだけで正確にF32へ復元でき、依存ライブラリを
増やさずに済みます。

### 2-5. なぜtorch（PyTorch）に依存しないのか

変換処理そのものはCPU上でのバイト列操作と量子化演算だけなので、NumPy・
gguf-py・safetensorsだけで完結します。torchは数GBの追加インストールになる上、
CUDA関連の初期化コストもあるため、変換ツール単体では意図的に導入していません
（`setup.bat`のコメント、`requirements.txt`参照）。E2E検証（段階A、後述）で
バックエンドのdequantコードを借用する際だけ、バックエンド同梱のvenv
（`Nz-LTX23-backend\.venv-engine`、torchインストール済み）を間借りします。

## 3. モジュール構成

```
src/converter/
├── __main__.py       python -m converter のエントリポイント（cli.mainを呼ぶだけ）
├── cli.py             サブコマンド（download/extract-typemap/convert/verify/all）の配線
├── download.py        Hugging Face Hubから元safetensorsをダウンロード
├── typemap.py         参照GGUFからtypemapを抽出・保存・監査（classify_by_ruleを含む）
├── metadata.py        safetensorsの__metadata__ブロック → GGUFのKVメタデータへの転記
├── quant_kernels.py   Q4_K/Q5_K/Q6_K量子化カーネル（llama.cpp C実装のNumPy移植）
├── convert.py         2パス・ストリーミング変換の本体（上記すべてを束ねる）
└── verify.py          出力GGUFと参照GGUFの構造比較（本ドキュメントの姉妹編で使用）
```

依存の向きは一方向です。`convert.py`が`metadata.py`・`quant_kernels.py`・
`typemap.py`を呼び出す側で、これらのモジュール自身は互いに依存しません。
`verify.py`は変換処理から独立しており、出来上がったGGUFファイル2つ（出力と参照）
だけを見て判定します。

## 4. 処理フロー図

### 4-1. 変換全体（`run.bat all` = download → extract-typemap → convert → verify）

```
[Hugging Face Hub]
      │ download.py
      ▼
safetensors/sulphur_distil_bf16.safetensors (約43GB, bf16)

[参照GGUF] ──typemap.py(extract_typemap)──▶ typemap/ltx23_q4km_typemap.json
 (Nz-LTX23-backend                         (テンソル名→GGML型・形状のJSON、
  同梱の既存ファイル)                        4444テンソル分、コミット対象)

safetensors + typemap ──convert.py(convert)──▶ output/Sulphur-2-base-distil-Q4_K_M.gguf
                                                (約17.8GB、テンソル4444個)

output GGUF + 参照GGUF ──verify.py(verify_structure)──▶ 構造検証レポート(PASS/FAIL)
```

### 4-2. `convert()`内部の2パスの流れ（1テンソルあたり）

```
typemapの1レコード (name, ggml_type, shape_logical, nbytes)
        │
        ├─ パス1: writer.add_tensor_info(...)  ← データは触らない
        │
        └─ パス2: raw_key = diffusion[name]
                  ├─ F32/BF16 → reader.get_f32() / get_bf16_bytes()  (バイト変換のみ)
                  └─ Q4_K/Q5_K/Q6_K → reader.get_f32() → quant_kernels.quantize()
                          │
                          ▼
                  writer.write_tensor_data(payload)
                  del payload   ← 次のテンソルへ進む前に必ず解放
```

### 4-3. metadata.pyの役割

safetensorsの`__metadata__`ブロックにある`config`キー（モデル設定のJSON文字列。
バックエンドがモデルを構築する際に必須）を検証し、`general.quantization_version`
（固定値2）・`general.file_type`（固定値15）とあわせてGGUFのKVへ書き込みます。
`config`が欠落・空・JSONとして不正な場合は変換時点でエラーにして止めます
（バックエンドが実行時に初めて気づくより、変換時点で気づいたほうが手戻りが
小さいため）。

## 5. 制限事項

量子化カーネル自体の技術的な制限（imatrix非対応、Q5_K/Q6_Kのべき等率が85%前後に
とどまる理由とその品質への影響など）は `Docs/QUANT_KERNELS.md` にまとめてあります。
今回の実変換・検証結果は `Docs/VERIFICATION.md` を参照してください。
