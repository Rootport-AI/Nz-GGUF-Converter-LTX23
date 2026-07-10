# Nz-GGUF-Converter-LTX23

LTX 2.3のファインチューンモデル「Sulphur 2 base」のsafetensors（PyTorchの重みファイル形式、約43GB）を、GGUF（GPT-Generated Unified Format。llama.cpp系のツールで使われる量子化モデル形式）のQ4_K_M（4bit量子化の一種。重みを4bitに圧縮しつつ精度劣化を抑える方式）形式に変換する独立ツールです。

変換後のGGUFは、既存のバックエンド（Nz-LTX23-backend）が読み込んでいるLTX-2.3-22B-distilled-1.1-Q4_K_M.ggufと同じ構造・型マップに揃えることで、バックエンド側の推論コードを変更せずに差し替えられるようにします。

## セットアップ

1. `setup.bat` をダブルクリック、またはコマンドプロンプトから実行してください。
   - このディレクトリ内に `.venv` という名前のPython仮想環境を作成します（ホスト側のPython環境は変更しません）。
   - 既に `.venv` がある場合は作成をスキップし、依存パッケージのインストールのみ行います。
   - `requirements.txt` に記載された依存パッケージ（gguf, numpy, safetensors, huggingface_hub, tqdm, pytest）をインストールします。torch（PyTorch）はインストールしません。変換処理はCPU上でsafetensors/numpy/ggufライブラリのみを使って行うためです。
2. 基となるPythonインタープリタ（Python 3.11以上が必要）は次の優先順位で自動選択されます: 環境変数 `NZKONV_PYTHON`（Python 3.11以上の実行ファイルのパスを指定）→ `py -3.12` → `py -3.11` → `py -3.13` → バックエンド（Nz-LTX23-backend）に同梱されているCPython 3.12（読み取り専用で借用）。
3. 注意: バックエンド同梱のCPythonが選ばれた場合、作成される `.venv` はバックエンドの `.python` ディレクトリを参照します（`.venv\pyvenv.cfg` に記録されます）。バックエンドのフォルダを移動・削除するとこの仮想環境は壊れるため、その場合は `.venv` を削除して `setup.bat` を再実行してください。

## 使い方

すべてのコマンドは `run.bat` 経由で実行します。`run.bat` は `.venv` が無ければ「run setup.bat first」と表示して終了します。

```
run.bat download          元のsafetensorsファイルをHugging Face Hubからダウンロードする
run.bat extract-typemap   参照GGUF（config.tomlのreference.gguf_path）からテンソルの型・形状の対応表（typemap）を抽出する
run.bat convert           safetensorsを読み込み、typemapに従ってQ4_K_M GGUFへ変換する
run.bat verify            生成したGGUFの構造・テンソル形状が参照GGUFと一致するか検証する
```

設定は `config.toml` にまとめてあります（ダウンロード元のrepo_id、参照GGUFのパス、出力先など）。

## GPU排他ルール（重要）

- 変換処理（download / extract-typemap / convert / verify）はすべてCPU上で行われ、GPUを使用しません。
- ただし、変換後のGGUFを実際にバックエンドで読み込んで生成テスト（E2Eテスト）を行う際は、GPUメモリを他のプロセスと共有できません。E2E生成テストを実行している間は、他のGPUを使うプロセス（AviUtl2本体でのプレビュー生成など）を同時に起動しないでください。

## ディスク使用量の目安

- ダウンロードする元のsafetensors（Sulphur 2 base, bf16）: 約43GB
- 変換後の出力GGUF（Q4_K_M）: 約16.5GB
- 合計で60GB程度の空き容量を確保してから作業を始めてください。

## ディレクトリ構成

```
safetensors/   ダウンロードした元のsafetensorsを置く場所（.gitignore対象）
output/        変換後のGGUFの出力先（.gitignore対象）
typemap/       参照GGUFから抽出した型マップJSON（コミット対象）
src/converter/ 変換ツール本体のPythonパッケージ
tests/         テストコード
Docs/          設計メモ等のドキュメント
```
