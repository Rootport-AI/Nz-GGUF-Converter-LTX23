# 引き継ぎ資料（HANDOUT）

このドキュメントは、Sulphur-2-base（LTX-2.3のファインチューンモデル）を
GGUF（Q4_K_M形式）へ変換するツールの使い方と、今回の変換作業の結果・注意点を
まとめたものです。技術的な設計の理由は`Docs/DESIGN.md`、検証の詳細は
`Docs/VERIFICATION.md`を参照してください。

## 1. ツールの使い方

### 1-1. 初回セットアップ

`setup.bat`をダブルクリック、またはコマンドプロンプトから実行します。このディレ
クトリ内に`.venv`という専用のPython仮想環境を作り、変換に必要なパッケージ
（gguf、numpy、safetensors、huggingface_hub、tqdm、pytest）をインストールします。
torch（PyTorch、機械学習用の計算ライブラリ）はインストールしません。変換処理は
CPU上でnumpy/ggufライブラリだけを使って行うためです（詳しい理由は
`Docs/DESIGN.md`の「2-5」参照）。

既に`.venv`がある場合は作成をスキップし、依存パッケージの再インストールだけを
行います。

### 1-2. 実行コマンド

すべて`run.bat`経由で実行します。プロジェクトのルートディレクトリ
（`Nz-GGUF-Converter-LTX23`）で実行してください。

```
run.bat all                一連の処理（download→extract-typemap→convert→verify）を通しで実行
run.bat download           元のsafetensorsファイルをHugging Face Hubからダウンロード
run.bat extract-typemap    参照GGUFからテンソルの型・形状の対応表（typemap）を抽出
run.bat convert            safetensorsを読み込み、typemapに従ってQ4_K_M GGUFへ変換
run.bat verify             生成したGGUFの構造が参照GGUFと一致するか検証
```

個別コマンドは、前段の成果物（safetensorsファイル、typemap JSON）が既にある場合に
その工程だけをやり直したい時に使います。設定は`config.toml`にまとまっています
（ダウンロード元、参照GGUFのパス、出力先ファイル名など）。

### 1-3. 生成物の場所

- ダウンロードした元のsafetensors: `safetensors\sulphur_distil_bf16.safetensors`
  （約43GB、`.gitignore`対象なのでコミットされません）
- 変換後の出力GGUF: `output\Sulphur-2-base-distil-Q4_K_M.gguf`
  （約17.8GB、こちらも`.gitignore`対象）
- typemap JSON: `typemap\ltx23_q4km_typemap.json`
  （コミット対象。参照GGUFから抽出したテンソル型マップで、再変換時はこれを
  再利用できます）

今回の実変換では、変換開始が2026-07-10 10:16頃、完了が同日12:50頃で、所要時間は
約2時間34分でした。出力ファイルサイズは17,763,014,976バイトで、参照GGUF
（17,763,015,328バイト）とほぼ同一（差0.002%）です。

### 1-4. 変換したGGUFとLoRAの配置と選択

出力GGUF（`output\*.gguf`）は、このプロジェクトのフォルダに置いてあるだけではバック
エンド（Nz-LTX23-backend）から見えません。バックエンドで使うには次の手順が必要です。

1. `output\*.gguf`を、バックエンドの`Nz-LTX23-backend\models\ltx-2.3-gguf\`配下に
   **新しいサブフォルダを作って**コピーします（既存の`LTX-2.3-distilled-1.1\`と
   兄弟フォルダにし、既存の参照GGUFは上書きしないでください）。例:
   `models\ltx-2.3-gguf\Sulphur-2-base-distil-1.0\Sulphur-2-base-distil-Q4_K_M.gguf`。
2. バックエンドは`models\ltx-2.3-gguf\`配下を**再帰的に**スキャンして`.gguf`ファイルを
   自動登録します（登録名＝ファイル名から拡張子を除いたもの）。コピーするだけで
   反映され、`config.yaml`の編集は不要です。
3. バックエンド起動後、UIの「Models」設定またはAPI `POST /pipeline/load`でこの
   登録名を選べば読み込まれます。

`safetensors\distill_loras\`に取得した蒸留LoRAも、`Nz-LTX23-backend\models\loras\`へ
コピーするだけで自動認識されます。生成時はAPIの`loras:[{name, strength}]`、または
Gradio UIのプロンプト内`<lora:名前:強度>`記法で指定します（同じく設定ファイルの編集は
不要）。

実際の手順とログは`Docs/VERIFICATION.md`の「5. E2E段階B」を参照してください
（GPU実生成スモークまで実施済みです）。

## 2. 43GBのsafetensorsを消してよいタイミング

`safetensors\sulphur_distil_bf16.safetensors`（約43GB）は、以下がすべて済んだ
あとであれば削除して構いません。

1. `run.bat verify`が全項目PASSで完了していること（今回は完了・全PASS済み、
   `Docs/VERIFICATION.md`参照）。
2. できればバックエンドで実際に動画を1本生成できるところまで確認できていると
   なお安心です（`Docs/VERIFICATION.md`の「E2E段階B」手順）。これは今回は
   未実施です。
3. 再ダウンロードの手間を許容できること。削除したあとで再変換したくなった場合は
   `run.bat download`から再取得できますが、Hugging Faceからの約43GBの再ダウン
   ロードが発生します。typemap JSON（`typemap\ltx23_q4km_typemap.json`）は
   小さいファイルなのでコミットしてあり、再ダウンロード後の`convert`はこの
   typemapをそのまま再利用できます（`extract-typemap`をやり直す必要はありません）。

迷う場合は、E2E段階B（実際の動画生成）で問題ないことを確認するまでは残しておく
のが安全です。ディスク容量に余裕があるなら、しばらく様子を見てから消すことを
おすすめします。

## 3. 既知の制限

### 3-1. Q5_K/Q6_Kの「べき等率85%」について（品質問題ではありません）

`Docs/QUANT_KERNELS.md`に詳しい検証結果がありますが、量子化カーネルの動作確認
テストの1つに「参照GGUFの実テンソルを元の値に戻し、もう一度量子化しなおして、
元のバイト列とどれだけ一致するか」を測るものがあります。この一致率（べき等率）
が、Q4_Kでは99.90%とほぼ完全に近いのに対し、Q5_K/Q6_Kでは85%前後にとどまります。

これは**カーネルの不具合ではありません**。K量子化というアルゴリズムの性質上、
「一度圧縮してから戻した値」に対してもう一度最適なスケールを探索すると、元の
探索とは別の（しかし同じくらい妥当な）答えに着地することがあるためです。
Q5_K/Q6_Kのほうがより細かい量子化の刻み幅を持つため、この現象が起きやすくなり
ます。実際に量子化した際の誤差（SQNR、信号対量子化雑音比。大きいほど高品質）は
Q6_Kで72.7dBと非常に高く、品質そのものに問題はありません。「べき等率が低い
=品質が悪い」ではなく、「べき等率」という指標自体がこのアルゴリズムでは
100%になりにくい性質のものだと理解してください。

### 3-2. 変換所要時間の目安

今回の実測では約2時間34分でした。`Docs/QUANT_KERNELS.md`のスループット実測
（Q4_K/Q5_K/Q6_Kそれぞれの量子化速度）から見積もった量子化処理自体の時間は
合計1時間30分程度ですが、実際にはこれに加えてF32/BF16テンソルのコピー・
ディスクI/O（読み書き、約43GB読んで約17.8GB書く）などが乗るため、実測のほうが
長くなります。マシンやディスクの状態によって変動しうるため、「約3時間程度は
見ておく」くらいの目安で捉えてください。

### 3-3. `run.bat`のインタープリタがバックエンド同梱CPythonに依存しうる件

`setup.bat`は、Pythonインタープリタを次の優先順位で自動選択します。

1. 環境変数`NZKONV_PYTHON`（明示的に指定した場合）
2. `py -3.12` → `py -3.11` → `py -3.13`（Pythonランチャー経由）
3. 上記がどれも見つからない場合、**バックエンド（Nz-LTX23-backend）に同梱されて
   いるCPython 3.12を読み取り専用で借用**します。

3のフォールバックが使われた場合、作成される`.venv`は内部的にバックエンドの
`.python`ディレクトリを参照する形になります（`.venv\pyvenv.cfg`に記録されます）。
つまり、**バックエンドのフォルダを将来移動・削除すると、このツールの`.venv`が
壊れます**。その場合は`.venv`フォルダを削除して`setup.bat`をもう一度実行すれば
再構築されますが、上記1か2の方法でPythonが用意できる環境であれば、そちらを
優先したほうが独立性が保てます。

今回の環境では実際に3のフォールバックが使われており、`.venv\pyvenv.cfg`を見ると
`home = S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-LTX23-backend\.python\cpython-3.12.9-windows-x86_64-none`
と、バックエンド同梱のCPythonを指していることが確認できます。つまり**この
プロジェクトの`.venv`は、現状すでにバックエンドフォルダに依存しています**。
バックエンドを移動・削除する予定がある場合は、事前に`NZKONV_PYTHON`環境変数か
`py`ランチャーで独立したPython 3.11以上を用意し、`.venv`を作り直しておくことを
おすすめします。

## 4. トラブルシューティング

### `run.bat`が「.venv not found. Please run setup.bat first.」と表示される

`setup.bat`をまだ実行していないか、`.venv`フォルダを削除した後です。
`setup.bat`を実行してください。

### `run.bat convert`が「Source safetensors not found」で失敗する

`safetensors\sulphur_distil_bf16.safetensors`が存在しません。`run.bat download`
を先に実行してください（削除済みの場合や、初回セットアップがまだの場合）。

### `run.bat verify`が失敗する（FAILが出る）

`config.toml`の`[reference].gguf_path`が指しているバックエンド側の参照GGUF
（`Nz-LTX23-backend\models\ltx-2.3-gguf\LTX-2.3-distilled-1.1\LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`）
が存在し、破損していないか確認してください。それでも失敗する場合は、
どの項目が`FAIL`になったかを確認し（`tensor_count`/`tensor_names_order`/
`tensor_types_shapes`/`kv_keys`など）、`src/converter/verify.py`のdocstringで
各チェックの意味を確認したうえで、変換をやり直す（`run.bat convert`から）ことを
検討してください。

### E2E段階A（`scripts\e2e_load_check.py`）を再実行したい

このスクリプトはバックエンド同梱のtorch環境が必要なので、**バックエンドの
venvで実行してください**（このプロジェクトの`.venv`ではありません）。

```
S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-LTX23-backend\.venv-engine\Scripts\python.exe scripts\e2e_load_check.py
```

プロジェクトのルート（`Nz-GGUF-Converter-LTX23`）で実行してください。
`--max-tensors N`オプションを付けると、最初のN個のテンソルだけで動作確認する
簡易実行ができます。

### 変換にとても時間がかかる／途中で止まっているように見える

`run.bat convert`は`tqdm`（進捗バー）でテンソルごとの進捗を表示します。
1テンソルずつ処理する設計上、進捗バーがゆっくり進むのは正常です（3-2節の
所要時間目安を参照）。ディスクの空き容量不足（元ファイル約43GB＋出力約17.8GBで
合計60GB程度必要）で書き込みが止まっている可能性もあるため、ディスクの空き
容量も確認してください。
