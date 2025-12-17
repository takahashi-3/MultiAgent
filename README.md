# 内容
- DEIM2026を対象
  - 「これまでの訓練状況＋顧客役の性格を考慮した**指示役LLM**による動的なタスク割り当て」の実現
  - 音声入出力を使用した接客訓練の実現

# 実行方法
- **`pyenv`がインストール済みであることを想定**
- `git clone -b DEIM2026 http...`　を実行して、リポジトリをクローン
- `MultiAgent`というディレクトリができるので、そちらに移動し `pyenv local 3.11.1` を実行して使用する`python`のバージョンを決定
  - `3.11.1` がインストールされていない場合には、`pyenv install 3.11.1` を実行
- 以下の作業は`venv`など仮想環境を作成して、行ってください
  - `pip install -r requirements.txt` を実行
  - `LangGraphMultiAgent_ActivateNode.py` を実行して、訓練開始
