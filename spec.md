# RTMO -> BlockGCN 動作分類システム 仕様書

## 1. 文書の目的

本書は、`RTMO` による人物・骨格推定結果を入力として、Skeleton-based GCN 系モデルで人物ごとの動作ラベルを分類する PoC システムの仕様を定義する。

初期実装では `CTR-GCN` を優先し、将来的な本命アーキテクチャとして `BlockGCN` を視野に入れる。

本仕様は以下を目的とする。

* 現時点で合意できている実装方針を明文化する
* すぐに実装へ着手できる初期設定を定める
* 将来拡張の方向性を先に整理しておく
* まだ未確定な論点を `保留事項` として切り出す

本書は [preresearch.md](/Users/takanotaisei/Documents/NI/action-det/preresearch.md) をもとに、今回の議論内容を反映して具体化したものである。

---

## 2. PoC のスコープ

### 2.1 対象

PoC では、固定カメラ映像から人物ごとの動作をリアルタイムに近い形で分類する。

対象タスクは以下とする。

1. フレーム入力を取得する
2. `RTMO` で人物ごとの 2D 骨格を推定する
3. 軽量な ID 割当ロジックで短期的な人物対応を維持する
4. `track_id` ごとに骨格系列をバッファする
5. 初期実装では `CTR-GCN` で人物ごとの動作ラベルを推論する
6. 将来的に `BlockGCN` へ移行可能な形で入出力を設計する
6. 平滑化後のラベルを JSON と可視化に出力する

### 2.2 非対象

PoC の初期実装では以下を対象外とする。

* RGB を直接入力する動画分類
* `approaching_zone` `leaving_zone` `in/out` のような移動ベースイベントの実装
* 長期 ReID を伴う本格 tracking
* 複数カメラ統合
* 物体依存の細粒度動作認識
* 自然言語説明生成

---

## 3. 基本方針

### 3.1 採用構成

初期 PoC の採用構成は以下とする。

* 前段: `RTMO`
* 中間: `軽量 ID 割当ロジック`
* 後段の初期本命: `CTR-GCN`
* 後段の次点候補: `HD-GCN`
* 後段の将来本命: `BlockGCN`
* 骨格表現: `2D skeleton`

### 3.2 2D skeleton を採用する理由

本システムでは `RTMO` の出力である 2D keypoints をそのまま主入力とする。

理由は以下である。

* RTMO と自然に接続できる
* 入力が軽量でリアルタイム性に有利
* PoC 対象ラベルでは 2D でも成立しやすい
* 将来の独自データ収集でも取り回しやすい

3D skeleton は PoC では採用しない。

### 3.3 動作ラベルと移動イベントの分離

本システムでは、Skeleton-based GCN backend が担当する分類対象と、後段ルールベースで扱うイベントを分離する。

GCN backend の担当:

* `standing`
* `walking`
* `sitting`
* `crouching`
* `hand_raising`

後段ルールベースの将来候補:

* `approaching_zone`
* `leaving_zone`
* `in/out`
* 滞在時間
* エリア滞留

PoC では後者は実装しない。

---

## 4. 想定ユースケース

### 4.1 初期ユースケース

* 店舗や施設内における人物行動の簡易把握
* 人物ごとの状態ラベル付与
* 将来の人数カウントや移動イベント実装の土台作り

### 4.2 取り扱う初期ラベル

PoC の初期ラベルは以下の 5 クラスを基本とする。

* `standing`
* `walking`
* `sitting`
* `crouching`
* `hand_raising`

必要に応じて将来追加候補とする。

* `bending`
* `lying`
* `hand_waving`
* `falling`

---

## 5. システム全体アーキテクチャ

基本フローは以下とする。

`映像入力 -> RTMO -> 軽量 ID 割当 -> track ごとの時系列バッファ -> CTR-GCN (初期) -> ラベル平滑化 -> JSON/可視化出力`

### 5.1 モデルロードマップ

後段モデルの採用順は以下とする。

* 初期実装本命: `CTR-GCN`
* 次点候補: `HD-GCN`
* 将来本命: `BlockGCN`

採用理由:

* `CTR-GCN` は公式実装と pretrained weight が利用しやすく、初期統合のハードルが低い
* `HD-GCN` も公式実装と pretrained weight を持つが、初期統合コストは `CTR-GCN` よりやや高い
* `BlockGCN` は将来本命だが、公開 weight の扱いと RTMO 2D 入力への適用方針を別途詰める必要がある

### 5.2 映像入力

初期入力条件は以下を想定する。

* 入力: `USB camera` または `video file`
* 解像度: `1280x720` を初期標準
* FPS: `15-30`
* カメラ: 固定カメラ
* 画角: 全身が望ましい。最低でも上半身主要関節が安定観測できること

### 5.3 前段推論

`RTMO` により各フレームから以下を取得する。

* `bbox`
* `keypoints`
* `keypoint confidence`

PoC では RTMO を人物検出と骨格推定の統合出力源として扱う。
初期モデルは `rtmo-s_8xb32-600e_body7-640x640` を採用し、軽量構成から検証を開始する。

### 5.4 軽量 ID 割当

PoC では外部 tracker を導入せず、`RTMO only + lightweight ID assignment` を採用する。

ID 割当では以下の情報を用いて、前フレームとの人物対応を行う。

* bbox の `IoU`
* bbox 中心距離
* 可視 keypoint の平均距離

初期実装では以下を基本とする。

* 対応付け方式: `Hungarian matching`
* `max_missing_frames`: `5-10`
* 短時間遮蔽時は track を即破棄しない

ここでの目的は長期追跡ではなく、`BlockGCN` 入力用の短期一貫性確保である。

### 5.5 時系列バッファ

`track_id` ごとにリングバッファを持つ。

* バッファ単位: `1 track_id = 1 person clip`
* 格納内容: `2D keypoints + confidence`
* バッファ長: `clip_len`
* 更新方式: スライディングウィンドウ

---

## 6. 初期分類器と入力仕様

### 6.1 入力単位

PoC では `single-person, single-clip` 入力を採用する。

* `1 input = 1 track_id の連続骨格系列`
* 同時人物が複数いても、推論入力は人物ごとに独立させる
* 周辺人物文脈は PoC では入力しない

### 6.2 初期分類器の出力ラベル

初期実装では pretrained model のラベル空間をそのまま扱い、`NTU label` を debug overlay と JSON に出力する。

初期方針:

* pretrained label space: `NTU RGB+D 120`
* debug 表示: `ntu_label`
* 将来の最終ラベル空間: `standing / walking / sitting / crouching / hand_raising`

この方針により、初期段階では pretrained model を流用しつつ、将来的に独自 5 クラスへ fine-tune または head 差し替えを行う。

### 6.3 入力モダリティ

PoC の初期実装では `Joint only` を採用する。

理由は以下である。

* `BlockGCN` 論文では 4-stream が最良だが、`Joint only` でも強い性能が出ている
* 実装複雑性を抑えながら PoC を早く成立させやすい
* RTMO 出力との接続が最も単純である

将来候補:

* `J + JM`
* `J + B + JM + BM`

### 6.4 入力テンソル設計

PoC の初期案は以下とする。

* `num_person = 1`
* `num_joints = 17`
* `in_channels = 3`

チャネル内容:

* `x`
* `y`
* `confidence`

`num_joints = 17` は COCO 系 keypoint 定義を標準とする想定である。

### 6.5 keypoint schema

初期実装では、RTMO から得られる keypoint を `COCO-17` 相当の 17 関節定義へ統一する。

ただし、`CTR-GCN` / `HD-GCN` / `BlockGCN` の公式 pretrained は基本的に `NTU 25-joint skeleton` 前提であるため、初期実装では `COCO-17 -> pseudo NTU-25 adapter` を別途用意する。

補足:

* `NTU-25` は Kinect 由来の skeleton schema であり、一般的な RGB 2D pose estimator の標準出力ではない
* 近い選択肢として `OpenPose BODY_25` は存在するが、`NTU-25` そのものではない
* PoC では RTMO を維持し、adapter で吸収する方針を優先する

想定関節:

* nose
* left_eye
* right_eye
* left_ear
* right_ear
* left_shoulder
* right_shoulder
* left_elbow
* right_elbow
* left_wrist
* right_wrist
* left_hip
* right_hip
* left_knee
* right_knee
* left_ankle
* right_ankle

RTMO の出力順序がこれと異なる場合は、`pose/` 側で明示的に mapping を行う。

整形方針は以下とする。

* `COCO-17` を canonical order として固定する
* RTMO 出力をこの順序へ明示変換する
* 各関節は `(x, y, confidence)` の 3 要素へ統一する
* `mid_hip` は入力関節に含めず、正規化用補助点としてのみ使う
* 最終入力テンソルは `shape = [C, T, V, M] = [3, clip_len, 17, 1]` を基本とする

### 6.5 座標前処理

PoC では後段を人物位置に依存させすぎないため、人物中心基準の正規化を行う。

初期仕様:

* 座標中心: `mid_hip` を優先
* `mid_hip` が不安定な場合は `bbox center` を代替使用
* スケール正規化: `bbox height` を基本とする
* 出力は人物局所座標系へ変換する

この方針により、BlockGCN には「人物の絶対位置」ではなく「姿勢と動き」を主に学習させる。

### 6.6 欠損点処理

RTMO 出力には欠損や低信頼点が含まれるため、以下を初期方針とする。

* `confidence` がしきい値未満の点は欠損候補とみなす
* 初期 PoC では短い欠損に対して `前値保持` を採用する
* 欠損点の `confidence` は保持する
* 長い欠損では track 自体を無効化または再初期化対象とする

初期しきい値は `0.2-0.3` を候補とし、検証で確定する。
初期 PoC では `前値保持 + confidence維持` を正式採用とする。

---

## 7. BlockGCN 採用設定

### 7.1 論文ベースの採用根拠

[`BlockGCN: Redefine Topology Awareness for Skeleton-Based Action Recognition (CVPR 2024)`](/Users/takanotaisei/Documents/NI/action-det/paper/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.pdf) の内容をベースに、以下を採用候補とする。

* `BlockGC` は vanilla GC より少ないパラメータで高性能
* `Joint only` でも強いベースラインになる
* `Static Topological Encoding` は `SPD` が最終採用
* `Dynamic Topological Encoding` は有効
* `groups = 8` がアブレーション上で最良
* `dynamic encoding` は `feature-wise`
* `static encoding` は `shared`
* topological encoding は多層挿入が有効
* barcode vectorization feature size は `64` が最良

### 7.2 初期採用設定

BlockGCN の初期 PoC 設定は以下とする。

```yaml
action:
  model: blockgcn
  stream: joint
  clip_len: 32
  stride: 2
  num_person: 1
  num_joints: 17
  in_channels: 3
  positional_embedding: true
  blockgc_groups: 8
  static_topology: spd_shared
  dynamic_topology: none
  topo_insert_layers: 10
```

### 7.3 解釈上の注意

上記のうち、以下は論文のアブレーションに比較的近い採用である。

* `blockgc_groups = 8`
* `static_topology = spd_shared`
* `topo_insert_layers = 10`

一方で以下は PoC のリアルタイム要件に合わせた工学的調整である。

* `clip_len = 32`
* `stride = 2`
* `Joint only` で開始すること
* `x, y, confidence` を入力チャネルとすること
* 初期 PoC では `Dynamic Topological Encoding` を入れないこと

### 7.4 最小成立構成

PoC の進行を優先するため、実装難易度が高い要素は段階導入を許容する。

最小成立構成は以下とする。

```yaml
action:
  model: blockgcn
  stream: joint
  clip_len: 32
  stride: 2
  num_person: 1
  num_joints: 17
  in_channels: 3
  positional_embedding: true
  blockgc_groups: 8
  static_topology: spd_shared
  dynamic_topology: none
```

この最小成立構成で推論導線を先に確立し、その後に `Dynamic Topological Encoding` を追加して精度差を評価する。

### 7.5 将来拡張設定

PoC 初期版が成立した後、精度改善の追加実験として以下を導入候補とする。

```yaml
action:
  dynamic_topology: feature_wise
  barcode_dim: 64
```

`Dynamic Topological Encoding` は将来の精度改善候補として扱い、初期版の必須要件には含めない。

---

## 8. 時系列窓設計

### 8.1 基本方針

`BlockGCN` 論文では学習時に系列を `64 frames` へ揃えているが、PoC では遅延要件を優先して短めの窓を採用する。

### 8.2 初期推奨値

* `clip_len = 32`
* `stride = 2`
* 推論更新: `2 frames ごと`

### 8.3 採用理由

* 30FPS 環境で約 1.07 秒の文脈を確保できる
* `1-2秒以内` の遅延目標に収めやすい
* `standing / walking / sitting / crouching / hand_raising` の初期ラベルと相性が良い

### 8.4 比較候補

評価時には以下を比較候補とする。

* `clip_len = 24`
* `clip_len = 32`
* `clip_len = 48`

`64` は論文再現確認用には有効だが、PoC 本線では遅延増加の懸念が大きい。

---

## 9. ラベル平滑化

推論結果の揺れを抑えるため、後処理としてラベル平滑化を導入する。

初期方針:

* score smoothing
* moving average または majority vote
* `score threshold`
* 最短継続時間制約

初期案:

* `3回` の連続優勢で確定
* `score < threshold` の場合は前ラベル維持

初期推奨値は以下とする。

* `smooth_confirm_count = 3`
* `score_thr = 0.6`
* `score < threshold` の場合は前ラベルを維持する

詳細なしきい値は評価で最終決定する。

---

## 10. 出力仕様

### 10.1 人物ごとの出力

人物ごとに以下を出力する。

* `track_id`
* `bbox`
* `keypoints`
* `action_label`
* `action_score`
* `timestamp`

必要に応じて補助情報として以下も保持可能とする。

* `raw_action_scores`
* `smoothed_action_label`
* `is_track_stable`
* `state`

### 10.2 デバッグ可視化

PoC では、`JSON出力` と同等以上に `debug visualization` を重視する。

初期実装では、動画または webcam 上に以下を重畳表示できることを必須要件とする。

* 人物 `bbox`
* 骨格 keypoints と skeleton edges
* `track_id`
* `action_label`
* `action_score`
* 必要に応じて `state`

補助表示候補:

* FPS
* frame index
* 現在の `clip_len`

この可視化は、以下の確認を主目的とする。

* RTMO の骨格推定が安定しているか
* ID 割当が破綻していないか
* 動作ラベルが妥当か
* smoothing が期待通りに効いているか

### 10.3 JSON 例

```json
{
  "timestamp": 1710000000.123,
  "persons": [
    {
      "track_id": 3,
      "bbox": [312, 145, 420, 398],
      "action_label": "standing",
      "action_score": 0.91,
      "keypoints": [[0.11, 0.22, 0.95], [0.12, 0.21, 0.91]]
    }
  ]
}
```

PoC では `zone_id` や `in/out event` は出力しない。
また、`unknown` を学習クラスには含めず、必要に応じて後処理状態として `transition` または `uncertain` を扱う。
`JSON出力` は保存・連携用途の補助出力とし、初期デバッグでは overlay 表示を優先する。

---

## 11. ソフトウェア構成

想定ディレクトリ構成は以下とする。

### `video_input/`

* webcam 入力
* 動画ファイル入力

### `pose/`

* RTMO 推論ラッパ
* keypoint schema mapping
* 正規化前処理

### `tracking/`

* 軽量 ID 割当
* track state 管理
* missing frame 管理

### `action/`

* BlockGCN 推論ラッパ
* clip 生成
* 欠損点処理
* ラベル平滑化

### `pipeline/`

* 推論全体制御
* フレーム同期
* バッファ連携

### `util/`

* overlay 描画
* JSON 出力
* ログ保存

### `config/`

* モデルパス
* 閾値
* clip 長
* ラベル定義

### `scripts/`

* `setup.sh`
* checkpoint ダウンロード補助
* 実行補助スクリプト

### ルートファイル

* `README.md`
* `.gitignore`
* 実行エントリポイント

---

## 12. 設定ファイル初期案

```yaml
input:
  source: webcam
  width: 1280
  height: 720
  fps: 30

pose:
  model: rtmo-s
  checkpoint: rtmo-s_8xb32-600e_body7-640x640
  conf_thr: 0.3
  keypoint_schema: coco17

tracking:
  enable_lightweight_tracking: true
  match_method: hungarian
  max_missing_frames: 8
  iou_weight: 0.5
  center_dist_weight: 0.3
  keypoint_dist_weight: 0.2

action:
  model: blockgcn
  stream: joint
  clip_len: 32
  stride: 2
  num_person: 1
  num_joints: 17
  in_channels: 3
  positional_embedding: true
  blockgc_groups: 8
  static_topology: spd_shared
  dynamic_topology: none
  topo_insert_layers: 10
  confidence_mode: input_channel
  score_thr: 0.6
  smooth_confirm_count: 3
  use_unknown_class: false

output:
  draw_overlay: true
  show_track_id: true
  show_action_label: true
  show_action_score: true
  show_fps: true
  save_json: true
  save_video: false
```

---

## 13. 想定実行環境

本システムの実行対象は、現在作業している PC ではなく、`Ubuntu + CUDA` 環境とする。

### 13.1 想定OS

* `Ubuntu 22.04 LTS` を第一候補
* `Ubuntu 24.04 LTS` は依存互換性を確認した上で追従

### 13.2 想定GPU環境

* NVIDIA GPU 搭載
* CUDA 利用可能
* PyTorch の CUDA build が利用可能

### 13.3 想定セットアップ方針

初期実装では、環境構築の再現性を担保するため、以下を成果物に含める。

* `scripts/setup.sh`
* `README.md`

`setup.sh` の役割:

* Ubuntu 環境で必要パッケージを導入する
* Python 仮想環境を構築する
* PyTorch, MMPose, MMAction2 系依存を導入する
* 必要に応じて checkpoint 取得手順を補助する

`README.md` の役割:

* 対応OSと前提条件を明記する
* セットアップ手順を記載する
* 推論実行方法を記載する
* 設定ファイルや checkpoint の置き方を説明する

---

## 14. 性能要件

### 14.1 初期目標

* 単一 GPU 環境でリアルタイムに近い推論
* `15 FPS 以上` を最低目標
* `20-30 FPS` を推奨目標
* 同時人物数 `1-5人`
* 動作分類遅延 `1-2秒以内`

### 14.2 検証条件

最低限以下を測定する。

* FPS
* end-to-end latency
* class-wise precision / recall / F1
* label switching の頻度
* track 継続率

---

## 15. 学習・評価方針

### 15.1 PoC 初期段階

PoC 初期段階では以下を段階的に進める。

1. RTMO -> BlockGCN の推論導線を構築する
2. 既存データまたは簡易自前データで動作確認する
3. 初期ラベル 5 クラスに対する推論安定性を確認する

### 15.2 データセット方針

公開 skeleton action recognition データセットを、PoC 初期の第一候補とする。

初期段階では以下を主目的とする。

* BlockGCN 系手法の再現確認
* RTMO -> BlockGCN 導線の検証
* 初期ラベルに近いカテゴリでの動作確認

ただし商用 PoC の本命は将来の自前データである。

理由:

* 実運用カメラ条件と乖離しやすい
* 初期ラベル定義と完全一致しない可能性が高い
* 2D RTMO 骨格と公開ベンチマークの分布差がある

### 15.3 将来の fine-tuning

将来的には以下を前提とする。

* 固定カメラ条件での自前動画収集
* person-level ラベル付与
* 必要に応じた clip-level 境界注釈
* クラス不均衡を考慮した学習設計

---

## 16. 実装フェーズ

### Phase 1: ベースライン導線

目的:

* RTMO 出力を BlockGCN 入力に変換する
* 単一人物クリップで推論を通す

成果物:

* 単一動画推論スクリプト
* JSON 出力
* 骨格系列の可視化

### Phase 2: 軽量 ID 割当とリアルタイム処理

目的:

* `RTMO only + lightweight ID assignment` を導入する
* track ごとの時系列バッファを動かす

成果物:

* webcam リアルタイムデモ
* track ごとの action 出力
* FPS 計測

### Phase 3: BlockGCN 設定最適化

目的:

* `clip_len`
* `stride`
* smoothing
* 欠損点処理

を調整し、PoC 安定性を高める。

### Phase 4: 自前データ対応

目的:

* ラベル定義の固定
* 自前データ収集
* fine-tuning

### Phase 5: 将来拡張

目的:

* 移動ベースイベントのルール実装
* multi-stream 化
* ONNX / TensorRT 化

---

## 17. 将来拡張の展望

### 17.1 移動ベースイベント

PoC 後には、人物位置の移動や zone 定義を用いたルールベース判定を追加する。

候補:

* `approaching_zone`
* `leaving_zone`
* `enter`
* `exit`
* `dwell`

この場合、現行の軽量 ID 割当を将来の tracking 実装へ自然に拡張できるように設計する。

### 17.2 モダリティ拡張

PoC の次段階では `Joint only` から以下へ拡張可能とする。

* `Joint + Joint Motion`
* `Joint + Bone + Joint Motion + Bone Motion`

指先などが分かれば細かい行動の認識につながるかも

### 17.3 ラベル拡張

対象ラベルは将来的に以下を候補とする。

* `bending`
* `lying`
* `hand_waving`
* `falling`

ただし `falling` は recall 重視の別評価設計が必要になる。

### 17.4 実行最適化

将来的に以下を検討する。

* ONNX Runtime
* TensorRT
* バッチ化または非同期化
* edge GPU への最適化

---

## 18. リスクと対策

### 18.1 骨格欠損

リスク:

* 遮蔽や画角外で関節が欠落する

対策:

* confidence を入力に含める
* 短期補間
* track 維持猶予

### 18.2 ID switch

リスク:

* すれ違いや遮蔽で track_id が入れ替わる

対策:

* bbox と keypoint の複合距離で対応付け
* missing frame 許容
* 初期 PoC は少人数環境を前提とする

### 18.3 ラベル揺れ

リスク:

* `standing` と `walking` が頻繁に切り替わる

対策:

* smoothing
* confirm count
* score threshold

### 18.4 ドメインギャップ

リスク:

* 公開ベースの設定が実運用に合わない

対策:

* 自前データで早期検証
* camera condition 固定で評価
* label definition の厳密化

---

## 19. 実装タスク分解

### 19.1 実装対象成果物

初期実装の成果物は以下とする。

* `README.md`
* `scripts/setup.sh`
* `config/` 配下の初期設定
* `pose/` の RTMO ラッパ
* `tracking/` の Hungarian ベース ID 割当
* `action/` の BlockGCN 入力整形と推論ラッパ
* `pipeline/` の end-to-end 推論導線
* `output/` への JSON / overlay 出力
* デバッグ用可視化
* 実行エントリポイント

### 19.2 タスク一覧

#### Task 0: リポジトリ基盤整備

目的:

* 実装しやすいディレクトリ骨格を作る
* Ubuntu 実行前提の構成を固定する

作業:

* 基本ディレクトリ作成
* `.gitignore` 整備
* 設定ファイル置き場の用意
* 実行エントリポイント名の決定

成果物:

* ディレクトリ骨格
* `.gitignore`

#### Task 1: 環境構築ドキュメントとセットアップ

目的:

* Ubuntu CUDA 環境で再現可能な導入手順を整える

作業:

* `README.md` 作成
* `scripts/setup.sh` 作成
* Python バージョン、venv 作成手順の明記
* PyTorch, MMPose, MMAction2, OpenCV 等の依存導入
* checkpoint 配置または取得手順の記載

成果物:

* `README.md`
* `scripts/setup.sh`

完了条件:

* Ubuntu 環境で README に従ってセットアップできる

#### Task 2: 設定管理の実装

目的:

* モデル設定と閾値をコードから分離する

作業:

* YAML または同等設定ファイルの導入
* pose / tracking / action / output の設定定義
* CLI から設定ファイルを読めるようにする

成果物:

* `config/*.yaml`
* 設定ローダ

#### Task 3: RTMO 推論ラッパ

目的:

* `rtmo-s_8xb32-600e_body7-640x640` を用いてフレーム単位の pose 推論を動かす

作業:

* MMPose 推論ラッパ実装
* checkpoint / config 読み込み
* bbox, keypoints, confidence の標準出力化
* GPU / device 指定対応

成果物:

* `pose/rtmo_estimator.py`

完了条件:

* 単一画像または単一フレームで pose 推論結果を取得できる

#### Task 4: COCO-17 整形と正規化

目的:

* RTMO 出力を BlockGCN 入力へ渡せる形に統一する

作業:

* `COCO-17` canonical order への mapping
* `(x, y, confidence)` 形式への統一
* `mid_hip` / `bbox center` ベース正規化
* 欠損点の前値保持ロジック
* `[C, T, V, M]` 形式への整形関数実装

成果物:

* `pose/keypoint_mapper.py`
* `action/preprocess.py`

完了条件:

* 任意長の pose 系列を BlockGCN 入力テンソルへ変換できる

#### Task 5: 軽量 tracking 実装

目的:

* frame 間で短期的な `track_id` を維持する

作業:

* Hungarian matching 実装
* cost 設計
* track state 管理
* missing frame 管理

成果物:

* `tracking/assigner.py`
* `tracking/track_manager.py`

完了条件:

* 少人数動画で `track_id` が短時間安定して維持される

#### Task 6: BlockGCN 推論ラッパ

目的:

* `Joint only` の BlockGCN 推論を実行可能にする

作業:

* BlockGCN モデル読込ラッパ
* `clip_len=32` のクリップ生成
* 推論結果のクラススコア出力
* 学習済み weight の読み込み点を整理

成果物:

* `action/blockgcn_infer.py`

完了条件:

* 単一クリップに対してクラススコアを返せる

#### Task 7: ラベル平滑化と状態管理

目的:

* 生推論結果を運用可能なラベルへ整える

作業:

* `smooth_confirm_count=3` 実装
* `score_thr=0.6` 初期実装
* `transition / uncertain` 状態管理

成果物:

* `action/smoother.py`

#### Task 8: end-to-end パイプライン

目的:

* 動画 / webcam から debug overlay / JSON 出力までを一貫動作させる

作業:

* video input 実装
* RTMO -> tracking -> buffer -> BlockGCN -> smoothing の接続
* 骨格 overlay 描画
* bbox, `track_id`, `action_label`, `action_score` の重畳表示
* FPS 表示
* JSON 出力

成果物:

* `pipeline/runtime.py`
* `util/json_writer.py`
* `util/visualizer.py`
* メイン実行スクリプト

完了条件:

* 動画入力で人物ごとの骨格 overlay と action_label を連続表示できる
* 必要に応じて同内容を JSON として保存できる

#### Task 9: ベンチマークと検証

目的:

* 初期 PoC の性能と安定性を確認する

作業:

* FPS 計測
* latency 計測
* `clip_len=24/32/48` 比較
* `score_thr` 比較

成果物:

* 計測ログ
* 比較メモ

#### Task 10: README 仕上げ

目的:

* 実装完了後の利用手順を最終化する

作業:

* セットアップ済み README の更新
* 実行例の追記
* 想定ディレクトリ構成の追記
* 既知の制約の追記

成果物:

* 完成版 `README.md`

### 19.3 実装順序の推奨

以下の順で実装する。

1. Task 0
2. Task 1
3. Task 2
4. Task 3
5. Task 4
6. Task 5
7. Task 6
8. Task 7
9. Task 8
10. Task 9
11. Task 10

### 19.4 初回実装スコープ

次の実装指示では、少なくとも以下を 1 セットとして実装対象に含めることを推奨する。

* Task 0
* Task 1
* Task 2
* Task 3
* Task 4
* Task 5
* Task 8 の最小版

これにより、`Ubuntu CUDA 環境でセットアップ可能` かつ `RTMO-s body7 の骨格 overlay と action 表示を確認できる` ところまでを早期に確認できる。

---

## 20. 保留事項

以下は実装着手時点では未確定であり、別途合意を取る。

1. `clip_len = 24 / 32 / 48` の比較結果を踏まえ、最終本命値をどれにするか
2. smoothing の `score threshold` の最終値をどう置くか
3. Ubuntu CUDA 環境向けの依存バージョン固定をどう置くか

---

## 21. 現時点の最終推奨

実装開始時の最終推奨構成は以下とする。

* `RTMO`
* `RTMO-s`
* `rtmo-s_8xb32-600e_body7-640x640`
* `軽量 ID 割当`
* `BlockGCN`
* `2D skeleton`
* `Joint only`
* `COCO-17`
* `Dynamic Topological Encoding` なし
* `clip_len = 32`
* `stride = 2`
* `Hungarian matching`
* 欠損点処理は `前値保持 + confidence維持`
* `confidence` は入力チャネルとして扱う
* `unknown` は学習クラスに含めない
* `transition / uncertain` は後処理状態として扱う
* `5 classes`

この構成は、PoC の成立性、リアルタイム性、将来拡張性のバランスが最も良い。
