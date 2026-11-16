# ============================================================
# バレーボールフォーム比較アプリ
# Google Colab / Render + Gradio + MediaPipe
# ============================================================

import os
import math
import urllib.request

import cv2
import numpy as np
import gradio as gr
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image


# ------------------------------------------------------------
# 1. AIモデルの準備
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pose_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("AIモデルをダウンロードしています...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/1/"
        "pose_landmarker_heavy.task"
    )
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("AIモデルの準備ができました。")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options
)
detector = vision.PoseLandmarker.create_from_options(options)


# ------------------------------------------------------------
# 2. 画像・動画の読み込み
# ------------------------------------------------------------

VIDEO_EXTENSIONS = (
    ".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"
)

# 動画の仮解析を軽量化
# 以前は最大180コマだったが、最大30コマに減らす。
MAX_SCAN_SAMPLES = 30
MIN_SCAN_INTERVAL_SEC = 0.5

# 表示する候補画像も減らして、画面表示を軽くする。
FRAME_DISPLAY_INTERVAL_SEC = 0.25
MAX_DISPLAY_FRAMES = 16

# 腰が通常位置より上がったと判断する差（正規化座標）
JUMP_HEIGHT_THRESHOLD = 0.035


def detect_pose(rgb_image):
    """RGB画像から姿勢を検出する"""
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb_image)
    )
    return detector.detect(image)


def _resize_for_detection(rgb):
    """ジャンプ検出用に画像を縮小する"""
    h, w = rgb.shape[:2]
    longest = max(h, w)

    if longest <= 256:
        return rgb

    scale = 256.0 / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(
        rgb,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


def _find_jump_interval(sample_times, hip_y_values, duration):
    """
    腰の上下移動からジャンプ区間を推定する。
    戻り値はジャンプ開始・終了の秒数。検出できない場合はNone。
    """
    valid = [
        (i, y) for i, y in enumerate(hip_y_values)
        if y is not None and np.isfinite(y)
    ]

    if len(valid) < 4:
        return None

    values = np.array([y for _, y in valid], dtype=float)
    baseline = float(np.percentile(values, 70))

    airborne = [
        y is not None
        and np.isfinite(y)
        and y < baseline - JUMP_HEIGHT_THRESHOLD
        for y in hip_y_values
    ]

    groups = []
    start = None

    for i, is_airborne in enumerate(airborne):
        if is_airborne and start is None:
            start = i
        elif not is_airborne and start is not None:
            groups.append((start, i - 1))
            start = None

    if start is not None:
        groups.append((start, len(airborne) - 1))

    groups = [(a, b) for a, b in groups if b > a]

    if not groups:
        return None

    def group_strength(group):
        a, b = group
        group_values = [
            hip_y_values[i]
            for i in range(a, b + 1)
            if hip_y_values[i] is not None
        ]
        if not group_values:
            return 0.0
        return baseline - min(group_values)

    best_start, best_end = max(groups, key=group_strength)

    start_time = max(0.0, sample_times[best_start] - 0.5)
    end_time = min(duration, sample_times[best_end] + 0.5)

    if end_time <= start_time:
        return None

    return start_time, end_time


def _read_frame_at(cap, frame_number):
    """指定したフレーム番号の画像をRGBで取得する"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ret, bgr = cap.read()

    if not ret or bgr is None:
        return None

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prepare_media(file_path, progress=gr.Progress()):
    """
    画像・動画を読み込む。
    動画は最大30コマだけAIで確認してジャンプ区間を推定する。
    詳しい姿勢解析は、比較ボタンを押したときの選択フレームだけ行う。
    """
    if file_path is None:
        raise gr.Error("画像または動画をアップロードしてください。")

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in VIDEO_EXTENSIONS:
        progress(0.2, desc="画像を読み込んでいます")
        bgr = cv2.imread(file_path)

        if bgr is None:
            raise gr.Error("画像を読み込めませんでした。")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gallery = [(Image.fromarray(rgb), "画像")]
        progress(1.0, desc="完了")
        return gallery, [rgb], rgb, rgb

    print(f"動画の処理を開始します: {os.path.basename(file_path)}")
    progress(0.02, desc="動画を開いています")

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        cap.release()
        raise gr.Error("動画を開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise gr.Error("動画のフレームレートを取得できませんでした。")

    if total_frames <= 0:
        cap.release()
        raise gr.Error("動画の長さを取得できませんでした。")

    duration = total_frames / fps

    # 動画の長さに応じて最大30地点を選ぶ。
    sample_count = min(
        MAX_SCAN_SAMPLES,
        max(4, int(np.ceil(duration / MIN_SCAN_INTERVAL_SEC)))
    )
    sample_times = np.linspace(
        0,
        max(0.0, duration - 1.0 / fps),
        sample_count
    )

    hip_y_values = []

    print(
        f"動画情報: {duration:.1f}秒 / {fps:.1f}fps / "
        f"{total_frames}フレーム / AI確認 {sample_count}地点"
    )

    for i, sec in enumerate(sample_times):
        frame_number = min(
            total_frames - 1,
            int(round(float(sec) * fps))
        )
        rgb = _read_frame_at(cap, frame_number)

        if rgb is None:
            hip_y_values.append(None)
        else:
            try:
                small_rgb = _resize_for_detection(rgb)
                result = detect_pose(small_rgb)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    hip_y = (
                        landmarks[23].y + landmarks[24].y
                    ) / 2.0
                    hip_y_values.append(float(hip_y))
                else:
                    hip_y_values.append(None)

            except Exception as e:
                print(f"ジャンプ検出をスキップ: {e}")
                hip_y_values.append(None)

        progress(
            0.05 + 0.65 * ((i + 1) / sample_count),
            desc=f"ジャンプを探しています ({i + 1}/{sample_count})"
        )

    print("ジャンプ区間を判定しています...")
    jump_interval = _find_jump_interval(
        sample_times.tolist(),
        hip_y_values,
        duration
    )

    if jump_interval is None:
        cap.release()
        print("ジャンプを自動検出できませんでした。")
        raise gr.Error(
            "ジャンプを自動検出できませんでした。"
            "人物の全身、特に腰から足までが映る動画を使ってください。"
            "ジャンプの前後に立っている時間がある動画がおすすめです。"
        )

    jump_start, jump_end = jump_interval
    print(
        f"ジャンプ区間: {jump_start:.2f}秒〜{jump_end:.2f}秒"
    )

    progress(0.75, desc="ジャンプ中の画像を取り出しています")

    display_times = np.arange(
        jump_start,
        jump_end + 0.001,
        FRAME_DISPLAY_INTERVAL_SEC
    )

    if len(display_times) > MAX_DISPLAY_FRAMES:
        display_times = np.linspace(
            jump_start,
            jump_end,
            MAX_DISPLAY_FRAMES
        )

    candidates = []
    candidate_times = []

    for i, sec in enumerate(display_times):
        frame_number = min(
            total_frames - 1,
            int(round(float(sec) * fps))
        )
        rgb = _read_frame_at(cap, frame_number)

        if rgb is not None:
            candidates.append(rgb)
            candidate_times.append(float(sec))

        progress(
            0.75 + 0.20 * ((i + 1) / max(1, len(display_times))),
            desc=f"候補画像を準備しています ({i + 1}/{len(display_times)})"
        )

    cap.release()

    if not candidates:
        raise gr.Error("ジャンプ区間の画像を取り出せませんでした。")

    gallery = [
        (Image.fromarray(frame), f"{sec:.2f}秒")
        for frame, sec in zip(candidates, candidate_times)
    ]

    print(
        f"動画処理完了: {len(candidates)}コマを候補として表示します。"
    )
    progress(1.0, desc="完了")

    return gallery, candidates, candidates[0], candidates[0]


# ------------------------------------------------------------
# 3. ギャラリーのクリックで選択フレームを更新
# ------------------------------------------------------------

def select_frame(candidates, evt: gr.SelectData):
    if not candidates:
        raise gr.Error("選択できるフレームがありません。")

    index = evt.index

    if isinstance(index, (tuple, list)):
        index = index[0]

    index = int(index)

    if index < 0 or index >= len(candidates):
        raise gr.Error("フレームの選択位置が正しくありません。")

    selected = candidates[index]
    return selected, selected


# ------------------------------------------------------------
# 4. 関節の角度を計算
# ------------------------------------------------------------

def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    if angle < 0:
        angle += 360

    if angle > 180:
        angle = 360 - angle

    return angle


def analyze_frame(rgb):
    result = detect_pose(rgb)
    points = {}
    angles = {}

    if not result.pose_landmarks:
        return points, angles

    h, w = rgb.shape[:2]
    landmarks = result.pose_landmarks[0]

    for i, lm in enumerate(landmarks):
        points[i] = np.array([lm.x * w, lm.y * h])

    required = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    if not all(i in points for i in required):
        return points, angles

    angles["右肘"] = calculate_angle(
        points[12], points[14], points[16]
    )
    angles["左肘"] = calculate_angle(
        points[11], points[13], points[15]
    )
    angles["右膝"] = calculate_angle(
        points[24], points[26], points[28]
    )
    angles["左膝"] = calculate_angle(
        points[23], points[25], points[27]
    )

    return points, angles


# ------------------------------------------------------------
# 5. 重ね合わせと角度比較
# ------------------------------------------------------------

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14),
    (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]


def compare_forms(reference_rgb, self_rgb):
    if reference_rgb is None or self_rgb is None:
        raise gr.Error("お手本と自分のフレームを両方選んでください。")

    ref = np.asarray(
        reference_rgb.convert("RGB")
        if isinstance(reference_rgb, Image.Image)
        else reference_rgb
    ).copy()

    own = np.asarray(
        self_rgb.convert("RGB")
        if isinstance(self_rgb, Image.Image)
        else self_rgb
    ).copy()

    pts_r, ang_r = analyze_frame(ref)
    pts_s, ang_s = analyze_frame(own)

    if not pts_r or not pts_s:
        raise gr.Error(
            "人物の姿勢を検出できませんでした。全身が写る画像を使ってください。"
        )

    required = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    if not all(i in pts_r for i in required) or not all(
        i in pts_s for i in required
    ):
        raise gr.Error(
            "肩・肘・腰・膝・足首の一部を検出できませんでした。"
            "全身が写る画像を使ってください。"
        )

    h_s, w_s = own.shape[:2]

    hip_r = (pts_r[23] + pts_r[24]) / 2
    hip_s = (pts_s[23] + pts_s[24]) / 2

    length_r = np.linalg.norm(pts_r[12] - pts_r[24])
    length_s = np.linalg.norm(pts_s[12] - pts_s[24])

    if length_r < 1e-6:
        raise gr.Error("お手本の肩・腰の位置を認識できませんでした。")

    scale = length_s / length_r

    matrix = np.float32([
        [scale, 0, hip_s[0] - hip_r[0] * scale],
        [0, scale, hip_s[1] - hip_r[1] * scale]
    ])

    aligned_ref = cv2.warpAffine(ref, matrix, (w_s, h_s))
    overlay = cv2.addWeighted(own, 0.8, aligned_ref, 0.3, 0)

    # 自分：緑、お手本：赤
    for start, end in CONNECTIONS:
        p1 = tuple(np.round(pts_s[start]).astype(int))
        p2 = tuple(np.round(pts_s[end]).astype(int))
        cv2.line(overlay, p1, p2, (0, 255, 0), 3)

    for start, end in CONNECTIONS:
        p1 = (pts_r[start] - hip_r) * scale + hip_s
        p2 = (pts_r[end] - hip_r) * scale + hip_s
        p1 = tuple(np.round(p1).astype(int))
        p2 = tuple(np.round(p2).astype(int))
        cv2.line(overlay, p1, p2, (255, 0, 0), 3)

    rows = [
        "| 関節 | お手本 | 自分 | 差 |",
        "|---|---:|---:|---:|"
    ]

    for joint in ["右肘", "左肘", "右膝", "左膝"]:
        a = ang_r.get(joint)
        b = ang_s.get(joint)

        if a is None or b is None:
            rows.append(f"| {joint} | 算出不可 | 算出不可 | - |")
        else:
            diff = abs(a - b)
            rows.append(
                f"| {joint} | {a:.1f}° | {b:.1f}° | {diff:.1f}° |"
            )

    angle_table = "\n".join(rows)

    return (
        Image.fromarray(ref),
        Image.fromarray(own),
        Image.fromarray(overlay),
        angle_table
    )


# ------------------------------------------------------------
# 6. Web画面
# ------------------------------------------------------------

with gr.Blocks(title="バレーボールフォーム比較") as app:
    gr.Markdown(
        """
        # 🏐 バレーボールフォーム比較
        お手本と自分の画像・動画からフレームを選び、
        フォームの重ね合わせと関節角度を比較します。

        **動画の場合：** ジャンプ区間を自動で検出し、
        ジャンプ直前から着地後までのコマを表示します。
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## ① お手本")
            ref_file = gr.File(
                label="お手本の画像または動画",
                type="filepath",
                file_types=["image", "video"]
            )
            ref_gallery = gr.Gallery(
                label="比較したいコマをクリック",
                columns=4,
                height=300,
                object_fit="contain"
            )
            ref_preview = gr.Image(label="選択中のお手本")
            ref_candidates = gr.State([])
            ref_selected = gr.State(None)

        with gr.Column():
            gr.Markdown("## ② 自分のフォーム")
            self_file = gr.File(
                label="自分の画像または動画",
                type="filepath",
                file_types=["image", "video"]
            )
            self_gallery = gr.Gallery(
                label="比較したいコマをクリック",
                columns=4,
                height=300,
                object_fit="contain"
            )
            self_preview = gr.Image(label="選択中の自分のフォーム")
            self_candidates = gr.State([])
            self_selected = gr.State(None)

    ref_file.change(
        fn=prepare_media,
        inputs=ref_file,
        outputs=[
            ref_gallery, ref_candidates,
            ref_selected, ref_preview
        ],
        show_progress="full"
    )

    self_file.change(
        fn=prepare_media,
        inputs=self_file,
        outputs=[
            self_gallery, self_candidates,
            self_selected, self_preview
        ],
        show_progress="full"
    )

    ref_gallery.select(
        fn=select_frame,
        inputs=ref_candidates,
        outputs=[ref_selected, ref_preview]
    )

    self_gallery.select(
        fn=select_frame,
        inputs=self_candidates,
        outputs=[self_selected, self_preview]
    )

    compare_button = gr.Button(
        "🔍 フォームを比較する",
        variant="primary"
    )

    with gr.Row():
        result_ref = gr.Image(label="お手本")
        result_self = gr.Image(label="自分のフォーム")

    result_overlay = gr.Image(label="フォームの重ね合わせ")

    gr.Markdown("## 📐 関節角度の比較")
    angle_result = gr.Markdown()

    compare_button.click(
        fn=compare_forms,
        inputs=[ref_selected, self_selected],
        outputs=[
            result_ref, result_self,
            result_overlay, angle_result
        ]
    )


# ------------------------------------------------------------
# 7. Renderで起動
# ------------------------------------------------------------

print("Webアプリを起動しています。")
port = int(os.environ.get("PORT", 7860))
app.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False
)
