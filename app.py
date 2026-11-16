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
# 2. 画像・動画の読み込みと手動範囲選択
# ------------------------------------------------------------

VIDEO_EXTENSIONS = (
    ".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"
)

# 候補を作る間隔。0.1秒 = 1秒間に約10コマ
FRAME_INTERVAL_SECONDS = 0.1


def detect_pose(rgb_image):
    """RGB画像から姿勢を検出する"""
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb_image)
    )
    return detector.detect(image)


def load_media(file_path):
    """アップロード直後の処理。画像はそのまま、動画は範囲選択画面を準備する。"""
    if file_path is None:
        raise gr.Error("画像または動画をアップロードしてください。")

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in VIDEO_EXTENSIONS:
        bgr = cv2.imread(file_path)
        if bgr is None:
            raise gr.Error("画像を読み込めませんでした。")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gallery = [(Image.fromarray(rgb), "画像")]
        return (
            gr.update(value=None, visible=False),
            file_path,
            gr.update(visible=False),
            gr.update(visible=False),
            gallery,
            [rgb],
            rgb,
            rgb,
            gr.update(visible=False),
            "画像を読み込みました。"
        )

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise gr.Error("動画を開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not fps or not np.isfinite(fps) or fps <= 0 or total_frames <= 0:
        raise gr.Error("動画の長さを取得できませんでした。")

    duration = total_frames / fps
    slider_max = max(0.1, round(duration, 2))

    return (
        gr.update(value=file_path, visible=True),
        file_path,
        gr.update(minimum=0, maximum=slider_max, value=0, step=0.1, visible=True),
        gr.update(minimum=0, maximum=slider_max, value=slider_max, step=0.1, visible=True),
        [],
        [],
        None,
        None,
        gr.update(visible=True),
        (
            f"動画の長さは **{duration:.2f}秒** です。"
            "動画を再生して、ジャンプの少し前から着地後までの"
            "開始・終了位置を指定してください。"
            "<br>指定範囲から **0.1秒ごと** にコマ候補を作ります。"
        )
    )


def make_candidates(file_path, start_sec, end_sec, progress=gr.Progress()):
    """指定範囲から0.1秒ごとに静止画候補を作る。候補数の上限は設けない。"""
    if not file_path:
        raise gr.Error("先に動画をアップロードしてください。")

    try:
        start_sec = float(start_sec)
        end_sec = float(end_sec)
    except (TypeError, ValueError):
        raise gr.Error("開始位置と終了位置を指定してください。")

    if end_sec <= start_sec:
        raise gr.Error("終了位置は開始位置より後にしてください。")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise gr.Error("動画を開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or not np.isfinite(fps) or fps <= 0 or total_frames <= 0:
        cap.release()
        raise gr.Error("動画情報を読み取れませんでした。")

    duration = total_frames / fps
    start_sec = max(0.0, min(start_sec, duration))
    end_sec = max(0.0, min(end_sec, duration))
    if end_sec <= start_sec:
        cap.release()
        raise gr.Error("指定した範囲が短すぎます。開始・終了位置を確認してください。")

    # 0.1秒ごとの時刻を作成。最後の時刻も範囲内なら追加する。
    span = end_sec - start_sec
    count = int(math.floor(span / FRAME_INTERVAL_SECONDS)) + 1
    times = [
        start_sec + i * FRAME_INTERVAL_SECONDS
        for i in range(count)
    ]
    if not times or end_sec - times[-1] > 0.000001:
        times.append(end_sec)

    candidates = []
    candidate_times = []
    total = len(times)

    print(
        f"候補フレーム作成開始: {start_sec:.2f}秒〜{end_sec:.2f}秒"
        f" / 予定 {total}コマ / 0.1秒間隔"
    )

    for i, sec in enumerate(times):
        frame_number = min(
            total_frames - 1,
            max(0, int(round(float(sec) * fps)))
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, bgr = cap.read()
        if ok and bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            candidates.append(rgb)
            candidate_times.append(float(sec))

        if i % 10 == 0 or i == total - 1:
            progress(
                (i + 1) / total,
                desc=f"コマを取り出しています… {i + 1}/{total}"
            )

    cap.release()

    if not candidates:
        raise gr.Error("指定した範囲からコマを取り出せませんでした。")

    gallery = [
        (Image.fromarray(frame), f"{sec:.2f}秒")
        for frame, sec in zip(candidates, candidate_times)
    ]

    print(
        f"候補フレーム作成完了: {len(candidates)}コマ"
        f" / {start_sec:.2f}秒〜{end_sec:.2f}秒"
    )

    return (
        gallery,
        candidates,
        candidates[0],
        candidates[0],
        (
            f"**{len(candidates)}コマ** の候補を作りました"
            f"（約0.1秒ごと）。比較したい瞬間の画像をクリックしてください。"
        )
    )


# ------------------------------------------------------------
# 3. ギャラリーのクリックで選択フレームを更新
# ------------------------------------------------------------

def select_frame(candidates, evt: gr.SelectData):
    if not candidates:
        raise gr.Error("選択できるフレームがありません。")

    index = evt.index

    # Gradioのバージョンによるindex形式の違いに対応
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

        **動画の場合：** 動画を確認し、ジャンプの少し前から着地後までの範囲を自分で指定します。その範囲を **0.1秒ごと** にコマに分け、候補から比較したい瞬間を選べます。
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
            ref_video_preview = gr.Video(label="お手本の動画を確認", visible=False)
            ref_media_path = gr.State(None)
            ref_start = gr.Slider(0, 60, value=0, step=0.1, label="開始位置（秒）", visible=False)
            ref_end = gr.Slider(0, 60, value=5, step=0.1, label="終了位置（秒）", visible=False)
            ref_make_button = gr.Button("指定した範囲からコマ候補を作る", visible=False)
            ref_status = gr.Markdown()
            ref_gallery = gr.Gallery(
                label="比較したいコマをクリック",
                columns=4,
                height=500,
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
            self_video_preview = gr.Video(label="自分の動画を確認", visible=False)
            self_media_path = gr.State(None)
            self_start = gr.Slider(0, 60, value=0, step=0.1, label="開始位置（秒）", visible=False)
            self_end = gr.Slider(0, 60, value=5, step=0.1, label="終了位置（秒）", visible=False)
            self_make_button = gr.Button("指定した範囲からコマ候補を作る", visible=False)
            self_status = gr.Markdown()
            self_gallery = gr.Gallery(
                label="比較したいコマをクリック",
                columns=4,
                height=500,
                object_fit="contain"
            )
            self_preview = gr.Image(label="選択中の自分のフォーム")
            self_candidates = gr.State([])
            self_selected = gr.State(None)

    ref_file.change(
        fn=load_media,
        inputs=ref_file,
        outputs=[
            ref_video_preview, ref_media_path, ref_start, ref_end,
            ref_gallery, ref_candidates, ref_selected, ref_preview,
            ref_make_button, ref_status
        ]
    )

    self_file.change(
        fn=load_media,
        inputs=self_file,
        outputs=[
            self_video_preview, self_media_path, self_start, self_end,
            self_gallery, self_candidates, self_selected, self_preview,
            self_make_button, self_status
        ]
    )

    ref_make_button.click(
        fn=make_candidates,
        inputs=[ref_media_path, ref_start, ref_end],
        outputs=[ref_gallery, ref_candidates, ref_selected, ref_preview, ref_status]
    )

    self_make_button.click(
        fn=make_candidates,
        inputs=[self_media_path, self_start, self_end],
        outputs=[self_gallery, self_candidates, self_selected, self_preview, self_status]
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
