
# ============================================================
# バレーボールフォーム比較アプリ
# Google Colab + Gradio + MediaPipe
# ============================================================

import os
import math
import subprocess

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
    import urllib.request
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

def detect_pose(rgb_image):
    """RGB画像から姿勢を検出する"""
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb_image)
    )
    return detector.detect(image)


def prepare_media(file_path):
    """
    アップロード画像・動画から比較候補を作る。
    戻り値：
    ギャラリー、候補フレーム一覧、初期選択フレーム、プレビュー
    """
    if file_path is None:
        raise gr.Error("画像または動画をアップロードしてください。")

    ext = os.path.splitext(file_path)[1].lower()

    # 画像の場合
    if ext not in VIDEO_EXTENSIONS:
        bgr = cv2.imread(file_path)

        if bgr is None:
            raise gr.Error("画像を読み込めませんでした。")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gallery = [(Image.fromarray(rgb), "画像")]
        return gallery, [rgb], rgb, rgb

    # 動画の場合
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise gr.Error("動画を開けませんでした。")

    frames = []
    hip_y_list = []

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

        # 腰の高さを調べる
        try:
            result = detect_pose(rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                hip_y = (lm[23].y + lm[24].y) / 2.0
                hip_y_list.append(hip_y)
            else:
                hip_y_list.append(None)

        except Exception:
            hip_y_list.append(None)

    cap.release()

    if not frames:
        raise gr.Error("動画からフレームを取得できませんでした。")

    valid_y = [y for y in hip_y_list if y is not None]

    if valid_y:
        min_y = min(valid_y)
        max_y = max(valid_y)
        threshold = min_y + (max_y - min_y) * 0.45

        candidate_indices = [
            i for i, y in enumerate(hip_y_list)
            if y is not None and y <= threshold
        ]
    else:
        candidate_indices = []

    # ジャンプ候補が見つからなければ動画全体から候補を選ぶ
    if not candidate_indices:
        candidate_indices = list(range(len(frames)))

    # 最大20コマを表示
    if len(candidate_indices) > 20:
        positions = np.linspace(
            0, len(candidate_indices) - 1, 20, dtype=int
        )
        candidate_indices = [
            candidate_indices[i] for i in positions
        ]

    candidates = [frames[i] for i in candidate_indices]

    gallery = [
        (Image.fromarray(frame), f"Frame {index}")
        for frame, index in zip(candidates, candidate_indices)
    ]

    # 最初の候補を初期選択にする。
    # ギャラリーをクリックすると選択した候補に更新される。
    return gallery, candidates, candidates[0], candidates[0]


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

    ref = np.asarray(reference_rgb.convert("RGB") if isinstance(
        reference_rgb, Image.Image
    ) else reference_rgb).copy()

    own = np.asarray(self_rgb.convert("RGB") if isinstance(
        self_rgb, Image.Image
    ) else self_rgb).copy()

    pts_r, ang_r = analyze_frame(ref)
    pts_s, ang_s = analyze_frame(own)

    if not pts_r or not pts_s:
        raise gr.Error(
            "人物の姿勢を検出できませんでした。全身が写る画像を使ってください。"
        )

    # 自分の画像の大きさに合わせてお手本を変形
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

    # 自分とお手本を重ねる
    overlay = cv2.addWeighted(own, 0.8, aligned_ref, 0.3, 0)

    # OpenCVの色指定はBGRなので、RGB用の色を明示して描画する
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

    # 角度比較表
    rows = [
        "| 関節 | お手本 | 自分 | 差 |",
        "|---|---:|---:|---:|"
    ]

    for joint in ["右肘", "左肘", "右膝", "左膝"]:
        a = ang_r.get(joint, 0)
        b = ang_s.get(joint, 0)
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
        ]
    )

    self_file.change(
        fn=prepare_media,
        inputs=self_file,
        outputs=[
            self_gallery, self_candidates,
            self_selected, self_preview
        ]
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
# 7. 共有URLを発行
# ------------------------------------------------------------

print("Webアプリを起動しています。URLが表示されるまで待ってね。")
port = int(os.environ.get("PORT", 7860))
app.launch(server_name="0.0.0.0", server_port=port, share=False)
