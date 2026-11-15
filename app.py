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
# 7. Renderで起動
# ------------------------------------------------------------

print("Webアプリを起動しています。")
port = int(os.environ.get("PORT", 7860))
app.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False
)
