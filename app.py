# app.py - Aniga3: Giao diện Gradio gọn gàng

import gradio as gr
import numpy as np
import json
from PIL import Image

from config import CONFIG, MASKABLE_CLASSES, DEFAULT_MASK_CLASSES


def run_pipeline_wrapper(raw_image, clean_image, device_mode, ocr_mode, selected_classes,
                          blur, min_area, cleanup, overlap_threshold, expand, feather):
    """Wrapper gọi pipeline chính và format output cho Gradio."""
    if raw_image is None:
        gr.Warning("Vui lòng tải lên ảnh gốc!")
        return None, None

    if clean_image is None:
        gr.Warning("Vui lòng tải lên ảnh clean (dùng cho tạo mask)!")
        return None, None

    if not selected_classes:
        gr.Warning("Vui lòng chọn ít nhất 1 class cho mask!")
        return None, None

    # Import core ở đây để tránh circular import và cho phép reload
    import importlib
    import core
    importlib.reload(core)

    # Lọc bỏ "Full Box (b1-b5)" khỏi danh sách class thực tế
    actual_classes = [c for c in selected_classes if c in MASKABLE_CLASSES]

    mask_params = {
        'blur': blur,
        'min_area': min_area,
        'cleanup': cleanup,
        'overlap_threshold': overlap_threshold,
        'expand': expand,
        'feather': feather,
    }

    # Chạy pipeline
    result = core.run_full_pipeline(
        raw_image_pil=raw_image,
        clean_image_pil=clean_image,
        device_mode=device_mode,
        ocr_mode=ocr_mode,
        mask_classes=actual_classes,
        mask_params=mask_params,
    )

    # Format JSON output
    bbox_json_str = json.dumps(result["bbox_json"], indent=2, ensure_ascii=False)

    # Thêm logs vào JSON
    logs_text = "\n".join(result["logs"])
    full_output = f"// === LOG ===\n// {chr(10).join(result['logs'])}\n\n{bbox_json_str}"

    # Mask output (grayscale numpy → hiển thị được)
    mask_image = result["final_mask"]

    return full_output, mask_image


def on_mask_selection_change(current_selection):
    """Xử lý logic Full Box (b1-b5)."""
    b_set = {"b1", "b2", "b3", "b4", "b5"}
    special_key = "Full Box (b1-b5)"
    current_set = set(current_selection)

    if special_key in current_set:
        # Thêm tất cả b nếu chưa có
        for b in b_set:
            if b not in current_set:
                current_set.add(b)
    else:
        # Kiểm tra nếu tất cả b đã được chọn thủ công
        if b_set.issubset(current_set):
            current_set.add(special_key)

    return gr.update(value=list(current_set))


# ============================================================================
# GIAO DIỆN GRADIO
# ============================================================================

def create_ui():
    mask_defaults = CONFIG['mask_defaults']

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css="footer {display: none !important;}",
        title="Aniga3 - Phát hiện & Tạo Mask"
    ) as demo:

        gr.Markdown("# 🎯 Aniga3 - Phát hiện & Tạo Mask")
        gr.Markdown("Phiên bản gọn gàng: Upload 2 ảnh → Nhận JSON BBox + Mask cuối cùng")

        # --- INPUT ---
        with gr.Row():
            with gr.Column():
                raw_image = gr.Image(type="pil", label="📷 Ảnh gốc (Raw)")
            with gr.Column():
                clean_image = gr.Image(type="pil", label="🧹 Ảnh clean (Đã xóa Text/SFX)")

        # --- CẤU HÌNH ---
        with gr.Accordion("⚙️ Cấu hình", open=True):
            with gr.Row():
                # Cột 1: Cấu hình chính
                with gr.Column(scale=1):
                    device_mode = gr.Radio(
                        ["Auto", "CPU", "GPU"],
                        value="Auto",
                        label="Chế độ phần cứng",
                        info="Auto: ưu tiên GPU nếu có",
                    )
                    ocr_mode = gr.Radio(
                        ["Không bật", "Tiếng Anh (Tinh chỉnh box)", "Tiếng Nhật (Chỉ trích xuất)"],
                        value="Không bật",
                        label="Chế độ OCR",
                        info="Yêu cầu bật Ensemble. OCR data sẽ nằm trong JSON output.",
                    )

                # Cột 2: Class selector
                with gr.Column(scale=1):
                    mask_classes_selector = gr.CheckboxGroup(
                        choices=["text", "text2", "b1", "b2", "b3", "b4", "b5", "Full Box (b1-b5)"],
                        value=DEFAULT_MASK_CLASSES,
                        label="Chọn class để tạo Mask YOLO",
                        info="Chọn class có bbox sẽ được dùng để tạo mask.",
                    )

                # Cột 3: Mask params
                with gr.Column(scale=1):
                    gr.Markdown("**Thông số Mask**")
                    mask_blur = gr.Slider(1, 21, mask_defaults.get('blur', 5), step=2, label="Blur")
                    mask_min_area = gr.Slider(10, 5000, mask_defaults.get('min_area', 100), step=10, label="Min Area")
                    mask_cleanup = gr.Slider(1, 31, mask_defaults.get('cleanup', 7), step=2, label="Cleanup")
                    mask_overlap = gr.Slider(0.0, 1.0, mask_defaults.get('overlap_threshold', 0.1), step=0.05, label="Overlap Thr")
                    mask_expand = gr.Slider(1, 51, mask_defaults.get('expand', 31), step=2, label="Expand")
                    mask_feather = gr.Slider(1, 151, mask_defaults.get('feather', 21), step=2, label="Feather")

        # --- NÚT CHẠY ---
        run_button = gr.Button("▶ Chạy Pipeline", variant="primary", size="lg")

        # --- OUTPUT ---
        gr.Markdown("---\n### 📊 Kết quả")
        with gr.Row():
            with gr.Column():
                bbox_output = gr.Textbox(
                    label="📋 JSON BBox Output",
                    lines=20,
                    interactive=True,
                    show_copy_button=True,
                )
            with gr.Column():
                mask_output = gr.Image(label="🎭 Mask cuối cùng")

        # --- EVENT HANDLERS ---
        mask_classes_selector.change(
            fn=on_mask_selection_change,
            inputs=[mask_classes_selector],
            outputs=[mask_classes_selector],
        )

        run_button.click(
            fn=run_pipeline_wrapper,
            inputs=[
                raw_image, clean_image,
                device_mode, ocr_mode, mask_classes_selector,
                mask_blur, mask_min_area, mask_cleanup, mask_overlap,
                mask_expand, mask_feather,
            ],
            outputs=[bbox_output, mask_output],
        )

    return demo


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True, debug=True)
