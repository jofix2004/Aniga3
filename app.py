# app.py - Aniga3: Giao diện Gradio gọn gàng

import gradio as gr
import numpy as np
import json
import cv2
from PIL import Image, ImageDraw, ImageFont

from config import CONFIG, MASKABLE_CLASSES, DEFAULT_MASK_CLASSES, DEFAULT_CLASS_COLORS


def _draw_bboxes_on_image(image_pil, bbox_json_list):
    """Vẽ bounding box lên ảnh PIL với màu theo class."""
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)

    # Thử load font lớn hơn, fallback nếu không có
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for item in bbox_json_list:
        x1, y1, x2, y2 = item["bbox"]
        class_name = item["class"]
        conf = item["confidence"]
        ocr_text = item.get("ocr_text", "")

        # Lấy màu từ config
        hex_color = DEFAULT_CLASS_COLORS.get(class_name, "#ffffff")
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        color = (r, g, b)

        # Vẽ box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Vẽ label
        label = f"{class_name} {conf:.2f}"
        if ocr_text:
            label += f" | {ocr_text[:30]}"

        # Background cho text
        bbox_text = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 20), label, fill="white", font=font)

    return img


def _apply_mask_overlay(raw_image_pil, clean_image_pil, mask_np):
    """Áp dụng mask: vùng trắng = ảnh gốc, vùng đen = ảnh clean."""
    raw_np = np.array(raw_image_pil)
    clean_np = np.array(clean_image_pil)

    # Resize clean nếu khác kích thước
    if clean_np.shape[:2] != raw_np.shape[:2]:
        clean_np = cv2.resize(clean_np, (raw_np.shape[1], raw_np.shape[0]))

    # Resize mask nếu khác kích thước
    if mask_np.shape[:2] != raw_np.shape[:2]:
        mask_np = cv2.resize(mask_np, (raw_np.shape[1], raw_np.shape[0]))

    # Chuẩn hóa mask thành float [0, 1]
    mask_float = mask_np.astype(np.float32) / 255.0
    if len(mask_float.shape) == 2:
        mask_float = mask_float[:, :, np.newaxis]

    # Blend: vùng mask trắng = raw, vùng đen = clean
    overlay = (raw_np * mask_float + clean_np * (1 - mask_float)).astype(np.uint8)
    return Image.fromarray(overlay)


def run_pipeline_wrapper(raw_image, clean_image, device_mode, ocr_mode, selected_classes,
                          blur, min_area, cleanup, overlap_threshold, expand, feather):
    """Wrapper gọi pipeline chính và format output cho Gradio."""
    if raw_image is None:
        gr.Warning("Vui lòng tải lên ảnh gốc!")
        return None, None, None, None

    if clean_image is None:
        gr.Warning("Vui lòng tải lên ảnh clean (dùng cho tạo mask)!")
        return None, None, None, None

    if not selected_classes:
        gr.Warning("Vui lòng chọn ít nhất 1 class cho mask!")
        return None, None, None, None

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
    logs_text = "\n".join(result["logs"])
    full_output = f"// === LOG ===\n// {chr(10).join(result['logs'])}\n\n{bbox_json_str}"

    # Mask output
    mask_image = result["final_mask"]

    # Vẽ BBox lên ảnh gốc
    bbox_annotated = _draw_bboxes_on_image(raw_image, result["bbox_json"])

    # Áp dụng mask overlay
    mask_overlay = _apply_mask_overlay(raw_image, clean_image, mask_image)

    return full_output, bbox_annotated, mask_image, mask_overlay


def on_mask_selection_change(current_selection):
    """Xử lý logic Full Box (b1-b5)."""
    b_set = {"b1", "b2", "b3", "b4", "b5"}
    special_key = "Full Box (b1-b5)"
    current_set = set(current_selection)

    if special_key in current_set:
        for b in b_set:
            if b not in current_set:
                current_set.add(b)
    else:
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
        gr.Markdown("Phiên bản gọn gàng: Upload 2 ảnh → Nhận BBox + Mask + Overlay")

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
                        info="OCR data sẽ nằm trong JSON output.",
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

        # Hàng 1: JSON + BBox trực quan
        with gr.Row():
            with gr.Column():
                bbox_output = gr.Textbox(
                    label="📋 JSON BBox Output",
                    lines=20,
                    interactive=True,
                    show_copy_button=True,
                )
            with gr.Column():
                bbox_image_output = gr.Image(label="🔲 Ảnh vẽ BBox")

        # Hàng 2: Mask + Overlay
        with gr.Row():
            with gr.Column():
                mask_output = gr.Image(label="🎭 Mask cuối cùng")
            with gr.Column():
                overlay_output = gr.Image(label="✨ Áp dụng Mask (Raw ↔ Clean)")

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
            outputs=[bbox_output, bbox_image_output, mask_output, overlay_output],
        )

    return demo


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True, debug=True)
