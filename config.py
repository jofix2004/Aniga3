# config.py - Aniga3: Cấu hình mặc định (hardcode)

import torch
import os

# --- ĐƯỜNG DẪN MODEL CỐ ĐỊNH (Colab) ---
MODEL_PATHS = [
    "/content/Aniga3/model1.pt",  # YOLOv8 (Ultralytics)
    "/content/Aniga3/model2.pt",  # YOLOv9 gốc (DetectMultiBackend)
    "/content/Aniga3/model3.pt",  # YOLOv11 (Ultralytics)
]

# Model 2 là YOLOv9 gốc (dùng DetectMultiBackend thay vì Ultralytics YOLO)
IS_MODEL2_YOLOV9_ORIGINAL = True

# --- THƯ MỤC CACHE ---
MODEL_CACHE_DIR = "_model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# --- CẤU HÌNH MẶC ĐỊNH ---
CONFIG = {
    # Thông số xử lý cho từng model đơn
    "single_model_defaults": {
        "conf_threshold": 0.05,
        "iou_same_class_threshold": 0.05,
        "iou_diff_class_threshold": 0.95,
    },

    # Thông số pipeline Ensemble
    "ensemble_defaults": {
        "wbf_iou_thr": 0.55,
        "wbf_skip_box_thr": 0.01,
        "final_conf_threshold": 0.4,
        "parent_child_containment_thr": 0.8,
        "ensemble_inter_class_iou_thr": 0.95,
        "ensemble_intra_group_iou_thr": 0.6,
    },

    # Thông số tạo Mask
    "mask_defaults": {
        "blur": 5,
        "min_area": 100,
        "cleanup": 7,
        "overlap_threshold": 0.1,
        "expand": 31,
        "feather": 21,
    },

    # Nhóm class cho Ensemble Pipeline
    "class_groups": {
        "group_a": ["b1", "b2", "b3", "b4", "b5"],
        "group_b": ["text"],
        "group_c_peer": ["text2", "text"],
    },

    # Nhóm áp dụng lọc nội bộ
    "intra_group_filtering_enabled_groups": ["group_a", "group_c_peer"],

    # Nhóm tương đồng ngữ nghĩa
    "semantically_similar_groups": [["text2", "text3"]],

    # Logic phân cấp nội bộ class
    "intra_class_hierarchical_logic": {
        "enabled_classes": ["text", "text2", "text3"],
        "parent_min_score_for_discard": 0.2,
        "child_min_score_for_override": 0.6,
    },
}

# --- MÀU SẮC MẶC ĐỊNH CHO BBOX ---
DEFAULT_CLASS_COLORS = {
    "b1": "#d62728",
    "b2": "#1f77b4",
    "b3": "#2ca02c",
    "b4": "#ff7f0e",
    "b5": "#9467bd",
    "text": "#8c564b",
    "text2": "#e377c2",
    "text3": "#17becf",
}

# --- DANH SÁCH CLASS CÓ THỂ TẠO MASK ---
MASKABLE_CLASSES = ["text", "text2", "b1", "b2", "b3", "b4", "b5"]

# --- MẶC ĐỊNH CLASS ĐƯỢC BẬT CHO MASK ---
DEFAULT_MASK_CLASSES = ["text", "b1", "b2", "b3", "b4", "b5"]


def resolve_device(device_mode: str) -> str:
    """Xác định device thực tế từ chế độ người dùng chọn."""
    if device_mode == "Auto":
        return "GPU" if torch.cuda.is_available() else "CPU"
    return device_mode
