# core.py - Aniga3: Logic xử lý chính (Ensemble + OCR + Mask)

import torch
import cv2
import numpy as np
import json
import time
import threading
import os
import sys
import shutil
import filecmp
from pathlib import Path
from PIL import Image
from matplotlib.colors import to_rgb
from ultralytics import YOLO
import torchvision.ops as ops
from skimage.metrics import structural_similarity as ssim

try:
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False

try:
    from manga_ocr import MangaOcr
    MANGA_OCR_AVAILABLE = True
except ImportError:
    MANGA_OCR_AVAILABLE = False

from numba import jit
from numba.typed import List

# --- PHẦN KHỞI TẠO YOLOv9 GỐC ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
    from utils.torch_utils import select_device
    YOLOV9_AVAILABLE = True
except ImportError:
    YOLOV9_AVAILABLE = False

from config import CONFIG, MODEL_PATHS, IS_MODEL2_YOLOV9_ORIGINAL, MODEL_CACHE_DIR, MASKABLE_CLASSES

# --- GLOBAL CACHES ---
OCR_MODELS = {}
MANGA_OCR_MODELS = {}
LOADED_MODELS_CACHE = [None, None, None]


# ============================================================================
# PHẦN 1: CÁC HÀM TIỆN ÍCH (NUMBA JIT)
# ============================================================================

@jit(nopython=True, cache=True)
def _get_box_area_numba(box):
    return (box[2] - box[0]) * (box[3] - box[1])

@jit(nopython=True, cache=True)
def calculate_iou_numba(box1, box2):
    x1_inter = max(box1[0], box2[0]); y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2]); y2_inter = min(box1[3], box2[3])
    inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)
    if inter_area == 0.0: return 0.0
    box1_area = _get_box_area_numba(box1); box2_area = _get_box_area_numba(box2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0

@jit(nopython=True, cache=True)
def _check_containment_numba(box1, box2, thr):
    area1, area2 = _get_box_area_numba(box1), _get_box_area_numba(box2)
    if area1 == 0.0 or area2 == 0.0: return False
    bigger, smaller = (box1, box2) if area1 > area2 else (box2, box1)
    inter_x1, inter_y1 = max(bigger[0], smaller[0]), max(bigger[1], smaller[1])
    inter_x2, inter_y2 = min(bigger[2], smaller[2]), min(bigger[3], smaller[3])
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    containment_ratio = inter_area / _get_box_area_numba(smaller)
    return containment_ratio >= thr

@jit(nopython=True, cache=True)
def _is_box_a_inside_b_numba(box_a, box_b, containment_threshold=0.8):
    x1_inter = max(box_a[0], box_b[0]); y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2]); y2_inter = min(box_a[3], box_b[3])
    inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)
    if inter_area == 0.0: return False
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    if area_a == 0.0: return False
    return (inter_area / area_a) >= containment_threshold


# ============================================================================
# PHẦN 2: ENSEMBLE DETECTION PIPELINE
# ============================================================================

class AdvancedDetectionPipeline:
    def __init__(self, class_mapping, config, image_size):
        self.class_mapping = class_mapping
        self.reverse_class_mapping = {v: k for k, v in class_mapping.items()}
        self.config = config
        self.image_width, self.image_height = image_size
        self._map_names_to_ids()

    def _map_names_to_ids(self):
        groups = self.config.get('class_groups', {})
        hier_config = self.config.get('intra_class_hierarchical_logic', {})
        similar_groups_config = self.config.get('semantically_similar_groups', [])

        self.group_b_ids = {self.reverse_class_mapping.get(n) for n in groups.get('group_b', [])} - {None}
        self.hierarchical_class_ids = {self.reverse_class_mapping.get(n) for n in hier_config.get('enabled_classes', [])} - {None}

        self.similarity_map_numba = List()
        temp_map = {}
        for group in similar_groups_config:
            group_ids = {self.reverse_class_mapping.get(name) for name in group if name in self.reverse_class_mapping}
            for cls_id in group_ids:
                temp_map[cls_id] = group_ids
        for k, v_set in temp_map.items():
            if k is not None:
                v_list = List([float(item) for item in v_set if item is not None])
                self.similarity_map_numba.append((float(k), v_list))

        self.intra_group_filterable_ids = {
            self.reverse_class_mapping.get(cn)
            for gn in self.config.get('intra_group_filtering_enabled_groups', [])
            if gn in groups
            for cn in groups[gn]
            if cn in self.reverse_class_mapping
        }
        self.class_id_to_group_name = {
            self.reverse_class_mapping.get(cn): gn
            for gn in self.config.get('intra_group_filtering_enabled_groups', [])
            if gn in groups
            for cn in groups[gn]
            if cn in self.reverse_class_mapping
        }

        self.text_id = self.reverse_class_mapping.get('text', -1)
        self.text2_id = self.reverse_class_mapping.get('text2', -1)
        self.b1_id = self.reverse_class_mapping.get('b1', -1)
        self.b3_id = self.reverse_class_mapping.get('b3', -1)
        self.group_a_ids = {self.reverse_class_mapping.get(n) for n in groups.get('group_a', [])} - {None}

    def _custom_fusion(self, raw_results, iou_matrix_gpu):
        params = self.config['current_settings']
        iou_thr = params['wbf_iou_thr']
        containment_thr = params.get('parent_child_containment_thr', 0.8)

        all_boxes_list = List()
        model_index = 0
        for model_result in raw_results:
            for box in model_result:
                if box[4] >= params.get('wbf_skip_box_thr', 0.0):
                    all_boxes_list.append(np.array([box[0], box[1], box[2], box[3], box[4], float(box[5]), float(model_index)], dtype=np.float64))
            model_index += 1

        num_models_active = len(raw_results)
        if not all_boxes_list:
            return []

        iou_matrix_cpu = iou_matrix_gpu.cpu().numpy()
        clusters = _cluster_boxes_smart_numba(all_boxes_list, iou_thr, containment_thr, self.similarity_map_numba, iou_matrix_cpu)

        fused_boxes = List()
        for c_indices in clusters:
            if not c_indices:
                continue
            cluster = List([all_boxes_list[i] for i in c_indices])
            fused_box = _process_cluster_adaptive_numba(cluster, num_models_active, b1_id=self.b1_id, b3_id=self.b3_id)
            if fused_box is not None:
                fused_boxes.append(fused_box)

        return [[b[0], b[1], b[2], b[3], b[4], int(b[5])] for b in fused_boxes]

    def _apply_containment_bonus_penalty(self, predictions):
        """Áp dụng luật thưởng/phạt dựa trên vị trí tương đối."""
        if not predictions or not self.group_a_ids or (self.text_id == -1 and self.text2_id == -1):
            return predictions

        group_a_boxes, text_boxes, text2_boxes, other_boxes = [], [], [], []
        for p in predictions:
            class_id = p[5]
            if class_id in self.group_a_ids:
                group_a_boxes.append(p)
            elif class_id == self.text_id:
                text_boxes.append(p)
            elif class_id == self.text2_id:
                text2_boxes.append(p)
            else:
                other_boxes.append(p)

        modified_predictions = []

        for t2_box in text2_boxes:
            for t_box in text_boxes:
                if calculate_iou_numba(np.array(t2_box[:4], dtype=np.float64), np.array(t_box[:4], dtype=np.float64)) > 0.3:
                    t2_box[4] *= 0.7
                    break
            modified_predictions.append(t2_box)

        for t_box in text_boxes:
            if group_a_boxes:
                for a_box in group_a_boxes:
                    if _is_box_a_inside_b_numba(np.array(t_box[:4]), np.array(a_box[:4])):
                        t_box[4] *= 1.3
                        if t_box[4] > 1.0:
                            t_box[4] = 1.0
                        break
            modified_predictions.append(t_box)

        return modified_predictions + group_a_boxes + other_boxes

    def _filter_inter_class_overlap(self, predictions, iou_matrix_gpu):
        inter_class_iou_thr = self.config['current_settings']['ensemble_inter_class_iou_thr']
        preds_sorted = sorted(predictions, key=lambda x: x[4], reverse=True)
        if not preds_sorted:
            return []

        boxes_tensor = torch.tensor([p[:4] for p in preds_sorted], dtype=torch.float32, device=iou_matrix_gpu.device)
        class_ids = torch.tensor([p[5] for p in preds_sorted], device=iou_matrix_gpu.device)
        iou_matrix = ops.box_iou(boxes_tensor, boxes_tensor)

        num_boxes = len(preds_sorted)
        keep_mask = torch.ones(num_boxes, dtype=torch.bool, device=iou_matrix_gpu.device)
        group_b_ids_tensor = torch.tensor(list(self.group_b_ids), dtype=class_ids.dtype, device=class_ids.device)

        for i in range(num_boxes):
            if not keep_mask[i]:
                continue
            if torch.any(class_ids[i] == group_b_ids_tensor):
                continue
            is_not_group_b_j = ~torch.any(class_ids[i+1:, None] == group_b_ids_tensor, dim=1)
            is_overlapped_j = iou_matrix[i, i+1:] > inter_class_iou_thr
            suppress_mask = is_overlapped_j & is_not_group_b_j
            keep_mask[i+1:][suppress_mask] = False

        kept_indices = torch.where(keep_mask)[0].cpu().numpy()
        return [preds_sorted[i] for i in kept_indices]

    def _filter_intra_group_overlap(self, predictions):
        iou_thr = self.config['current_settings']['ensemble_intra_group_iou_thr']
        preds_sorted = sorted(predictions, key=lambda x: x[4], reverse=True)
        to_keep = [True] * len(preds_sorted)

        for i in range(len(preds_sorted)):
            if not to_keep[i]:
                continue
            box_i = preds_sorted[i]
            group_i = self.class_id_to_group_name.get(box_i[5])
            if group_i is None:
                continue
            for j in range(i + 1, len(preds_sorted)):
                if not to_keep[j]:
                    continue
                box_j = preds_sorted[j]
                group_j = self.class_id_to_group_name.get(box_j[5])
                if group_j != group_i:
                    continue
                if calculate_iou_numba(np.array(box_i[:4], dtype=np.float64), np.array(box_j[:4], dtype=np.float64)) > iou_thr:
                    to_keep[j] = False

        return [preds_sorted[i] for i in range(len(preds_sorted)) if to_keep[i]]

    def process(self, raw_results, device_mode):
        logs = []
        t_start = time.perf_counter()
        device = 'cuda' if device_mode == 'GPU' and torch.cuda.is_available() else 'cpu'

        t_step_start = time.perf_counter()
        all_boxes_for_iou = [box for res in raw_results for box in res]
        iou_matrix_gpu = torch.empty(0, 0, device=device)
        if len(all_boxes_for_iou) > 1:
            boxes_tensor = torch.tensor([b[:4] for b in all_boxes_for_iou], dtype=torch.float32).to(device)
            iou_matrix_gpu = ops.box_iou(boxes_tensor, boxes_tensor)
        logs.append(f"- Tính toán ma trận IoU ({len(all_boxes_for_iou)} boxes) trên {device.upper()}: {time.perf_counter() - t_step_start:.4f}s")

        t_step_start = time.perf_counter()
        fused_preds = self._custom_fusion(raw_results, iou_matrix_gpu)
        logs.append(f"- Fusion & Gom cụm (Numba + Pre-computed IoU): {time.perf_counter() - t_step_start:.4f}s")

        t_step_start = time.perf_counter()
        bonus_penalty_preds = self._apply_containment_bonus_penalty(fused_preds)
        logs.append(f"- Áp dụng Luật Thưởng/Phạt Vị trí: {time.perf_counter() - t_step_start:.4f}s")

        t_step_start = time.perf_counter()
        inter_class_filtered_preds = []
        if bonus_penalty_preds:
            fused_boxes_tensor = torch.tensor([p[:4] for p in bonus_penalty_preds], dtype=torch.float32).to(device)
            fused_iou_matrix_gpu = ops.box_iou(fused_boxes_tensor, fused_boxes_tensor)
            inter_class_filtered_preds = self._filter_inter_class_overlap(bonus_penalty_preds, fused_iou_matrix_gpu)
        logs.append(f"- Lọc GIỮA các Nhóm ({device.upper()}): {time.perf_counter() - t_step_start:.4f}s")

        t_step_start = time.perf_counter()
        intra_group_filtered_preds = self._filter_intra_group_overlap(inter_class_filtered_preds)
        logs.append(f"- Lọc TRONG Nội Nhóm: {time.perf_counter() - t_step_start:.4f}s")

        t_step_start = time.perf_counter()
        final_conf_thr = self.config['current_settings'].get('final_conf_threshold', 0.0)
        final_preds = [p for p in intra_group_filtered_preds if p[4] >= final_conf_thr]
        logs.append(f"- Lọc Tin cậy Cuối cùng: {time.perf_counter() - t_step_start:.4f}s")

        logs.insert(0, f"Tổng thời gian Pipeline: {time.perf_counter() - t_start:.4f}s")
        return final_preds, logs


# ============================================================================
# PHẦN 3: CÁC HÀM NUMBA JIT CHO ENSEMBLE
# ============================================================================

@jit(nopython=True, cache=True)
def _are_boxes_related_numba(box1_idx, box2_idx, boxes, iou_thr, containment_thr, similarity_map, iou_matrix):
    box1 = boxes[box1_idx]; box2 = boxes[box2_idx]
    cls1, cls2 = int(box1[5]), int(box2[5])
    is_class_related = (cls1 == cls2)
    if not is_class_related:
        for k, v_list in similarity_map:
            if k == cls1:
                for v in v_list:
                    if v == cls2:
                        is_class_related = True; break
                break
    if not is_class_related:
        return False
    if iou_matrix[box1_idx, box2_idx] >= iou_thr:
        return True
    if _check_containment_numba(box1[:4], box2[:4], containment_thr):
        return True
    return False

@jit(nopython=True, cache=True)
def _cluster_boxes_smart_numba(boxes, iou_thr, containment_thr, similarity_map, iou_matrix):
    n_boxes = len(boxes)
    clusters = List()
    visited = np.zeros(n_boxes, dtype=np.bool_)
    for i in range(n_boxes):
        if visited[i]:
            continue
        current_cluster_indices = List()
        q = List()
        q.append(i)
        visited[i] = True
        while len(q) > 0:
            j = q.pop(0)
            current_cluster_indices.append(j)
            for k in range(n_boxes):
                if not visited[k] and _are_boxes_related_numba(j, k, boxes, iou_thr, containment_thr, similarity_map, iou_matrix):
                    visited[k] = True
                    q.append(k)
        clusters.append(current_cluster_indices)
    return clusters

@jit(nopython=True, cache=True)
def _process_cluster_adaptive_numba(cluster, num_models_active, b1_id=-1, b3_id=-1):
    if len(cluster) == 0:
        return None

    # Vote Override cho B1 & B3
    if len(cluster) == 3 and b1_id != -1 and b3_id != -1:
        models_in_cluster = List()
        for b in cluster:
            m_idx = int(b[6])
            is_new_model = True
            for existing_m in models_in_cluster:
                if m_idx == existing_m:
                    is_new_model = False; break
            if is_new_model:
                models_in_cluster.append(m_idx)

        if len(models_in_cluster) == 3:
            b1_count = 0
            b3_count = 0
            for b in cluster:
                cls_id = int(b[5])
                if cls_id == b1_id: b1_count += 1
                elif cls_id == b3_id: b3_count += 1

            target_id = -1
            if b1_count == 2 and b3_count == 1: target_id = b1_id
            elif b3_count == 2 and b1_count == 1: target_id = b3_id

            if target_id != -1:
                for b in cluster:
                    cls_id = int(b[5])
                    if cls_id == b1_id or cls_id == b3_id:
                        b[5] = float(target_id)

    # Bầu chọn class
    class_votes_keys = List()
    class_votes_values = List()
    for box in cluster:
        cls_id = int(box[5]); score = box[4]; found = False
        for i, k in enumerate(class_votes_keys):
            if k == cls_id:
                class_votes_values[i] += score; found = True; break
        if not found:
            class_votes_keys.append(cls_id); class_votes_values.append(score)

    if len(class_votes_keys) == 0:
        return None
    max_vote = -1.0; winning_class = -1
    for i in range(len(class_votes_values)):
        if class_votes_values[i] > max_vote:
            max_vote = class_votes_values[i]; winning_class = class_votes_keys[i]

    winning_boxes = List()
    for b in cluster:
        if int(b[5]) == winning_class:
            winning_boxes.append(b)

    if len(winning_boxes) == 0:
        return None

    # Consensus BBox
    if len(winning_boxes) >= 2:
        consensus_x1, consensus_y1 = 1e9, 1e9
        consensus_x2, consensus_y2 = -1e9, -1e9
        has_consensus = False

        for i in range(len(winning_boxes)):
            for j in range(i + 1, len(winning_boxes)):
                b1, b2 = winning_boxes[i], winning_boxes[j]
                inter_x1 = max(b1[0], b2[0]); inter_y1 = max(b1[1], b2[1])
                inter_x2 = min(b1[2], b2[2]); inter_y2 = min(b1[3], b2[3])

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    consensus_x1 = min(consensus_x1, inter_x1)
                    consensus_y1 = min(consensus_y1, inter_y1)
                    consensus_x2 = max(consensus_x2, inter_x2)
                    consensus_y2 = max(consensus_y2, inter_y2)
                    has_consensus = True

        if has_consensus:
            fused_box_coords = [consensus_x1, consensus_y1, consensus_x2, consensus_y2]
        else:
            min_x1, min_y1 = winning_boxes[0][0], winning_boxes[0][1]
            max_x2, max_y2 = winning_boxes[0][2], winning_boxes[0][3]
            for i in range(1, len(winning_boxes)):
                box = winning_boxes[i]
                min_x1 = min(min_x1, box[0]); min_y1 = min(min_y1, box[1])
                max_x2 = max(max_x2, box[2]); max_y2 = max(max_y2, box[3])
            fused_box_coords = [min_x1, min_y1, max_x2, max_y2]
    else:
        fused_box_coords = [winning_boxes[0][0], winning_boxes[0][1], winning_boxes[0][2], winning_boxes[0][3]]

    unique_models = List()
    for b in winning_boxes:
        model_idx = int(b[6]); is_in = False
        for um in unique_models:
            if um == model_idx: is_in = True; break
        if not is_in: unique_models.append(model_idx)

    num_contributing_models = len(unique_models)
    total_score_for_conf = 0.0
    for b in winning_boxes:
        total_score_for_conf += b[4]

    avg_score = total_score_for_conf / len(winning_boxes)
    final_score = 0.0
    if num_contributing_models == 1: final_score = avg_score * 0.4
    elif num_contributing_models == 2: final_score = avg_score * 0.8
    else: final_score = avg_score * 1.0

    return np.array([fused_box_coords[0], fused_box_coords[1], fused_box_coords[2], fused_box_coords[3], final_score, float(winning_class)])


# ============================================================================
# PHẦN 4: INFERENCE MODEL ĐƠN
# ============================================================================

def _filter_boxes_gpu(predictions_tensor, iou_diff_class_threshold):
    if predictions_tensor is None or predictions_tensor.shape[0] == 0:
        return []
    sorted_indices = predictions_tensor[:, 4].argsort(descending=True)
    preds_sorted = predictions_tensor[sorted_indices]
    boxes, scores, classes = preds_sorted[:, :4], preds_sorted[:, 4], preds_sorted[:, 5]
    keep_indices = torch.ones(len(preds_sorted), dtype=torch.bool, device=preds_sorted.device)
    for i in range(len(preds_sorted)):
        if not keep_indices[i]:
            continue
        iou = ops.box_iou(boxes[i:i+1], boxes[i+1:])
        different_class_mask = classes[i] != classes[i+1:]
        overlap_mask = iou[0] > iou_diff_class_threshold
        suppress_mask = different_class_mask & overlap_mask
        keep_indices[i+1:][suppress_mask] = False
    return preds_sorted[keep_indices].cpu().tolist()

def _inference_single_model(model_object, is_yolov9_original, image_pil, image_bgr, params):
    """Chạy inference cho 1 model."""
    conf_threshold, iou_same_class_threshold, iou_diff_class_threshold, device_mode = params
    t_start = time.perf_counter()
    boxes_after_nms_tensor = None
    names = model_object.names

    if is_yolov9_original:
        stride, pt, imgsz = model_object.stride, model_object.pt, (640, 640)
        im = letterbox(image_bgr, imgsz, stride=stride, auto=pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(model_object.device).float() / 255.0
        if len(im.shape) == 3:
            im = im[None]
        pred = model_object(im, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, conf_threshold, iou_same_class_threshold, max_det=1000)[0]
        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], image_bgr.shape).round()
            boxes_after_nms_tensor = pred
        else:
            boxes_after_nms_tensor = torch.empty(0, 6, device=model_object.device)
    else:
        results = model_object(image_pil, conf=conf_threshold, iou=iou_same_class_threshold, verbose=False)
        boxes_after_nms_tensor = results[0].boxes.data if len(results[0].boxes) > 0 else torch.empty(0, 6, device=model_object.device)

    final_boxes = _filter_boxes_gpu(boxes_after_nms_tensor, iou_diff_class_threshold)
    elapsed = time.perf_counter() - t_start
    return final_boxes, names, elapsed

def _process_single_model_thread(model_obj, is_original, image_pil, image_bgr, params, result_list, index):
    """Wrapper cho threading."""
    if model_obj is None:
        result_list[index] = ([], {}, 0.0)
        return
    try:
        boxes, names, elapsed = _inference_single_model(model_obj, is_original, image_pil, image_bgr, params)
        result_list[index] = (boxes, names, elapsed)
    except Exception as e:
        print(f"Lỗi khi suy luận model {index+1}: {e}")
        result_list[index] = ([], {}, 0.0)


# ============================================================================
# PHẦN 5: OCR
# ============================================================================

def _get_ocr_model(device_mode):
    global OCR_MODELS
    device_key = 'cuda' if device_mode == 'GPU' and torch.cuda.is_available() else 'cpu'
    if device_key in OCR_MODELS:
        return OCR_MODELS[device_key]
    if not DOCTR_AVAILABLE:
        raise ImportError("Thư viện 'python-doctr' chưa được cài đặt.")
    print(f"--- Tải model OCR (docTR) cho '{device_key}'... ---")
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True).to(device_key)
    OCR_MODELS[device_key] = model
    return model

def _get_manga_ocr_model(device_mode):
    global MANGA_OCR_MODELS
    device_key = 'GPU' if device_mode == 'GPU' and torch.cuda.is_available() else 'CPU'
    if device_key in MANGA_OCR_MODELS:
        return MANGA_OCR_MODELS[device_key]
    if not MANGA_OCR_AVAILABLE:
        raise ImportError("Thư viện 'manga-ocr' chưa được cài đặt.")
    force_cpu_flag = (device_key == 'CPU')
    print(f"--- Tải model Manga-OCR cho '{device_key}'... ---")
    model = MangaOcr(force_cpu=force_cpu_flag)
    MANGA_OCR_MODELS[device_key] = model
    return model

def _is_word_in_box(word_box, text_box, threshold=0.5):
    x1_inter = max(word_box[0], text_box[0]); y1_inter = max(word_box[1], text_box[1])
    x2_inter = min(word_box[2], text_box[2]); y2_inter = min(word_box[3], text_box[3])
    inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)
    word_area = (word_box[2] - word_box[0]) * (word_box[3] - word_box[1])
    if word_area == 0:
        return False
    return (inter_area / word_area) > threshold

def _run_ocr_english(image_np_bgr, final_boxes, model_names, device_mode):
    """OCR Tiếng Anh (docTR) - tinh chỉnh box + trích xuất text."""
    logs = []
    try:
        ocr_model = _get_ocr_model(device_mode)
    except Exception as e:
        logs.append(f"Lỗi tải docTR: {e}")
        return final_boxes, {}, logs

    t_start = time.perf_counter()
    image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
    ocr_result = ocr_model([image_np_rgb])
    h, w, _ = image_np_bgr.shape

    all_words = []
    for page in ocr_result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    x_min, y_min = word.geometry[0]
                    x_max, y_max = word.geometry[1]
                    abs_box = [int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)]
                    all_words.append({'box': abs_box, 'text': word.value})

    text_class_id = next((cls_id for cls_id, name in model_names.items() if name == 'text'), -1)
    if text_class_id == -1:
        logs.append("Không tìm thấy class 'text', bỏ qua OCR.")
        return final_boxes, {}, logs

    # Tạo mapping box → ocr_text và refine bbox
    ocr_data_map = {}  # key: tuple(x1,y1,x2,y2) -> text
    refined_boxes = []
    other_boxes = [b for b in final_boxes if b[5] != text_class_id]
    yolo_text_boxes = [b for b in final_boxes if b[5] == text_class_id]

    for text_box in yolo_text_boxes:
        contained_words = [word for word in all_words if _is_word_in_box(word['box'], text_box[:4])]
        if contained_words:
            word_coords = np.array([w['box'] for w in contained_words])
            min_x, min_y = np.min(word_coords[:, [0, 1]], axis=0)
            max_x, max_y = np.max(word_coords[:, [2, 3]], axis=0)
            refined_box = [min_x, min_y, max_x, max_y, text_box[4], text_box[5]]
            refined_boxes.append(refined_box)
            ocr_data_map[tuple(map(int, refined_box[:4]))] = ' '.join([w['text'] for w in contained_words])
        else:
            refined_boxes.append(text_box)

    logs.append(f"OCR (docTR): {time.perf_counter() - t_start:.4f}s")
    return refined_boxes + other_boxes, ocr_data_map, logs

def _run_ocr_japanese(image_np_bgr, final_boxes, model_names, device_mode):
    """OCR Tiếng Nhật (MangaOCR) - chỉ trích xuất text."""
    logs = []
    try:
        ocr_model = _get_manga_ocr_model(device_mode)
    except Exception as e:
        logs.append(f"Lỗi tải Manga-OCR: {e}")
        return final_boxes, {}, logs

    t_start = time.perf_counter()
    text_class_id = next((cls_id for cls_id, name in model_names.items() if name == 'text'), -1)
    if text_class_id == -1:
        logs.append("Không tìm thấy class 'text', bỏ qua OCR.")
        return final_boxes, {}, logs

    image_pil = Image.fromarray(cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB))
    ocr_data_map = {}
    yolo_text_boxes = [box for box in final_boxes if box[5] == text_class_id]

    for text_box in yolo_text_boxes:
        x1, y1, x2, y2 = map(int, text_box[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_pil.width, x2), min(image_pil.height, y2)
        if x1 >= x2 or y1 >= y2:
            continue
        try:
            cropped_pil = image_pil.crop((x1, y1, x2, y2))
            recognized_text = ocr_model(cropped_pil)
            ocr_data_map[(x1, y1, x2, y2)] = recognized_text
        except Exception as ocr_err:
            print(f"Lỗi Manga-OCR box ({x1},{y1})-({x2},{y2}): {ocr_err}")

    logs.append(f"OCR (Manga-OCR): {time.perf_counter() - t_start:.4f}s")
    return final_boxes, ocr_data_map, logs


# ============================================================================
# PHẦN 6: MASK GENERATION
# ============================================================================

def _create_mask_from_bboxes(image_shape, bboxes):
    """Tạo mask nhị phân từ danh sách bounding box."""
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    if not bboxes:
        return mask
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def _create_difference_mask(img_raw, img_clean, blur_level, min_area, cleanup_level):
    """Tạo mask từ sự khác biệt pixel (SSIM)."""
    h, w, _ = img_raw.shape
    img_clean_resized = cv2.resize(img_clean, (w, h))
    gray_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    gray_clean = cv2.cvtColor(img_clean_resized, cv2.COLOR_BGR2GRAY)

    blur_level = blur_level + 1 if blur_level % 2 == 0 else blur_level
    blurred_raw = cv2.GaussianBlur(gray_raw, (blur_level, blur_level), 0)
    blurred_clean = cv2.GaussianBlur(gray_clean, (blur_level, blur_level), 0)

    (_, diff) = ssim(blurred_raw, blurred_clean, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernel_open = np.ones((3, 3), np.uint8)
    mask_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=2)

    cleanup_level = cleanup_level + 1 if cleanup_level % 2 == 0 else cleanup_level
    kernel_close = np.ones((cleanup_level, cleanup_level), np.uint8)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask_closed)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(final_mask, [cnt], -1, (255), -1)
    return final_mask

def _feather_mask(mask, expand_size, feather_amount):
    """Làm mịn và mở rộng viền mask."""
    expand_size = expand_size + 1 if expand_size % 2 == 0 else expand_size
    feather_amount = feather_amount + 1 if feather_amount % 2 == 0 else feather_amount
    if expand_size > 1:
        kernel = np.ones((expand_size, expand_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        dilated_mask = mask
    if feather_amount > 1:
        feathered_mask = cv2.GaussianBlur(dilated_mask, (feather_amount, feather_amount), 0)
    else:
        feathered_mask = dilated_mask
    return feathered_mask

def _filter_blobs_by_overlap(mask_diff, mask_allowed, overlap_threshold=0.5):
    """Lọc connected components: giữ blob nằm trong mask_allowed > threshold."""
    if overlap_threshold <= 0:
        return cv2.bitwise_and(mask_diff, mask_allowed)

    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_diff)
    final_output = np.zeros_like(mask_diff)

    for i in range(1, num_labels):
        blob_mask = (labels_im == i).astype(np.uint8) * 255
        blob_area = stats[i, cv2.CC_STAT_AREA]
        intersection = cv2.bitwise_and(blob_mask, mask_allowed)
        intersection_area = cv2.countNonZero(intersection)

        if blob_area == 0:
            continue
        overlap_ratio = intersection_area / blob_area
        if overlap_ratio >= overlap_threshold:
            final_output = cv2.bitwise_or(final_output, blob_mask)

    return final_output


# ============================================================================
# PHẦN 7: HÀM CHÍNH - PIPELINE ĐẦY ĐỦ
# ============================================================================

def _load_and_cache_models(device_mode):
    """Tải và cache 3 YOLO models."""
    global LOADED_MODELS_CACHE
    logs = []
    device_str = 'cuda:0' if device_mode == 'GPU' and torch.cuda.is_available() else 'cpu'
    target_device = torch.device(device_str)
    models_to_run = [None, None, None]

    for i, model_path in enumerate(MODEL_PATHS):
        if not (model_path and os.path.exists(model_path)):
            continue

        cached = LOADED_MODELS_CACHE[i]
        is_original = (i == 1 and IS_MODEL2_YOLOV9_ORIGINAL)

        should_reload = (
            cached is None or
            cached.get('temp_path') != model_path or
            cached.get('device') != device_mode or
            (i == 1 and cached.get('is_original') != is_original)
        )

        if should_reload:
            model_filename = os.path.basename(model_path)
            persistent_path = os.path.join(MODEL_CACHE_DIR, f"cached_model_{i+1}_{model_filename}")
            if not os.path.exists(persistent_path) or not filecmp.cmp(model_path, persistent_path, shallow=False):
                shutil.copy(model_path, persistent_path)

            t_load = time.perf_counter()
            if is_original:
                model_obj = DetectMultiBackend(persistent_path, device=target_device, data=ROOT / 'data/coco.yaml')
            else:
                model_obj = YOLO(persistent_path)
                model_obj.to(target_device)

            LOADED_MODELS_CACHE[i] = {
                'path': persistent_path,
                'temp_path': model_path,
                'model': model_obj,
                'device': device_mode,
                'is_original': is_original,
            }
            logs.append(f"Tải Model {i+1}: {time.perf_counter() - t_load:.4f}s")
        else:
            logs.append(f"Model {i+1}: từ cache")

        if LOADED_MODELS_CACHE[i]:
            models_to_run[i] = LOADED_MODELS_CACHE[i]['model']

    return models_to_run, logs


def format_bbox_json(final_boxes, model_names, ocr_data_map=None):
    """Chuyển đổi bbox thành JSON format chuẩn."""
    if ocr_data_map is None:
        ocr_data_map = {}

    result = []
    for box in final_boxes:
        x1, y1, x2, y2, conf, cls_id = box
        cls_id = int(cls_id)
        class_name = model_names.get(cls_id, f"class_{cls_id}")

        # Tìm OCR text cho box này
        box_key = (int(x1), int(y1), int(x2), int(y2))
        ocr_text = ocr_data_map.get(box_key, "")

        result.append({
            "class": class_name,
            "bbox": [round(float(x1)), round(float(y1)), round(float(x2)), round(float(y2))],
            "confidence": round(float(conf), 4),
            "ocr_text": ocr_text,
        })

    return result


def run_full_pipeline(
    raw_image_pil,
    clean_image_pil,
    device_mode="Auto",
    ocr_mode="Không bật",
    mask_classes=None,
    mask_params=None,
):
    """
    Pipeline đầy đủ: Detection → Ensemble → OCR → Mask Generation.

    Args:
        raw_image_pil: Ảnh gốc (PIL Image)
        clean_image_pil: Ảnh clean/đã xóa text (PIL Image)
        device_mode: "Auto" / "CPU" / "GPU"
        ocr_mode: "Không bật" / "Tiếng Anh (Tinh chỉnh box)" / "Tiếng Nhật (Chỉ trích xuất)"
        mask_classes: List class để tạo mask, mặc định dùng config
        mask_params: Override mask params, mặc định dùng config

    Returns:
        dict: {
            "bbox_json": list,      # JSON data bounding boxes
            "final_mask": np.array, # Mask cuối (grayscale numpy)
            "logs": list            # Log chi tiết
        }
    """
    from config import resolve_device, DEFAULT_MASK_CLASSES

    if mask_classes is None:
        mask_classes = DEFAULT_MASK_CLASSES
    if mask_params is None:
        mask_params = CONFIG['mask_defaults'].copy()

    logs = []
    t_total_start = time.perf_counter()

    # --- Resolve device ---
    actual_device = resolve_device(device_mode)
    logs.append(f"Device: {actual_device} (chọn: {device_mode})")

    # --- Chuẩn bị ảnh ---
    raw_np_bgr = cv2.cvtColor(np.array(raw_image_pil), cv2.COLOR_RGB2BGR)
    clean_np_bgr = cv2.cvtColor(np.array(clean_image_pil), cv2.COLOR_RGB2BGR)

    # =============================================
    # BƯỚC 1: Tải model
    # =============================================
    t_step = time.perf_counter()
    models_to_run, model_logs = _load_and_cache_models(actual_device)
    logs.extend(model_logs)
    logs.append(f"[Bước 1] Tải Models: {time.perf_counter() - t_step:.4f}s")

    # =============================================
    # BƯỚC 2: Chạy 3 model song song
    # =============================================
    t_step = time.perf_counter()
    single_defaults = CONFIG['single_model_defaults']
    inference_params = (
        single_defaults['conf_threshold'],
        single_defaults['iou_same_class_threshold'],
        single_defaults['iou_diff_class_threshold'],
        actual_device,
    )

    final_results = [([], {}, 0.0)] * 3
    model_infos = [
        (models_to_run[0], False),
        (models_to_run[1], IS_MODEL2_YOLOV9_ORIGINAL),
        (models_to_run[2], False),
    ]

    threads = [
        threading.Thread(
            target=_process_single_model_thread,
            args=(*info, raw_image_pil, raw_np_bgr, inference_params, final_results, i)
        )
        for i, info in enumerate(model_infos)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logs.append(f"[Bước 2] Inference 3 Models: {time.perf_counter() - t_step:.4f}s")

    # =============================================
    # BƯỚC 3: Ensemble Pipeline
    # =============================================
    t_step = time.perf_counter()
    filtered_results_for_ensemble = [res[0] for res in final_results if res[0]]
    combined_names = {}
    for res in final_results:
        combined_names.update(res[1])

    final_boxes = []
    if len(filtered_results_for_ensemble) > 1 and combined_names:
        h, w, _ = raw_np_bgr.shape
        ensemble_defaults = CONFIG['ensemble_defaults']
        runtime_config = CONFIG.copy()
        runtime_config['current_settings'] = {
            "wbf_iou_thr": ensemble_defaults['wbf_iou_thr'],
            "wbf_skip_box_thr": ensemble_defaults['wbf_skip_box_thr'],
            "final_conf_threshold": ensemble_defaults['final_conf_threshold'],
            "parent_child_containment_thr": ensemble_defaults['parent_child_containment_thr'],
            "ensemble_inter_class_iou_thr": ensemble_defaults['ensemble_inter_class_iou_thr'],
            "ensemble_intra_group_iou_thr": ensemble_defaults['ensemble_intra_group_iou_thr'],
        }

        pipeline = AdvancedDetectionPipeline(
            class_mapping=combined_names,
            config=runtime_config,
            image_size=(w, h),
        )
        final_boxes, pipeline_logs = pipeline.process(filtered_results_for_ensemble, actual_device)
        logs.extend(pipeline_logs)
    elif len(filtered_results_for_ensemble) == 1:
        final_boxes = filtered_results_for_ensemble[0]
        logs.append("Chỉ có 1 model có kết quả, bỏ qua ensemble.")
    else:
        logs.append("Không có model nào có kết quả.")

    logs.append(f"[Bước 3] Ensemble: {time.perf_counter() - t_step:.4f}s")

    # =============================================
    # BƯỚC 4: OCR (nếu bật)
    # =============================================
    t_step = time.perf_counter()
    ocr_data_map = {}

    if ocr_mode != "Không bật" and final_boxes and combined_names:
        if ocr_mode == "Tiếng Anh (Tinh chỉnh box)":
            final_boxes, ocr_data_map, ocr_logs = _run_ocr_english(raw_np_bgr, final_boxes, combined_names, actual_device)
        elif ocr_mode == "Tiếng Nhật (Chỉ trích xuất)":
            final_boxes, ocr_data_map, ocr_logs = _run_ocr_japanese(raw_np_bgr, final_boxes, combined_names, actual_device)
        logs.extend(ocr_logs)
        logs.append(f"[Bước 4] OCR ({ocr_mode}): {time.perf_counter() - t_step:.4f}s")
    else:
        logs.append(f"[Bước 4] OCR: Không bật")

    # =============================================
    # BƯỚC 5: Tạo BBox JSON
    # =============================================
    bbox_json = format_bbox_json(final_boxes, combined_names, ocr_data_map)

    # =============================================
    # BƯỚC 6: Trích xuất bbox theo class cho mask
    # =============================================
    t_step = time.perf_counter()
    bboxes_by_class = {}
    for class_name in MASKABLE_CLASSES:
        class_id = next((cid for cid, name in combined_names.items() if name == class_name), -1)
        if class_id != -1:
            class_bboxes = [box for box in final_boxes if box[5] == class_id]
            if class_bboxes:
                bboxes_by_class[class_name] = class_bboxes

    # Phân loại selected/unselected
    selected_bboxes = []
    unselected_bboxes = []
    for class_name in MASKABLE_CLASSES:
        if class_name in bboxes_by_class:
            if class_name in mask_classes:
                selected_bboxes.extend(bboxes_by_class[class_name])
            else:
                unselected_bboxes.extend(bboxes_by_class[class_name])

    # =============================================
    # BƯỚC 7: Tạo Mask
    # =============================================
    final_mask = np.zeros((raw_np_bgr.shape[0], raw_np_bgr.shape[1]), dtype=np.uint8)

    if selected_bboxes:
        # 7a. Mask YOLO từ selected bbox
        yolo_mask = _create_mask_from_bboxes(raw_np_bgr.shape, selected_bboxes)

        # 7b. Diff mask (SSIM)
        diff_mask = _create_difference_mask(
            raw_np_bgr, clean_np_bgr,
            mask_params['blur'], mask_params['min_area'], mask_params['cleanup']
        )

        # 7c. Filter blob thông minh
        overlap_thresh = mask_params.get('overlap_threshold', 0.1)
        smart_filtered = _filter_blobs_by_overlap(diff_mask, yolo_mask, overlap_thresh)

        # 7d. Feather + expand
        final_mask = _feather_mask(smart_filtered, mask_params['expand'], mask_params['feather'])
        logs.append(f"[Bước 5] Tạo Mask ({len(selected_bboxes)} bbox): {time.perf_counter() - t_step:.4f}s")
    else:
        logs.append("[Bước 5] Không có bbox được chọn → Mask rỗng")

    # =============================================
    # KẾT THÚC
    # =============================================
    logs.insert(0, f"=== TỔNG THỜI GIAN: {time.perf_counter() - t_total_start:.4f}s ===")

    return {
        "bbox_json": bbox_json,
        "final_mask": final_mask,
        "logs": logs,
    }
