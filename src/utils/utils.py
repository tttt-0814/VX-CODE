from typing import Dict, List, Tuple, Union


import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures.instances import Instances
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference


def compute_predictions_for_proposals(
    model: nn.Module,
    inputs: List[Dict[str, torch.Tensor]],
    return_final_results: bool = True,
    model_name: str = "faster_r_cnn",
) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Union[List[Instances], None]]:
    if model_name == "detr":
        images = model.preprocess_image(inputs)
        output = model.detr(images)
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        mask_pred = output["pred_masks"] if model.mask_on else None

        results = model.inference(box_cls, box_pred, mask_pred, images.image_sizes)
        proposal_pred_boxes = results[0].pred_boxes.tensor[None, ...]
        proposal_pred_scores = F.softmax(box_cls, dim=-1)

        if return_final_results:
            results = model.inference(box_cls, box_pred, mask_pred, images.image_sizes)
            pred_instances = []
            for results_per_image in results:
                pred_instances.append(results_per_image)

    if model_name == "faster_r_cnn":
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)

        proposals, _ = model.proposal_generator(images, features, None)
        features = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = model.roi_heads.box_head(box_features)
        predictions = model.roi_heads.box_predictor(box_features)
        proposal_pred_boxes = model.roi_heads.box_predictor.predict_boxes(
            predictions, proposals
        )
        proposal_pred_scores = model.roi_heads.box_predictor.predict_probs(
            predictions, proposals
        )

        if return_final_results:
            image_shapes = [x.image_size for x in proposals]
            pred_instances, _ = fast_rcnn_inference(
                proposal_pred_boxes,
                proposal_pred_scores,
                image_shapes,
                model.roi_heads.box_predictor.test_score_thresh,
                model.roi_heads.box_predictor.test_nms_thresh,
                model.roi_heads.box_predictor.test_topk_per_image,
            )

    if return_final_results:
        return proposal_pred_boxes, proposal_pred_scores, pred_instances
    else:
        return proposal_pred_boxes, proposal_pred_scores


def get_target_instances_from_preds(
    model: nn.Module,
    inputs: List[Instances],
    score_thresh: float = None,
    model_name: str = "faster_r_cnn",
) -> Tuple[List[Instances], List[Instances], torch.Tensor]:
    if model_name == "faster_r_cnn":
        results = model.inference(inputs, do_postprocess=False)
    if model_name == "detr":
        results = model(inputs, do_postprocess=False)

    pred_scores = results[0].scores.detach()
    thresh = 0.0 if score_thresh is None else score_thresh
    target_mask = pred_scores > thresh
    target_flag = target_mask.to(torch.float32).to("cpu")
    filtered_results = [results[0][target_mask]]

    return results, filtered_results, target_flag


def insert_patch_img_on_canvas(
    start_img: torch.Tensor,
    patch_img: torch.Tensor,
    patch_id: torch.Tensor,
    patch_size: Tuple[int, int],
    num_patches: Tuple[int, int],
    padding_len: Tuple[int, int],
) -> torch.Tensor:
    patch_size_h, patch_size_w = patch_size
    num_patches_h, num_patches_w = num_patches
    padding_len_h, padding_len_w = padding_len
    h_id, w_id = patch_id.int()
    _patch_img = patch_img[:, h_id, w_id, :, :]

    if h_id + 1 == num_patches_h:
        _patch_img = _patch_img[:, : patch_size_h - padding_len_h, :]
    if w_id + 1 == num_patches_w:
        _patch_img = _patch_img[:, :, : patch_size_w - padding_len_w]

    input_img = start_img.clone()
    start_h = h_id * patch_size_h
    start_w = w_id * patch_size_w

    input_img[
        :,
        start_h : start_h + _patch_img.shape[1],
        start_w : start_w + _patch_img.shape[2],
    ] = _patch_img

    return input_img


def delete_patch_img_from_canvas(
    start_img: torch.Tensor,
    patch_img: torch.Tensor,
    patch_id: torch.Tensor,
    patch_size: Tuple[int, int],
    num_patches: Tuple[int, int],
    padding_len: Tuple[int, int],
) -> torch.Tensor:
    patch_size_h, patch_size_w = patch_size
    num_patches_h, num_patches_w = num_patches
    padding_len_h, padding_len_w = padding_len
    h_id, w_id = patch_id.int()
    _patch_img = patch_img[:, h_id, w_id, :, :]

    if h_id + 1 == num_patches_h:
        _patch_img = _patch_img[:, : patch_size_h - padding_len_h, :]
    if w_id + 1 == num_patches_w:
        _patch_img = _patch_img[:, :, : patch_size_w - padding_len_w]

    input_img = start_img.clone()
    start_h = h_id * patch_size_h
    start_w = w_id * patch_size_w

    input_img[
        :,
        start_h : start_h + _patch_img.shape[1],
        start_w : start_w + _patch_img.shape[2],
    ] = _patch_img

    return input_img
