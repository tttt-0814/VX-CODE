from typing import List, Tuple

import torch
from detectron2.structures import Boxes
from detectron2.structures.boxes import pairwise_iou


def compute_similarity_based_on_score_and_iou(
    target_box: Boxes,
    target_score: torch.Tensor,
    target_label: torch.Tensor,
    pred_boxes: Boxes,
    pred_scores: torch.Tensor,
    image_shape: Tuple[int, int],
    model_name: str = "faster_r_cnn",
) -> torch.tensor:
    # Get iou
    list_iou, _ = get_max_iou_id_in_proposals(
        target_box, target_label, pred_boxes, image_shape, model_name=model_name
    )
    # Get cosine similarity for score
    batch_size = len(pred_boxes)
    cos_sim = torch.nn.CosineSimilarity()
    list_cos_sim_scores = list()
    for i in range(batch_size):
        pred_score_i = pred_scores[i]
        cos_sim_scores = cos_sim(pred_score_i, target_score)
        list_cos_sim_scores.append(cos_sim_scores)

    batch_iou = torch.stack(list_iou)
    batch_cos_sim_scores = torch.stack(list_cos_sim_scores)
    pairwise_sim_all = batch_iou[:, :, 0] * batch_cos_sim_scores
    pairwise_sim_max, _ = torch.max(pairwise_sim_all, dim=1)

    return pairwise_sim_max, batch_iou[0], batch_cos_sim_scores[0]


def get_max_iou_id_in_proposals(
    target_box: Boxes,
    target_label: torch.tensor,
    proposal_boxes: Boxes,
    image_shape: Tuple[int, int],
    model_name: str = "faster_r_cnn",
) -> Tuple[List[torch.tensor], List[torch.tensor]]:
    list_iou, list_target_id = list(), list()
    if model_name == "faster_r_cnn":
        H, W = image_shape[0], image_shape[1]
        num_bbox_reg_classes = proposal_boxes[0].shape[1] // 4
        batch_size = len(proposal_boxes)
        # Convert to Boxes to use the 'clip' function ...
        for i in range(batch_size):
            proposal_boxes_i = proposal_boxes[i]
            proposal_boxes_i = Boxes(proposal_boxes_i.reshape(-1, 4))
            proposal_boxes_i.clip((H, W))
            proposal_boxes_i = proposal_boxes_i.tensor.view(
                -1, num_bbox_reg_classes, 4
            )  # R x C x 4
            proposal_boxes_i = Boxes(proposal_boxes_i[:, target_label, :])
            iou = pairwise_iou(proposal_boxes_i, target_box)
            target_id = torch.argmax(iou)
            list_iou.append(iou)
            list_target_id.append(target_id)

    if model_name == "detr":
        H, W = image_shape[0], image_shape[1]
        batch_size = len(proposal_boxes)
        # list_iou, list_target_id = list(), list()
        # Convert to Boxes to use the 'clip' function ...
        for i in range(batch_size):
            proposal_boxes_i = proposal_boxes[i]
            proposal_boxes_i = Boxes(proposal_boxes_i)
            proposal_boxes_i.clip((H, W))
            iou = pairwise_iou(proposal_boxes_i, target_box)
            target_id = torch.argmax(iou)
            list_iou.append(iou)
            list_target_id.append(target_id)

    return list_iou, list_target_id


def compute_similarity_score(
    target_box: Boxes,
    target_score: torch.Tensor,
    target_label: torch.Tensor,
    proposal_boxes: Boxes,
    proposal_scores: torch.Tensor,
    image_shape: Tuple[int, int],
    model_name: str = "faster_r_cnn",
) -> torch.tensor:
    return compute_similarity_based_on_score_and_iou(
        target_box,
        target_score,
        target_label,
        proposal_boxes,
        proposal_scores,
        image_shape,
        model_name=model_name,
    )
