from typing import Dict, List, Tuple
import numpy as np
import copy
import itertools

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from detectron2.structures.instances import Instances
from detectron2.structures import Boxes

from src.explanations.base import BaseExplainer
from src.utils.reward_functions import compute_similarity_score
from src.utils.utils import (
    compute_predictions_for_proposals,
    insert_patch_img_on_canvas,
    delete_patch_img_from_canvas,
)


class VXCODE(BaseExplainer):
    def __init__(
        self,
        model: nn.Module,
        model_name: str = "faster_r_cnn",
        mode: str = "del",
        num_patches_per_step: int = 1,
        batch_size: int = 1,
        box_area_ratio: float = 0.20,
        ratio_to_stop_combination: float = 0.1,
        topk: int = 30,
        use_ps: bool = True,
        use_rs: bool = True,
        perturbation_mode: str = "zero",
    ):
        super().__init__(model, model_name)

        self.mode = mode
        self.num_patches_per_step = num_patches_per_step
        self.batch_size = batch_size
        self.box_area_ratio = box_area_ratio
        self.ratio_to_stop_combination = ratio_to_stop_combination
        self.topk = topk

        self.use_ps = use_ps
        self.use_rs = use_rs
        self.perturbation_mode = perturbation_mode

        if not self.use_rs:
            self.ratio_to_stop_combination = 1.0

        if self.perturbation_mode == "blur":
            kernel_size = 201
            sigma_x = 200
            self.blur_func = torchvision.transforms.GaussianBlur(
                (kernel_size, kernel_size), sigma=sigma_x
            )

    def __call__(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        target_results: List[Instances],
        target_flag: torch.Tensor = None,
    ) -> Tuple[
        List[Instances],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        Tuple[int, int],
        Tuple[int, int],
    ]:
        # Get proposal predictions and identify target instances
        (_original_proposal_pred_boxes, _original_proposal_pred_scores, _) = (
            compute_predictions_for_proposals(
                self.model,
                inputs,
                return_final_results=True,
                model_name=self.model_name,
            )
        )
        _original_proposal_pred_boxes = _original_proposal_pred_boxes[0].detach()
        _original_proposal_pred_scores = _original_proposal_pred_scores[0].detach()

        # Identify target instances
        _target_boxes = Boxes(target_results[0].pred_boxes.tensor.detach())
        _target_scores = target_results[0].scores.detach()
        _target_labels = target_results[0].pred_classes.detach()
        _target_scores_expanded = _target_scores.view(-1, 1)
        _target_ids = (
            _original_proposal_pred_scores[:, _target_labels].T
            == _target_scores_expanded
        )
        target_scores = torch.concat(
            [_original_proposal_pred_scores[target_id] for target_id in _target_ids]
        )
        target_boxes = _target_boxes
        target_labels = _target_labels

        original_img = inputs[0]["image"]
        num_detected_instances = len(target_boxes)
        list_results_patch_id = [[] for i in range(num_detected_instances)]
        list_results_rewards = [[] for i in range(num_detected_instances)]
        list_results_patch_imgs = list()
        list_results_patch_size = list()
        list_results_padding_len = list()
        list_results_eval_heatmap = list()
        for idx, target_instance_num in enumerate(range(num_detected_instances)):
            target_box_i = target_boxes[target_instance_num]
            target_score_i = target_scores[target_instance_num]
            target_label_i = target_labels[target_instance_num]

            # Prepare canvas and pre-define conditions.
            if self.mode == "ins":
                if self.perturbation_mode == "zero":
                    canvas = torch.zeros_like(original_img)
                if self.perturbation_mode == "blur":
                    canvas = self.blur_func(original_img)
                img_for_preprocess = copy.deepcopy(original_img)
            else:
                canvas = copy.deepcopy(original_img)
                if self.perturbation_mode == "zero":
                    img_for_preprocess = torch.zeros_like(original_img)
                if self.perturbation_mode == "blur":
                    img_for_preprocess = self.blur_func(original_img)

            # Decide patch size and make padding image.
            (
                patch_img_i,
                patch_indices_i,
                patch_sizes_i,
                num_patches_i,
                padding_len_i,
            ) = self.preprocess(img_for_preprocess, target_box_i)
            list_results_patch_imgs.append(patch_img_i)
            list_results_patch_size.append(patch_sizes_i)
            list_results_padding_len.append(padding_len_i)

            total_num_patches_i = patch_indices_i.shape[0]
            num_topk_i = self.topk

            num_patches_to_stop_combination_i = (
                self.ratio_to_stop_combination * total_num_patches_i
            )
            _patch_indices_i = copy.deepcopy(patch_indices_i)

            # Start to compute rewards to identify patches
            while True:
                num_computed_patches = len(list_results_patch_id[target_instance_num])
                print(
                    "\rComputing for instance {}".format(
                        target_instance_num + 1,
                    ),
                    end=" ",
                )

                if (
                    self.num_patches_per_step > 1
                    and num_computed_patches < num_patches_to_stop_combination_i
                ):
                    flag_combination = True
                else:
                    flag_combination = False

                if len(_patch_indices_i) == 1:
                    flag_combination = False

                # Preprocess to get target patches
                if flag_combination:
                    if self.use_ps:
                        _, target_patch_indices_i = (
                            self.get_topk_patches_from_combinations(
                                inputs,
                                canvas,
                                patch_img_i,
                                _patch_indices_i,
                                patch_sizes_i,
                                num_patches_i,
                                padding_len_i,
                                target_box_i,
                                target_score_i,
                                target_label_i,
                                combinations=1,
                                topk=num_topk_i,
                            )
                        )
                        target_patch_indices_i = target_patch_indices_i.squeeze()
                    else:
                        target_patch_indices_i = _patch_indices_i
                    combinations = self.num_patches_per_step
                else:
                    target_patch_indices_i = _patch_indices_i
                    combinations = 1

                # Get patch pairs
                topk_values, selected_patch_pairs = (
                    self.get_topk_patches_from_combinations(
                        inputs,
                        canvas,
                        patch_img_i,
                        target_patch_indices_i,
                        patch_sizes_i,
                        num_patches_i,
                        padding_len_i,
                        target_box_i,
                        target_score_i,
                        target_label_i,
                        combinations=combinations,
                        topk=1,
                    )
                )

                if flag_combination:
                    # Align patches
                    target_patch_pairs = copy.deepcopy(selected_patch_pairs)
                    target_patch_pairs = target_patch_pairs.squeeze()
                    list_alignment_patch_id, list_alignment_rewards = (
                        self.get_alignment_patches_from_patch_pair(
                            inputs,
                            canvas,
                            target_patch_pairs,
                            patch_img_i,
                            patch_sizes_i,
                            num_patches_i,
                            padding_len_i,
                            target_box_i,
                            target_score_i,
                            target_label_i,
                        )
                    )
                    for i in range(len(list_alignment_patch_id)):
                        list_results_patch_id[target_instance_num].append(
                            list_alignment_patch_id[i]
                        )
                        list_results_rewards[target_instance_num].append(
                            list_alignment_rewards[i]
                        )
                    target_patch_id = list_alignment_patch_id
                else:
                    list_results_patch_id[target_instance_num].append(
                        selected_patch_pairs[0]
                    )
                    list_results_rewards[target_instance_num].append(topk_values[0])
                    target_patch_id = selected_patch_pairs

                # Update values
                _patch_indices_i = self.remove_target_patches(
                    _patch_indices_i, target_patch_id
                )
                if len(_patch_indices_i) == 0:
                    break
                canvas = self.make_input_img_from_patch_pairs(
                    canvas,
                    patch_img_i,
                    selected_patch_pairs,
                    patch_sizes_i,
                    num_patches_i,
                    padding_len_i,
                    mode=self.mode,
                )

            # Make heat map
            eval_heat_map_i = self.make_eval_heatmap(
                list_results_patch_id[target_instance_num],
                list_results_rewards[target_instance_num],
                patch_sizes_i,
                num_patches_i,
                padding_len_i,
                mode=self.mode,
            )
            list_results_eval_heatmap.append(eval_heat_map_i)

        return (
            target_results,
            list_results_eval_heatmap,
            list_results_patch_id,
            list_results_rewards,
            list_results_patch_imgs,
            list_results_patch_size,
            list_results_padding_len,
        )

    def preprocess(
        self, img: torch.Tensor, target_box: Boxes
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Tuple[int, int], Tuple[int, int], Tuple[int, int]
    ]:
        img_size_h, img_size_w = img.shape[1], img.shape[2]

        patch_size_h, patch_size_w, box_area_ratio = self.get_patch_size(
            img_size_h, img_size_w, target_box
        )
        max_size_h = (
            torch.tensor(img_size_h + (patch_size_h - 1)).div(
                patch_size_h, rounding_mode="floor"
            )
            * patch_size_h
        )
        max_size_w = (
            torch.tensor(img_size_w + (patch_size_w - 1)).div(
                patch_size_w, rounding_mode="floor"
            )
            * patch_size_w
        )
        padding_size = [
            0,
            max_size_w - img_size_w,
            0,
            max_size_h - img_size_h,
        ]
        padding_img = F.pad(img, padding_size, value=0)

        # Make patch images
        patch_img = padding_img.unfold(1, patch_size_h, patch_size_h).unfold(
            2, patch_size_w, patch_size_w
        )
        num_patches_h, num_patches_w = (
            patch_img.shape[1],
            patch_img.shape[2],
        )
        patch_indices = torch.Tensor(
            np.array(np.meshgrid(np.arange(num_patches_h), np.arange(num_patches_w)))
            .reshape(2, -1)
            .T
        )

        if box_area_ratio <= self.box_area_ratio:
            patch_indices = self.select_target_patch_indices_based_on_area(
                patch_indices,
                target_box,
                img_size_h,
                img_size_w,
                patch_size_h,
                patch_size_w,
            )
        padding_len_h = padding_img.shape[1] - img_size_h
        padding_len_w = padding_img.shape[2] - img_size_w
        patch_sizes = (patch_size_h, patch_size_w)
        num_patches = (num_patches_h, num_patches_w)
        padding_len = (padding_len_h, padding_len_w)

        return patch_img, patch_indices, patch_sizes, num_patches, padding_len

    def get_topk_patches_from_combinations(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        canvas: torch.Tensor,
        patch_img: torch.Tensor,
        patch_indices: torch.Tensor,
        patch_sizes: Tuple[int, int],
        num_patches: Tuple[int, int],
        padding_len: Tuple[int, int],
        target_box: Boxes,
        target_score: torch.Tensor,
        target_label: torch.Tensor,
        combinations: int = 1,
        topk: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if patch_indices.shape[0] >= combinations:
            patch_pairs = [
                list(combo)
                for combo in itertools.combinations(patch_indices.numpy(), combinations)
            ]
        else:
            patch_pairs = [
                list(combo)
                for combo in itertools.combinations(
                    patch_indices.numpy(), patch_indices.shape[0]
                )
            ]
        patch_pairs = torch.from_numpy(np.stack(patch_pairs))

        batch_patch_pairs = [
            patch_pairs[i : i + self.batch_size]
            for i in range(0, len(patch_pairs), self.batch_size)
        ]
        list_rewards = list()
        for _patch_pairs in batch_patch_pairs:
            # Make batch input image
            list_inputs = list()
            for _patch_pair in _patch_pairs:
                # Make input image
                input_img = copy.deepcopy(canvas)
                input_img = self.make_input_img_from_patch_pairs(
                    input_img,
                    patch_img,
                    _patch_pair,
                    patch_sizes,
                    num_patches,
                    padding_len,
                    mode=self.mode,
                )

                _input = copy.deepcopy(inputs[0])
                _input["image"] = input_img
                list_inputs.append(_input)

            _rewards, _, _ = self.compute_rewards_for_inputs(
                list_inputs, target_box, target_score, target_label
            )
            list_rewards.append(_rewards.detach().to("cpu"))
        rewards = torch.stack(list_rewards)

        topk = topk if rewards.shape[0] >= topk else rewards.shape[0]

        if self.mode == "ins":
            topk_rewards, topk_indices = torch.topk(rewards, topk, dim=0)
        else:
            topk_rewards, topk_indices = torch.topk(-1 * rewards, topk, dim=0)
            topk_rewards = -1 * topk_rewards
        selected_patch_pairs = patch_pairs[topk_indices.squeeze()]

        return topk_rewards, selected_patch_pairs

    def compute_rewards_for_inputs(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        target_box_i: Boxes,
        target_score_i: torch.Tensor,
        target_label_i: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proposal_pred_boxes, proposal_pred_scores = compute_predictions_for_proposals(
            self.model,
            inputs,
            return_final_results=False,
            model_name=self.model_name,
        )
        rewards, iou, cossim = self.compute_rewards(
            target_box_i,
            target_score_i,
            target_label_i,
            proposal_pred_boxes,
            proposal_pred_scores,
            (inputs[0]["image"].shape[1], inputs[0]["image"].shape[2]),
            model_name=self.model_name,
        )

        return (
            rewards,
            iou,
            cossim,
        )

    def get_alignment_patches_from_patch_pair(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        canvas: torch.Tensor,
        patch_pair: List[torch.Tensor],
        patch_img: torch.Tensor,
        patch_sizes: Tuple[int, int],
        num_patches: Tuple[int, int],
        padding_len: Tuple[int, int],
        target_box: Boxes,
        target_score: torch.Tensor,
        target_label: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        list_rewards = list()
        list_sum_rewards = list()
        permutations = list(itertools.permutations(patch_pair))
        for permutation in permutations:
            _list_rewards = list()
            _canvas = copy.deepcopy(canvas)
            for patch_id in permutation:
                input_img = self.make_input_img(
                    _canvas,
                    patch_img,
                    patch_id,
                    patch_sizes,
                    num_patches,
                    padding_len,
                    mode=self.mode,
                )
                _inputs = copy.deepcopy(inputs)
                _inputs[0]["image"] = input_img
                reward, _, _ = self.compute_rewards_for_inputs(
                    _inputs, target_box, target_score, target_label
                )
                _list_rewards.append(reward.detach().to("cpu"))
                _canvas = self.make_input_img(
                    _canvas,
                    patch_img,
                    patch_id,
                    patch_sizes,
                    num_patches,
                    padding_len,
                    mode=self.mode,
                )
            list_rewards.append(_list_rewards)
            list_sum_rewards.append(torch.stack(_list_rewards).sum())

        all_rewards = torch.stack(list_sum_rewards)
        if self.mode == "ins":
            _, selected_index = torch.max(all_rewards, dim=0)
        else:
            _, selected_index = torch.min(all_rewards, dim=0)

        selected_patch_ids = list(permutations[selected_index])
        selected_rewards = list_rewards[selected_index]

        return selected_patch_ids, selected_rewards

    def make_input_img_from_patch_pairs(
        self,
        start_img: torch.Tensor,
        patch_img: torch.Tensor,
        patch_pairs: List[torch.Tensor],
        patch_sizes: Tuple[int, int],
        num_patches: Tuple[int, int],
        padding_len: Tuple[int, int],
        mode: str = "ins",
    ) -> torch.Tensor:
        for _patch_id in patch_pairs:
            start_img = self.make_input_img(
                start_img,
                patch_img,
                _patch_id,
                patch_sizes,
                num_patches,
                padding_len,
                mode=mode,
            )

        return start_img

    def remove_target_patches(
        self, original_patches: torch.Tensor, target_patches: List[torch.Tensor]
    ) -> torch.Tensor:
        for target_patch in target_patches:
            remove_index = torch.sum(original_patches == target_patch, dim=-1) == 2
            original_patches = original_patches[torch.logical_not(remove_index)]

        return original_patches

    def compute_rewards(
        self,
        target_box: Boxes,
        target_score: torch.Tensor,
        target_label: torch.Tensor,
        proposal_boxes: Boxes,
        proposal_scores: torch.Tensor,
        image_shape: Tuple[int, int],
        model_name: str = "faster_r_cnn",
    ) -> torch.tensor:
        return compute_similarity_score(
            target_box,
            target_score,
            target_label,
            proposal_boxes,
            proposal_scores,
            image_shape,
            model_name=model_name,
        )

    def get_patch_size(
        self, img_size_h: int, img_size_w: int, box: Boxes
    ) -> Tuple[int, int, float]:
        img_area = img_size_h * img_size_w
        box_area_ratio = box.area() / img_area

        if 0 < box_area_ratio <= 0.01:
            num_divided = 24
        elif 0.01 < box_area_ratio <= 0.2:
            num_divided = 16
        else:
            num_divided = 8

        # num_divided = 12
        patch_size_h = int(torch.round(torch.tensor(img_size_h / num_divided)))
        patch_size_w = int(torch.round(torch.tensor(img_size_w / num_divided)))

        return patch_size_h, patch_size_w, box_area_ratio

    def select_target_patch_indices_based_on_area(
        self,
        patch_indices: torch.Tensor,
        box: Boxes,
        img_size_h: int,
        img_size_w: int,
        patch_size_h: int,
        patch_size_w: int,
    ) -> torch.Tensor:
        x_min, y_min, x_max, y_max = box.tensor.squeeze()
        search_length_x = img_size_w / 7
        search_length_y = img_size_h / 7
        search_x_min = torch.max(torch.tensor(0), x_min - search_length_x)
        search_x_max = torch.min(torch.tensor(img_size_w), x_max + search_length_x)
        search_y_min = torch.max(torch.tensor(0), y_min - search_length_y)
        search_y_max = torch.min(torch.tensor(img_size_h), y_max + search_length_y)
        new_patch_indices = copy.deepcopy(patch_indices)
        for i in range(len(patch_indices)):
            id_h, id_w = patch_indices[i][0], patch_indices[i][1]
            center_x = id_w * patch_size_w + 0.5 * patch_size_w
            center_y = id_h * patch_size_h + 0.5 * patch_size_h
            if not (
                search_x_min <= center_x <= search_x_max
                and search_y_min <= center_y <= search_y_max
            ):
                _mask = torch.sum(new_patch_indices == patch_indices[i], axis=1) != 2
                new_patch_indices = new_patch_indices[_mask]

        return new_patch_indices

    def make_input_img(
        self,
        start_img: torch.Tensor,
        patch_img: torch.Tensor,
        patch_id: torch.Tensor,
        patch_size: Tuple[int, int],
        num_patches: Tuple[int, int],
        padding_len: Tuple[int, int],
        mode: str = "ins",
    ) -> torch.Tensor:
        if mode == "ins":
            return insert_patch_img_on_canvas(
                start_img, patch_img, patch_id, patch_size, num_patches, padding_len
            )
        else:
            return delete_patch_img_from_canvas(
                start_img,
                patch_img,
                patch_id,
                patch_size,
                num_patches,
                padding_len,
            )

    def make_eval_heatmap(
        self,
        patch_id: torch.Tensor,
        rewards: torch.Tensor,
        patch_size: Tuple[int, int],
        num_patches: Tuple[int, int],
        padding_len: Tuple[int, int],
        mode: int = "ins",
    ) -> torch.Tensor:
        patch_size_h, patch_size_w = patch_size
        num_patches_h, num_patches_w = num_patches
        padding_len_h, padding_len_w = padding_len

        img_size_h = (num_patches_h - 1) * patch_size[0] + (
            patch_size_h - padding_len_h
        )
        img_size_w = (num_patches_w - 1) * patch_size[1] + (
            patch_size_w - padding_len_w
        )
        heatmap = torch.zeros([img_size_h, img_size_w])

        for i in range(len(patch_id)):
            h_id, w_id = patch_id[i].squeeze().int()

            if mode == "ins":
                if i == 0:
                    _reward = 1.0
                else:
                    _reward = 1 - rewards[i - 1]
            else:
                if i == 0:
                    _reward = 1.0
                else:
                    _reward = rewards[i - 1]

            _score = torch.zeros([patch_size_h, patch_size_w]) + _reward
            if h_id + 1 == num_patches_h:
                _score = _score[: patch_size_h - padding_len_h, :]
            if w_id + 1 == num_patches_w:
                _score = _score[:, : patch_size_w - padding_len_w]

            start_h = h_id * patch_size_h
            start_w = w_id * patch_size_w
            heatmap[
                start_h : start_h + _score.shape[0],
                start_w : start_w + _score.shape[1],
            ] = _score

        return heatmap
