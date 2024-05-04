import numpy as np
import cv2

# From https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection

class_names = ['ally_archer', 'ally_arrow', 'ally_baby_dragon', 'ally_giant', 'ally_bandit', 'ally_barbarian',
               'ally_bat', 'ally_knight', 'ally_minion', 'ally_mini_pekka', 'ally_musketeer', 'ally_brawler', 'ally_cursedhog', 
               'ally_dark_prince', 'ally_electro_wizard', 'ally_elite_barbarian', 'ally_elixirpump', 'ally_executioner', 'ally_fire_spirit', 'ally_fireball',
               'ally_firecracker', 'ally_fisherman', 'ally_balloon', 'ally_giant_bomb', 'ally_giant_skeleton', 'ally_goblin', 'ally_goblin_barrel', 'ally_goblin_cage', 
               'ally_goblin_hut', 'ally_goblindrill', 'ally_golem_mini', 'ally_graveyard', 'ally_guard', 'ally_hog_rider', 'ally_hungry_dragon', 'ally_hunter', 'ally_ice_golem', 
               'ally_ice_spirit', 'ally_ice_wizard', 'ally_infernodragon', 'ally_battleram', 'ally_knight_evo', 'ally_log', 'ally_lumberjack', 'ally_magic_archer', 'ally_mega_knight', 
               'ally_mega_minion', 'ally_mightyminer', 'ally_bombtower', 'ally_bomber', 'ally_mortar', 'ally_motherwitch', 'ally_bowler', 'ally_pekka', 'ally_poison', 'ally_prince', 
               'ally_princess', 'ally_rage', 'ally_rocket', 'ally_royale_giant', 'ally_royalghost', 'ally_skeleton', 'ally_skeletonking', 'ally_spear_goblin', 'ally_tesla', 
               'ally_tesla_hidden', 'ally_tombstone', 'ally_tornado', 'ally_valkyrie', 'ally_wallbreaker', 'ally_witch', 'ally_wizard', 'ally_xbow', 'ally_zappy', 'enemy_archer',
               'enemy_arrow', 'enemy_baby_dragon', 'enemy_bandit', 'enemy_barbarian', 'enemy_barbarian_evo', 'enemy_bat', 'enemy_bomb_tower', 'enemy_bomber', 'enemy_brawler',
               'enemy_dark_prince', 'enemy_dart_goblin', 'enemy_electro_wizard', 'enemy_electrogiant', 'enemy_elixirgolem', 'enemy_executioner', 'enemy_fire_spirit', 'enemy_fireball',
               'enemy_firecracker', 'enemy_furnace', 'enemy_giant', 'enemy_giant_bomb', 'enemy_giant_skeleton', 'enemy_goblin', 'enemy_goblin_barrel', 'enemy_goblin_cage', 'enemy_goblin_hut',
               'enemy_goblinbarrel', 'enemy_goldenknight', 'enemy_golem', 'enemy_golem_mini', 'enemy_guard', 'enemy_hog', 'enemy_hog_rider', 'enemy_hunter', 'enemy_ice_golem',
               'enemy_inferno_tower', 'enemy_infernodragon', 'enemy_knight', 'enemy_log', 'enemy_lumberjack', 'enemy_magic_archer', 'enemy_mega_knight', 'enemy_mega_minion',
               'enemy_miner', 'enemy_mini_pekka', 'enemy_minion', 'enemy_musketeer', 'enemy_night_witch', 'enemy_pekka', 'enemy_prince', 'enemy_princess', 'enemy_rage',
               'enemy_ramrider', 'enemy_rascalboy', 'enemy_rascalgirl', 'enemy_rocket', 'enemy_royal_ghost', 'enemy_skeleton', 'enemy_skeleton_barrel',
               'enemy_skeleton_evo', 'enemy_sparky', 'enemy_spear_goblin', 'enemy_tesla', 'enemy_tesla_hidden', 
               'enemy_tombstone', 'enemy_valkyrie', 'enemy_witch', 'enemy_wizard', 'enemy_xbow']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def multiclass_nms(boxes, scores, class_ids, iou_threshold):

    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box( image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
             thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)
