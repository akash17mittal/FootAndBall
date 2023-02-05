# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

#
# Run FootAndBall detector on ISSIA-CNR Soccer videos
#

import torch
import cv2
import os
import argparse
import tqdm

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import PLAYER_LABEL, BALL_LABEL


def draw_bboxes(image, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if label == PLAYER_LABEL:
            x1, y1, x2, y2 = box
            color = (255, 0, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (int(x1), max(0, int(y1) - 10)), font, 1, color, 2)

        elif label == BALL_LABEL:
            x1, y1, x2, y2 = box
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            color = (0, 0, 255)
            radius = 25
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (max(0, int(x - radius)), max(0, (y - radius - 10))), font, 1,
                        color, 2)

    return image


def get_detections(model, frame, device="cuda:0"):
    img_tensor = augmentations.numpy2tensor(frame)

    with torch.no_grad():
        # Add dimension for the batch size
        img_tensor = img_tensor.unsqueeze(dim=0).to(device)
        detections = model(img_tensor)[0]

    return detections


def get_model(weights_path="./models/model_20201019_1416_final.pth"):
    # Train the DeepBall ball detector model
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name', type=str, default='fb1')
    parser.add_argument('--ball_threshold', help='ball confidence detection threshold', type=float, default=0.7)
    parser.add_argument('--player_threshold', help='player confidence detection threshold', type=float, default=0.7)
    parser.add_argument('--device', help='device (CPU or CUDA)', type=str, default='cuda:0')
    args = parser.parse_args()

    model = footandball.model_factory(args.model, 'detect', ball_threshold=args.ball_threshold,
                                      player_threshold=args.player_threshold)

    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    if args.device == 'cpu':
        print('Loading CPU weights...')
        state_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)
    else:
        print('Loading GPU weights...')
        state_dict = torch.load(weights_path)

    model.load_state_dict(state_dict)
    # Set model to evaluation mode
    model.eval()

    return model
