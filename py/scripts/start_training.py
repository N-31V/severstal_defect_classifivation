#!/usr/bin/python3

import argparse
from model_toolkit import ModelToolkit
from torchvision import models


def create_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--epochs', type=int, default=20, help='int')
    argparser.add_argument('-n', '--num_workers', type=int, default=4, help='int')
    argparser.add_argument('-b', '--batch_size', type=int, default=4, help='int')
    argparser.add_argument('-c', '--checkpoint', type=str, default=None, help='model checkpoint')
    return argparser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    model = ModelToolkit(models.resnet50(), 'resNet50', checkpoint=args.checkpoint, batch_size=args.batch_size, num_workers=args.num_workers)
    model.train(args.epochs)
