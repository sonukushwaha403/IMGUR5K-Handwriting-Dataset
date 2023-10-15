od4
import random
import click
import cv2
from loguru import logger
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


@logger.catch
def crop_minAreaRect(img, bbox):    #xc>0     yc>1    w>2    h>3    a>4
    box = cv2.boxPoints(((bbox[0], bbox[1]), (bbox[2], bbox[3]), -bbox[4]))
    bbox[2], bbox[3] = int(bbox[2]), int(bbox[3])
    box = np.int0(box)
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, bbox[3] - 1],
                        [0, 0],
                        [bbox[2] - 1, 0],
                        [bbox[2] - 1, bbox[3] - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (bbox[2], bbox[3]))

    return warped


@click.command()
@click.option('-a', '--annotations-path',
              type=click.Path(exists=True,
                              file_okay=False,
                              readable=True,
                              path_type=Path),
              default=Path('dataset_info/'))
@click.option('-s', '--save_path',
              type=click.Path(file_okay=False,
                              writable=True,
                              path_type=Path),
              default=Path('cropped/'))
@click.option('--no-split', is_flag=True)
@click.option('--reduce', type=float)
@logger.catch
def main(annotations_path: Path, save_path: Path, no_split: bool, reduce: float):
    annotations: list[Path]
    outputs: list[Path]
    if no_split:
        annotations = [annotations_path / 'imgur5k_annotations.json']
        outputs = [save_path / 'whole']
    else:
        annotations = [
            annotations_path / 'imgur5k_annotations_train.json',
            annotations_path / 'imgur5k_annotations_val.json',
            annotations_path / 'imgur5k_annotations_test.json',
        ]
        outputs = [
            save_path / 'train',
            save_path / 'val',
            save_path / 'test'
        ]

    annotations_path.mkdir(parents=True, exist_ok=True)
    for annotation_path, output_path in tqdm(zip(annotations, outputs)):
        words = {}
        output_path.mkdir(parents=True, exist_ok=True)
        annotation = json.load(annotation_path.open('r'))
        annotations = list(annotation['index_to_ann_map'].items())
        if reduce is not None:
            random.shuffle(annotations)
            annotations = annotations[:int(len(annotations) * reduce)]
        for index_id, ann_ids in tqdm(annotations, leave=False):
            img_info = annotation['index_id'][index_id]
            img = cv2.imread(img_info['image_path'])
            if img is None:
                continue
            for ann_id in ann_ids:
                info = annotation['ann_id'][ann_id]
                info['word'] = str(info['word'])
                if len(info['word']) == 0:
                    continue

                words[ann_id] = info['word']

                if (output_path / f'{ann_id}.png').exists():
                    continue
                print(*info['bounding_box'])
                #print(type(*info))
                bbox = [float(val) for val in info['bounding_box'].strip('[ ]').split(', ')]
                img_cropped = crop_minAreaRect(img, bbox)
                cv2.imwrite(str(output_path / f'{ann_id}.png'), img_cropped)
        json.dump(words, (output_path / 'words.json').open('w'))


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter