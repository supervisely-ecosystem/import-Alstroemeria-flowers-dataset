
import zipfile, os, random, csv
import supervisely as sly
import sly_globals as g
from supervisely.io.fs import download, get_file_name, get_file_name_with_ext
import numpy as np
import gdown
from cv2 import connectedComponents


def read_csv(file_path):
    bboxes = []
    with open(file_path, newline='') as File:
        reader = csv.reader(File)
        for row in reader:
            bboxes.append(list(map(lambda x: round(float(x)), row[:4])))

    return bboxes


def create_ann(ann_path):
    labels = []
    ann_data = read_csv(ann_path)
    for curr_bbox in ann_data:
        rect = sly.Rectangle(curr_bbox[1], curr_bbox[0], curr_bbox[3], curr_bbox[2])
        label = sly.Label(rect, g.obj_class)
        labels.append(label)

    return sly.Annotation(img_size=g.images_shape, labels=labels)


def extract_zip():
    if zipfile.is_zipfile(g.archive_path):
        with zipfile.ZipFile(g.archive_path, 'r') as archive:
            archive.extractall(g.work_dir_path)
    else:
        g.logger.warn('Archive cannot be unpacked {}'.format(g.arch_name))
        g.my_app.stop()


@g.my_app.callback("import_alstroemeria")
@sly.timeit
def import_alstroemeria(api: sly.Api, task_id, context, state, app_logger):

    gdown.download(g.alstroemeria_url, g.archive_path, quiet=False)
    app_logger.warn('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', os.listdir(g.work_dir_path))
    extract_zip()

    alstroemeria_data_path = os.path.join(g.work_dir_path, sly.io.fs.get_file_name(g.arch_name))

    img_path = os.path.join(alstroemeria_data_path, g.images_folder)
    ann_path = os.path.join(alstroemeria_data_path, g.anns_folder)

    new_project = api.project.create(g.WORKSPACE_ID, g.project_name, change_name_if_conflict=True)
    api.project.update_meta(new_project.id, g.meta.to_json())

    new_dataset = api.dataset.create(new_project.id, g.dataset_name, change_name_if_conflict=True)

    sample_img_path = random.sample(os.listdir(img_path), g.sample_percent)

    progress = sly.Progress('Create dataset {}'.format(g.dataset_name), len(sample_img_path), app_logger)
    for img_batch in sly.batched(sample_img_path, batch_size=g.batch_size):

        img_pathes = [os.path.join(img_path, name) for name in img_batch]
        ann_pathes = [os.path.join(ann_path, get_file_name(name) + g.ann_ext) for name in img_batch]

        anns = [create_ann(ann_path) for ann_path in ann_pathes]

        img_infos = api.image.upload_paths(new_dataset.id, img_batch, img_pathes)
        img_ids = [im_info.id for im_info in img_infos]
        api.annotation.upload_anns(img_ids, anns)
        progress.iters_done_report(len(img_batch))

    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": g.TEAM_ID,
        "WORKSPACE_ID": g.WORKSPACE_ID
    })
    g.my_app.run(initial_events=[{"command": "import_alstroemeria"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)