
import os, sys
from pathlib import Path
import supervisely as sly


my_app = sly.AppService()
api: sly.Api = my_app.public_api

root_source_dir = str(Path(sys.argv[0]).parents[1])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

TASK_ID = int(os.environ["TASK_ID"])
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])

logger = sly.logger

sample_percent = int(os.environ["modal.state.samplePercent"])

project_name = 'Alstroemeria'
dataset_name = 'ds'
work_dir = 'alstroemeria_data'
alstroemeria_url = 'https://storage.googleapis.com/kaggle-data-sets/1363500/2292905/bundle/archive.zip?X-Goog-Algorithm=' \
                   'GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220223%2' \
                   'Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220223T134003Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=' \
                   'host&X-Goog-Signature=2ac0233217501616a0248af534c741c697c26d832e038ce3b14713eec8ce45475a502e73cc2a7e91b609' \
                   'ab2d33a5716f7a33b1fca617e216a81e6809825a2623298077154a872880c911a837136100e9b2da821155987d1a84bb3e376250d9' \
                   'c3ebef0d3631e7a2ad589b2d5ea1a948fb8eefb5b8907d0d09d53bd1ea41528d495fb13f66205b04d5e7749e11f73287b32b35bc95' \
                   'fc757385e3a1648e38e1f9463f2b1a2d048c060a60175b18a0bab070bb3d11b134bdedf53d2627a028e3ad34a3753eea8a6423da604' \
                   'f3d820ff860867ba9dd42c6e80560b7b853b062bb140d5cbda2da8cfaebc6b5d107b923808f0d1cbee5f8c1d83bb82205bd12fbeebc26'

arch_name = 'archive.zip'
images_folder = 'images'
anns_folder = 'annotations'
class_name = 'alstroemeria'
ann_ext = '.csv'
images_shape = (4032, 3024)

batch_size = 30

obj_class = sly.ObjClass(class_name, sly.Rectangle)
obj_class_collection = sly.ObjClassCollection([obj_class])

meta = sly.ProjectMeta(obj_classes=obj_class_collection)

storage_dir = my_app.data_dir
work_dir_path = os.path.join(storage_dir, work_dir)
sly.io.fs.mkdir(work_dir_path)
archive_path = os.path.join(work_dir_path, arch_name)
