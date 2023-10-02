import supervisely as sly
import os
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, get_file_size
import shutil
import json
from tqdm import tqdm
from glob import glob
import imagesize
import csv


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def create_ann(image_path):
    labels = []
    img_tags = []

    bbox_cords = ann_to_dict.get(os.path.basename(image_path))
    for bbox in bbox_cords:
        x_min, y_min, w, h = bbox
        left = int(x_min)
        right = int(x_min + w)
        top = int(y_min)
        bottom = int(y_min + h)
        rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
        obj_class = str_to_class.get(os.path.basename(os.path.dirname(os.path.dirname(image_path))))
        label = sly.Label(rectangle, obj_class)
        labels.append(label)
    ses_name = os.path.basename(os.path.dirname(image_path))
    tags = s_name_to_tags_dict.get(ses_name)

    for idx, tagmeta in enumerate(tm_list):
        tag = sly.Tag(tagmeta, tags[idx])
        img_tags.append(tag)

    img_width, img_height = imagesize.get(image_path)
    return sly.Annotation(img_size=(img_height, img_width), labels=labels, img_tags=img_tags)


obj_class_maize = sly.ObjClass("maize", sly.Rectangle, color=[247, 1, 142])
obj_class_sugarbeet = sly.ObjClass("sugarbeet", sly.Rectangle, color=[3, 194, 185])
obj_class_sunflower = sly.ObjClass("sunflower", sly.Rectangle, color=[201, 3, 27])

str_to_class = {
    "maize": obj_class_maize,
    "sugarbeet": obj_class_sugarbeet,
    "sunflower": obj_class_sunflower,
}

obj_class_list = [obj_class_maize, obj_class_sugarbeet, obj_class_sunflower]

tm_crop = sly.TagMeta("crop", sly.TagValueType.ANY_STRING)
tm_site = sly.TagMeta("site", sly.TagValueType.ANY_STRING)
tm_year = sly.TagMeta("year", sly.TagValueType.ANY_NUMBER)
tm_fid = sly.TagMeta("flight_id", sly.TagValueType.ANY_NUMBER)
tm_mpnum = sly.TagMeta("microplot_number", sly.TagValueType.ANY_NUMBER)
tm_pnum = sly.TagMeta("plant_number", sly.TagValueType.ANY_NUMBER)
tm_owner = sly.TagMeta("owner", sly.TagValueType.ANY_STRING)
tm_aqdate = sly.TagMeta("acquisition_date", sly.TagValueType.ANY_STRING)
tm_sowdate = sly.TagMeta("sowing_date", sly.TagValueType.ANY_STRING)
tm_lat_long = sly.TagMeta("lat_long", sly.TagValueType.ANY_STRING)
tm_row_num = sly.TagMeta("row_number", sly.TagValueType.ANY_NUMBER)
tm_row_size = sly.TagMeta("row_size(m)", sly.TagValueType.ANY_NUMBER)
tm_row_space = sly.TagMeta("row_space(cm)", sly.TagValueType.ANY_NUMBER)
tm_plant_distance = sly.TagMeta("plant_distance(cm)", sly.TagValueType.ANY_NUMBER)

tm_list = [
    tm_crop,
    tm_site,
    tm_year,
    tm_fid,
    tm_mpnum,
    tm_pnum,
    tm_owner,
    tm_aqdate,
    tm_sowdate,
    tm_lat_long,
    tm_row_num,
    tm_row_size,
    tm_row_space,
    tm_plant_distance,
]


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    global ann_to_dict
    global s_name_to_tags_dict
    project = api.project.create(workspace_id, project_name)
    meta = sly.ProjectMeta(
        obj_classes=obj_class_list,
        tag_metas=tm_list,
    )
    api.project.update_meta(project.id, meta.to_json())

    dataset_path = "/mnt/c/users/german/documents/uavmulticrop"
    dataset = api.dataset.create(project.id, "ds0", change_name_if_conflict=True)

    images_pathes = glob(dataset_path + "/*/*/*.png")

    ann_pathes = glob(dataset_path + "/*/*.json")
    id_to_filename = {}
    keys = []
    values = []
    for ann in ann_pathes:
        with open(
            ann,
            "r",
        ) as JSON:
            json_dict = json.load(JSON)
        for z in json_dict["images"]:
            id_val = z["id"]
            filename_val = z["file_name"]
            id_to_filename[id_val] = [filename_val]
        for x in json_dict["annotations"]:
            s_key = id_to_filename.get(x["image_id"])
            keys.append(s_key[0])
            values.append(x["bbox"])

    ann_to_dict = {}
    for k, v in zip(keys, values):
        if k in ann_to_dict:
            ann_to_dict[k].append(v)
        else:
            ann_to_dict[k] = [v]

    s_name_to_tags_dict = {}
    with open(os.path.join(dataset_path, "table_handcraftpaper.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            if row[0] == "":
                continue
            row = [val if val != "" else "n.a" for val in row]
            (
                crop,
                site,
                crop1,
                session_name,
                year,
                flight_id,
                mic_num,
                plant_num,
                owner,
                aq_date,
                sow_date,
                lat,
                long,
                row_num,
                row_size,
                row_space,
                plant_dis,
                comms,
                bs_line1,
                bs_line2,
            ) = row
            s_name_to_tags_dict[session_name] = [
                crop1,
                site,
                float(year),
                float(flight_id),
                float(mic_num),
                float(plant_num),
                owner,
                aq_date,
                sow_date,
                (lat + ", " + long),
                float(row_num),
                float(row_size.replace(",", ".")),
                float(row_space.replace(",", ".")),
                float(plant_dis.replace(",", ".")),
            ]

    batch_size = 33
    progress = sly.Progress("Create dataset {}".format("ds0"), len(images_pathes))
    for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
        img_names_batch = [sly.fs.get_file_name_with_ext(im_path) for im_path in img_pathes_batch]

        img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns = [create_ann(image_path) for image_path in img_pathes_batch]
        api.annotation.upload_anns(img_ids, anns)
    return project
