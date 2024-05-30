from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


if __name__ == '__main__':
    base = '/home/isensee/Downloads/AbdomenAtlas1.0Mini'
    cases = subdirs(base, join=False, prefix='BDMAP')

    target_dataset_id = 224
    target_dataset_name = f'Dataset{target_dataset_id:3.0f}_AbdomenAtlas1.0'

    raw_dir = '/home/isensee/drives/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/2024_JHU_benchmark'
    maybe_mkdir_p(join(raw_dir, target_dataset_name))
    imagesTr = join(raw_dir, target_dataset_name, 'imagesTr')
    labelsTr = join(raw_dir, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    for case in cases:
        shutil.copy(join(base, case, 'ct.nii.gz'), join(imagesTr, case + '_0000.nii.gz'))
        shutil.copy(join(base, case, 'combined_labels.nii.gz'), join(labelsTr, case + '.nii.gz'))

    labels = {
        "background": 0,
        "aorta": 1,
        "gall_bladder": 2,
        "kidney_left": 3,
        "kidney_right": 4,
        "liver": 5,
        "pancreas": 6,
        "postcava": 7,
        "spleen": 8,
        "stomach": 9
    }

    generate_dataset_json(
        join(raw_dir, target_dataset_name),
        {0: 'nonCT'},  # this was a mistake we did at the beginning and we keep it like that here for consistency
        labels,
        len(cases),
        '.nii.gz',
        None,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient'
    )