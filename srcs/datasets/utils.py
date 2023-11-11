import os
import os.path as osp


# =================
# basic functions
# =================
def file_traverse(dir, ext=None):
    """
    traverse all the files and get their paths
    Args:
        dir (str): root dir path
        ext (list[str], optional): included file extensions. Defaults to None, meaning inculding all files.
    """

    data_paths = []
    skip_num = 0
    file_num = 0

    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            img_path = osp.join(dirpath, filename)
            if ext and img_path.split('.')[-1] not in ext:
                print('Skip a file: %s' % (img_path))
                skip_num += 1
            else:
                data_paths.append(img_path)
                file_num += 1
    return sorted(data_paths), file_num, skip_num


def get_file_path(data_dir, ext=None):
    """
    Get file paths for given directory or directories

    Args:
        data_dir (str): root dir path
        ext (list[str], optional): included file extensions. Defaults to None, meaning inculding all files.
    """

    if isinstance(data_dir, str):
        # single dataset
        data_paths, file_num, skip_num = file_traverse(data_dir, ext)
    elif isinstance(data_dir, list):
        # multiple datasets
        data_paths, file_num, skip_num = [], 0, 0
        for data_dir_n in sorted(data_dir):
            data_paths_n, file_num_n, skip_num_n = file_traverse(
                data_dir_n, ext)
            data_paths.extend(data_paths_n)
            file_num += file_num_n
            skip_num += skip_num_n
    else:
        raise ValueError('data dir should be a str or a list of str')

    return sorted(data_paths), file_num, skip_num

