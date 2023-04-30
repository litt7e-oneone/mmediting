import os

def splitext(file_name):
    """
    Separate the file name and suffix.
    Args:
        file_name (str): file path
    Returns:
        name (str): file path without extanded name
        suffix (str): extanded name
        e.g. convert 'xx/xx/A.png' to 'xx/xx/A' and '.png'.
    """
    assert isinstance(file_name, str), 'input type must be str.'
    name = file_name
    suffix = ''
    for i in range(len(file_name)-1, -1, -1):
        c = file_name[i]
        if c == '.':
            name = file_name[:i]
            suffix = file_name[i:]
            break
    return name, suffix


def get_file_list(source_dir, file_suffix=['.bmp','.BMP','.jpg','.png'], sub_dir=False, only_name=False, all_files=False):
    """
    Gets the path to all files in the root directory with the file_suffix
    Args:

    Returns:
        list ['xx/xx/xx.xx']
    """
    if not (os.path.isdir(source_dir)):
        print(source_dir)
        raise ValueError('The input parameter must be a directory or folder')
    if not (sub_dir == True or sub_dir == False):
        print(sub_dir)
        raise ValueError('The input parameter can only be True or False')
    if not (only_name == True or only_name == False):
        print(only_name)
        raise ValueError('The input parameter can only be True or False')

    if isinstance(file_suffix, str):
        file_suffix = [file_suffix]
    ret = []
    # including all sub-directories
    if sub_dir:
        for root, dirs, files in os.walk(source_dir):
            for name in files:
                if all_files or splitext(name)[-1] in file_suffix:
                    if only_name:
                        ret.append(name)
                    else:
                        ret.append(os.path.join(root, name))

    # not including sub-directories
    else:
        names = os.listdir(source_dir)
        for name in names:
            if  all_files or splitext(name)[-1] in file_suffix:
                if only_name:
                        ret.append(name)
                else:
                    ret.append(os.path.join(source_dir, name))

    if len(ret) == 0:
        print('There is no', file_suffix, 'file in dir', source_dir)
    
    return ret


def rename(source_dir):
    os.rename(source_path, target_path)
    return

if __name__ == '__main__':
    rename()