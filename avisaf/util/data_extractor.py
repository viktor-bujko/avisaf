#!/usr/bin/env python3

import pandas as pd
from sys import stderr
import json
import sys
from pathlib import Path

SOURCES_ROOT_PATH = Path(sys.argv[0]).parent.parent.resolve()
PROJECT_ROOT_PATH = SOURCES_ROOT_PATH.parent.resolve()
sys.path.append(str(SOURCES_ROOT_PATH))


def get_entities(entities_file_path: Path = Path(PROJECT_ROOT_PATH, 'data_files', 'entities_labels.json').resolve()):
    """

    :type entities_file_path: Path
    :param entities_file_path:
    :return:
    """
    entities_file_path = entities_file_path.resolve()

    with entities_file_path.open(mode='r') as entities_file:
        return json.load(entities_file)


def get_training_data(training_data_file_path: Path):
    """
    :type training_data_file_path: Path
    :param training_data_file_path:
    :return: JSON list of (text, annotations) tuples.
    """

    if not training_data_file_path.is_absolute():
        training_data_file_path = training_data_file_path.resolve()

    with training_data_file_path.open(mode='r') as tr_data_file:
        return json.load(tr_data_file)


'''def _choose_file():
    """

    :return:
    """
    tkinter.Tk()
    file_path = filedialog.askopenfilename(initialdir='~',
                                           title='Select a file',
                                           filetypes=[('csv files', '*.csv')])

    return file_path'''


def get_narratives(lines: int = -1, file_path: Path = None, start_index: int = 0):
    """
    :type lines:        int
    :param lines:
    :type file_path:    Path
    :param file_path:
    :type start_index:  int
    :param start_index:
    :return: Returns a generator of all texts.
    """

    '''if file_path is None:
        file_path = _choose_file()'''

    file_path = str(file_path) if file_path.is_absolute() else str(file_path.resolve())

    report_df = pd.read_csv(file_path, skip_blank_lines=True, index_col=0, header=[0, 1])
    report_df.columns = report_df.columns.map('_'.join)

    try:
        narratives1 = report_df['Report 1_Narrative'].values.tolist()
        calls1 = report_df['Report 1_Callback'].values.tolist()
        narratives2 = report_df['Report 2_Narrative'].values.tolist()
        calls2 = report_df['Report 2_Callback'].values.tolist()

    except KeyError:
        print('No such key was found', file=stderr)
        return None

    length = len(narratives1)
    lists = [narratives1, calls1, narratives2, calls2]

    # assert all(len(lst) == length for lst in lists)
    end_index = start_index + lines
    for index in range(start_index, length):
        if lines != -1 and index >= end_index:
            break
        res = ' '.join([str(lst[index]) for lst in lists if str(lst[index]) != 'nan'])
        yield res


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    # target = sys.argv[2]
    texts = list(get_narratives(file_path=Path(path)))

    for text in texts:
        print(f'"{text}",')

    # print(len(texts))

    # with open(target, mode='x') as file:
    #    file.write(*texts, sep='\n')
