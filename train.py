import argparse
from engines.EndToEndTrainer import EndToEndTrainer
from engines.ExtractorTrainer import ExtractorTrainer


def get_args():
    parser = argparse.ArgumentParser(description='Train the Net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--endtoend', action='store_true', help='EndToEndTrainer')
    parser.add_argument('--extractor', action='store_true', help='ExtractorTrainer')

    return parser.parse_args()


if __name__ == '__main__':
    opt = get_args()
    if opt.endtoend == True:
        EndToEndTrainer().train()
    elif opt.extractor == True:
        ExtractorTrainer().train()
    else:
        ValueError('需要传参选择要训练的模型！')