import argparse
from engines.EndToEndTrainer import EndToEndTrainer
from engines.ExtractorTrainer import ExtractorTrainer
from engines.AETrainer import AETrainer


def get_args():
    parser = argparse.ArgumentParser(description='Train the Net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--endtoend', action='store_true', help='EndToEndTrainer')
    parser.add_argument('--extractor', action='store_true', help='ExtractorTrainer')
    parser.add_argument('--ae', action='store_true', help='AETrainer')

    return parser.parse_args()


if __name__ == '__main__':
    opt = get_args()
    if opt.endtoend == True:
        EndToEndTrainer().train()
    elif opt.extractor == True:
        ExtractorTrainer().train()
    elif opt.ae == True:
        AETrainer().train()
    else:
        ValueError('需要传参选择要训练的模型！')