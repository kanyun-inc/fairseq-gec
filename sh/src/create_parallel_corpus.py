import os
import sys
import re
import json
import logging
import argparse
import subprocess as subp


logging.basicConfig(
    format='%(asctime)s #L%(lineno)d %(levelname)s --- %(message)s',
    datefmt='%Y-%m-%d %H:%M',
    level=logging.DEBUG,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def create_parallel_corpus(
        fi: os.path.abspath,
        fo: os.path.abspath,
        ) -> None:
    """ line
    {
    'seq_words':    ['The', 'economy', "'s", ... , '.'], 
    'BIO':          ['B-A1', 'I-A1', 'I-A1', ... , 'O'],
    'pred_sense':   [6, 'taken', 'take.01', 'VBN'],
    'seq_marked':   ['The', 'economy', ... ,'will', 'be', '<PRED>', 'taken' , ...]
    'seq_tag_tokens': ['(#', 'The', 'economy', "'s", 'temperature', 'A1)', ... ]
    'src_lang':     '<EN>',
    'tgt_lang':     '<EN-SRL>'
    }
    """
    ### ここで target を作成 ###
    def insert_pred_token(seq_tag_tokens:list) -> list:
        # return ' '.join(line['seq_tag_tokens']) を変更
        try:
            pix = seq_tag_tokens.index('<PRED>')
            seq_tag_tokens.insert(pix+2, '</PRED>')
        except ValueError:
            pix = -1
        return seq_tag_tokens

    def convert_label_tokens(seq_tag_tokens:list) -> list:
        out = []
        label = None
        for rvs_token in seq_tag_tokens[::-1]:
            if rvs_token.endswith(')') and len(rvs_token) > 1:
                if label is not None:
                    raise LabelUnclosedError
                label = rvs_token[:-1]
                out.append(f"</{label}>")
            elif rvs_token.startswith('(#'):
                if label is None:
                    raise LabelUnclosedError
                out.append(f"<{label}>")
                label = None
            else:
                out.append(rvs_token)

        return out[::-1]


    pattern = re.compile('.*?/conll05.(?P<dset>.*?).json')
    name = pattern.search(str(fi)).group('dset')

    logger.debug('READ %s set ... %s' % (name, fi))

    fo_src = open(os.path.join(fo, '{}.src'.format(name)), 'w')
    fo_tgt = open(os.path.join(fo, '{}.tgt'.format(name)), 'w')

    for idx, line in enumerate(open(fi), start=1):
        line = json.loads(line.rstrip())
        src = '<EN-SRL> ' + ' '.join(insert_pred_token(line['seq_marked']))
        tgt = '<DE-SRL> ' + ' '.join(convert_label_tokens(insert_pred_token(line['seq_tag_tokens'])))
        
        fo_src.write(src + '\n')
        fo_tgt.write(tgt + '\n')

    fo_src.close(), fo_tgt.close()
    logger.debug('SIZE of %s set ... %d' % (name, idx))
    logger.info('WRITE source corpus ... %s' % fo_src.name)
    logger.info('WRITE target corpus ... %s' % fo_tgt.name)


def create_arg_parser():
    parser = argparse.ArgumentParser(
            description='Create paralle corpus from json-data after running CoNLL_to_JSON.py')
    parser.add_argument('-i', '--ddir', type=os.path.abspath, 
            default=os.path.abspath('datasets/json'), 
            help='path of data dir after CoNLL_to_JSON')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, 
            default=os.path.abspath('datasets/parallel'), 
            help='output data path of parallel corpus')
    parser.set_defaults(no_thres=False)
    return parser


def run():
    parser = create_arg_parser()
    args = parser.parse_args()

    command = 'ls {}'.format(args.ddir)
    proc = subp.run(command.split(), stdout=subp.PIPE, stderr=subp.PIPE)
    files = proc.stdout.decode('utf8').strip().split('\n')

    os.makedirs(args.outdir, exist_ok=True)

    for fname in files:
        fi = os.path.join(args.ddir, fname)
        create_parallel_corpus(fi, args.outdir)


if __name__ == '__main__':
    """ running example
    $ CoNLL_to_JSON=datasets/json
    $ python scripts/create_parallel_corpus.py \
        --ddir ${CoNLL_to_JSON} \
        --outdir datasets/parallel
    """
    run()
