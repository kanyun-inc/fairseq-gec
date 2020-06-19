import os
import sys
import json
import numpy as np
import argparse
# import tempfile
from logging import getLogger

from allennlp.models.semantic_role_labeler import convert_bio_tags_to_conll_format

# from log.set_log import set_log
# sys.path.append(os.getcwd())
# logger = set_log("eval.log")
logger = getLogger(__name__)
DEBUG_MATCH = True     # check if the created prop file match with the gold prop file

class PredJsonReader():
    ''' 
    READ    "predicted.json" from model prediction 
    WRITE   "eval.prop" to use srl-eval.pl

    NOTE
    use "gold.prop" 
    - to implement oracle-max eval
    - to escape mismatching "AM-ADV" or "C-V"
    - to write tgt verb as base
    ==============================================
    * saved_models: 
        saved_dir of model associations
    * json_file_from_predictor:
        file of model prediction
    * test:
        identifier of the use of test type (wsj/brown)
    * oracle:
        identifier of the use of oracle type (min/max)
    '''
    def __init__(self,
                 saved_models: str,
                 json_file_from_predictor: str,
                 test: str = 'wsj',
                 oracle: str = 'min',
                 ) -> None:
        
        self._test = test
        self._oracle = oracle
        self._predictor_output = os.path.abspath(os.path.join(saved_models, json_file_from_predictor))
        self._gold_prop = os.path.abspath('datasets/conll05/conll05.test.{}.prop'.format(test))
        self._pred_prop = os.path.abspath(os.path.join(saved_models, 'JSON_to_CoNLL/eval.test.{}.{}.prop'.format(test, oracle)))
        
        sys.path.append(self._predictor_output)
        sys.path.append(self._gold_prop)
        sys.path.append(self._pred_prop)
        

    def _reader(self):
        """
        yield each instance
        ===========================
        line_obj:
            * metadata:             Dict
            * predicted_log_probs:  List[ float ]           #beam
            * predictions:          List[ List(ids) ]       #beam
            * predicted_tokens:     List[ List(tokens) ]    #beam
        ---------------------------
        ## metadata
            * souruce_tokens:               ['<EN-SRL>', 'Some', ..., 'i.'] 
            * verb:                         'installed'
            * src_lang:                     '<EN>'
            * tgt_lang:                     '<EN-SRL>'
            * original_BIO:                 ['B-A1', 'I-A1', ..., 'O']
            * original_predicate_senses:    []
            * predicate_senses:             [5, 'installed', '-', 'VBN']
            * original_target:              ['(#', 'Some', ..., '.']
        """
        count_different_seq_len, count_inappropriate_bracket, count_without_pred = 0, 0, 0
        
        for instance_no, line in enumerate(open(self._predictor_output), start=1):
            line_obj = json.loads(line.strip())
            metadata = line_obj['metadata']
            
            sentence = metadata['source_tokens'][1:]        # exclude '<EN-SRL>'
            verb = tuple(metadata['predicate_senses'][:2])  # (v_idx, verb)
            
            predicted_target = line_obj['predicted_tokens'][0] if not DEBUG_MATCH else \
                               metadata['original_target']
            
            bio_gold = metadata['original_BIO']
            bio_pred, invalid_bracket = self.create_predicted_BIO(predicted_target)
            
            conll_formatted_gold_tag = convert_bio_tags_to_conll_format(bio_gold)
            conll_formatted_predicted_tag = convert_bio_tags_to_conll_format(bio_pred)
            
            ### counter ###
            if len(bio_gold) != len(bio_pred):
                count_different_seq_len += 1
            if invalid_bracket:
                count_inappropriate_bracket += 1
            if verb[0] == -1:
                count_without_pred += 1
            
            yield verb, sentence, conll_formatted_predicted_tag, conll_formatted_gold_tag, line_obj
        
        logger.info("COUNT:")
        logger.info(" - instances: %d" % instance_no)
        logger.info(" - different_seq_len: %d" % count_different_seq_len)
        logger.info(" - inappropriate_bracket: %d" % count_inappropriate_bracket)
        logger.info(" - without_pred: %d" % count_without_pred)

    
    def create_predicted_BIO(self, predicted_target:list) -> list:
        props, prop = [], 'O'
        isInvalid = False
        io_bracket = 0  # 0:out, 1:in, else:invalid 
        IN, OUT = 1, 0
        
        for word in reversed(predicted_target):
            if word.endswith(')') and len(word) > 1:
                if io_bracket == IN:
                    isInvalid = True
                    continue
                io_bracket += 1
                prop = word[:-1]
            elif word == '(#':
                if io_bracket == OUT: 
                    isInvalid = True
                    continue
                if not prop == "O":
                    props[-1] = "B-"+prop
                prop = 'O'
                io_bracket -= 1
            else:
                props.append(prop) if prop == "O" else props.append("I-"+prop)
        
        return list(reversed(props)), isInvalid


    def _write_fprop(self, fo_prop):
        """
        WRITE   output file w/ prop-format (fo_prop)
        """
        prev_sent = ''
        buffer_sent = []

        fo, fr = open(fo_prop, 'w'), open(self._gold_prop, 'r')

        gold_lines = self.read_sent_block(fr)
        isValid = True
        
        for (vi, verb), sent, pred, gold, line_obj in self._reader():

            ### CASE: mismatch the seq-len between pred and gold
            if not len(sent) == len(pred):
                isValid = False
                pred = ['*' for _ in gold]  # consider as incorrect
            
            ### new sentence
            if not prev_sent == sent:
                if buffer_sent:
                    self.write_buffer_sent(fo, buffer_sent, gold_lines, isValid, self._oracle)
                    buffer_sent = []
                    gold_lines = self.read_sent_block(fr)
                
                isValid = True
                buffer_sent = [[verb, pred[i]] if i == vi else ['-', pred[i]] for i in range(len(sent))] \
                              if not vi == -1 else [['-'] for _ in range(len(sent))]      
                
            ### same sentence (but other verb: from the 2nd col) -> add cols
            else:
                for wi, row in enumerate(buffer_sent):
                    row.append(pred[wi])
                    if wi == vi:
                        row[0] = verb # overwrite the target verb
        
            prev_sent = sent
        
        ## end of sentence
        self.write_buffer_sent(fo, buffer_sent, gold_lines, isValid, self._oracle)
            
        fo.close(), fr.close()
        logger.info('write to ... %s' % fo_prop)

    
    def read_sent_block(self, fprop) -> list:
        gold_lines = []
        gold_line = fprop.readline()
        while (len(gold_line.strip()) > 0):
            gold_lines.append(gold_line.split())
            gold_line = fprop.readline()
        return gold_lines
    
    
    def write_buffer_sent(self, fo, buffer_sent, gold_sent, isValid, oracle):
        if isValid:
            for row, gold in zip(buffer_sent, gold_sent):
                output = [g if ("C-V" in g or "AM-ADV" in g) else b for b,g in zip(row[1:], gold[1:])] # from the 2nd col
                output.insert(0, gold[0]) # 1st col (as base)
                fo.write(self.align_format(output))
            fo.write('\n')
        else:
            for wi, (row, line_gold) in enumerate(zip(buffer_sent, gold_sent)):
                output = ['*' if oracle=='min' else g for g in line_gold[1:]]   # from the 2nd col
                output.insert(0, gold_sent[wi][0])  # 1st col (as base)
                fo.write(self.align_format(output))
            fo.write('\n')

    
    def align_format(self, out:list) -> str:
        """
        align between gold.prop to use "diff -q" command
        """
        exception = ["R-AM-TMP", "R-AM-ADV", "R-AM-EXT", "R-AM-CAU", "R-AM-MNR", "R-AM-LOC"] # len(label)
        line = out[0]
        right_overflow = len(out[0]) - 1
        if len(out) > 1:
            for i, token in enumerate(out[1:]):
                prev = out[i]
                space = 23 if i == 0 else 15
                asta = token.rfind('*')
                n = space - asta - right_overflow
                if any(map(lambda x: x in token, exception)): #例外
                    n += 1
                if any(map(lambda x: x in prev, exception)):
                    n -= 1
                right_overflow = len(token) - asta - 1
                line += " "*n + token
            n = 6 - right_overflow
            if any(map(lambda x: x in out[-1], exception)):
                n -= 1
            line += " "*n + "\n"
        else:
            line += " "*14 + "\n"
        return line


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('saved_models', type=str)
    parser.add_argument('--test', default='wsj', type=str, help='wsj/brown')
    parser.add_argument('--oracle', default='min', type=str, help='min/max')
    parser.set_defaults(no_thres=False)
    return parser


if __name__ == '__main__':

    parser = create_arg_parser()
    args = parser.parse_args()

    os.makedirs(os.path.join(args.saved_models, 'JSON_to_CoNLL'), exist_ok=True)

    predicted_json = 'predicted.test.{}.json'.format(args.test)    # from predictor
    
    logger.debug("read from ... %s" % predicted_json)
    logger.debug("test ... %s" % args.test)
    logger.debug("oracle ... %s" % args.oracle)

    reader = PredJsonReader(
            args.saved_models,  # saved_models
            predicted_json,     # json file from predictor
            args.test,          # wsj/brown
            args.oracle,        # min/max
            )

    reader._write_fprop(reader._pred_prop)
    sys.stdout.write('DONE!\n\n')

