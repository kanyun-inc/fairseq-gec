from collections import defaultdict
import re

#################################  CLASSES FOR DATASETS CONLL FORMAT ##############################
class CoNLL05_Test_Token():
    def __init__(self, raw_line, word_ix):
        info = raw_line.split()
        self.id = word_ix
        self.position = word_ix #0-based
        self.word = info[0]
        self.pos_tag = info[1]
        self.constituent = info[2]
        self.NE = info[3]
        self.pred_lemma = info[4]
        self.pred_sense = "-"
        if self.pred_lemma != "-":
            self.is_pred = True
        else:
            self.is_pred = False
        if len(info) > 4:
            self.labels = info[5:]
        else:
            self.labels = []


class CoNLL05_Token():
    def __init__(self, raw_line, word_ix):
        info = raw_line.split()
        # ['The', 'DT', '(S1(S(NP(NP*', '*', '-', '-', '(A1*']
        self.id = word_ix
        self.position = word_ix #0-based
        self.word = info[0]
        self.pos_tag = info[1]
        self.constituent = info[2]
        self.NE = info[3]
        if info[4] != "-":
            self.pred_sense = info[5] + "." + info[4]
            self.is_pred = True
        else:
            self.pred_sense = "-"
            self.is_pred = False
        if len(info) > 5:
            self.labels = info[6:]
        else:
            self.labels = []


class CoNLL09_Token():
    def __init__(self, raw_line, word_ix):
        info = raw_line.split()
        # print(info)
        # # ['1', 'Frau', 'Frau', 'Frau', 'NN', 'NN', '_', 'nom|sg|fem', '5', '5', 'CJ', 'CJ', '_', '_', 'AM-DIS', '_']
        self.id = int(info[0]) # 1-based ID as in the CoNLL file
        self.position = word_ix # 0-based position in sentence
        self.word = info[1]
        self.lemma = info[2]
        self.pos_tag = info[4]
        self.head = info[8]
        self.dep_tag = info[10]
        self.is_pred = True if info[12] == "Y" else False
        if self.is_pred:
            self.pred_sense = info[13].strip("[]")
            self.pred_sense_id = str(self.position) + "##" + self.pred_sense
        else:
            self.pred_sense = None
            self.pred_sense_id = ""
        if len(info) > 14:
            self.labels = info[14:]
        else:
            self.labels = []


class CoNLL09_FrenchTestToken():
    def __init__(self, raw_line, word_ix):
        info = raw_line.split()
        # print(info)
        # # ['1', 'Frau', 'Frau', 'Frau', 'NN', 'NN', '_', 'nom|sg|fem', '5', '5', 'CJ', 'CJ', '_', '_', 'AM-DIS', '_']
        self.id = int(info[0]) # 1-based ID as in the CoNLL file
        self.position = word_ix # 0-based position in sentence
        self.word = info[1]
        self.lemma = info[2]
        self.pos_tag = info[4]
        self.head = info[8]
        self.dep_tag = info[10]
        self.is_pred = False if info[13] == "_" else True
        if self.is_pred:
            self.pred_sense = info[13]
            self.pred_sense_id = str(self.position) + "##" + self.pred_sense
        else:
            self.pred_sense = None
            self.pred_sense_id = ""
        if len(info) > 14:
            self.labels = info[14:]
        else:
            self.labels = []


#################  CLASSES FOR HANDLNG PREDICATE-ARGUMENT STRUCTURE ###############################
class ArgumentSpan(): # Used in CoNLL-05 CoNLL-12 Annotations
    def __init__(self, init_pos, end_pos, tag, span, predicate_id, predicate_sense):
        self.position = (init_pos, end_pos)
        self.tag = tag
        self.span = span # list of Tokens
        self.head_word = None # could find head with heuristics later...
        self.belongs_to_pred = predicate_id # Sense of the predicate
        self.parent_pred = predicate_sense # String

    def get(self):
        return self.parent_pred, self.tag, self.head_word

    def show(self):
        return "[{} : {}. HEAD = {}]".format(self.tag, self.span, self.head_word)


class ArgumentHead(): # Used in CoNLL-09 Annotations
    def __init__(self, position, tag, word, predicate_id, parent_pred):
        self.position = position
        self.tag = tag
        self.head_word = word
        self.belongs_to_pred = predicate_id # Id in the sentence
        self.parent_pred = parent_pred # Token

    def get(self):
        return self.parent_pred, self.tag, self.head_word

    def show(self):
        return "[{} : {}]".format(self.tag, self.head_word)



################################# GETTING SENTENCE ANNOTATIONS ####################################
class AnnotatedSentence():
    def __init__(self):
        self.tokens = []
        self.only_senses = []
        self.predicates = []
        self.argument_structure = {}
        self.op_lbl_05 = re.compile("\(([A-Z]+(-)*[A-Z]*(-)*[A-Z]*[0-9]*)", re.IGNORECASE)
        self.cl_lbl_05 = re.compile("\*\)")
        self.BIO_sequences = {}
        self.predicates_global_seq = []
        self.predicates_sequences = {}
        self.predicate_indices = []
        self.nominal_predicates = []

    def _make_head_spans(self, pred_arg):
        # Find positions of predicates inside the sentence
        global_pred_seq = ["O"]*len(self.tokens)
        pred2ix = []
        for tok in self.tokens:
            pred_seq = ["O"]*len(self.tokens)
            if tok.is_pred:
                pred_seq[tok.position] = "V"
                global_pred_seq[tok.position] = "V"
                self.predicate_indices.append(tok.position)
                self.predicates_sequences[tok.pred_sense_id] = pred_seq
                pred2ix.append((tok.pred_sense_id, tok.pred_sense, tok.position))
        # Create Argument-Head BIO sequences
        for (tok_pos, word, pred_sense, tok_tag), args in pred_arg.items():
            BIO_seq = ["O"]*len(self.tokens)
            if "V" in tok_tag:
                BIO_seq[tok_pos] = "B-V"
                for arg_head in args:
                    BIO_seq[arg_head.position] = "B-" + arg_head.tag
                self.BIO_sequences[(tok_pos, pred_sense)] = BIO_seq
            else:
                BIO_seq[tok_pos] = "B-N-V"
                for arg_head in args:
                    if arg_head.position != tok_pos:
                        BIO_seq[arg_head.position] = "B-" + arg_head.tag
                    else:
                        BIO_seq[arg_head.position] = "B-" + arg_head.tag + "-N-V"
                self.BIO_sequences[(tok_pos, pred_sense)] = BIO_seq
        self.predicates_global_seq = global_pred_seq #Just one seq for all predicates
        self.argument_structure = pred_arg

    # Works for Span-Based Annotations (CoNLL05 and CoNLL05_Test)
    def _unify_spans(self, group_labels, fix_amadv=False, fix_cv=False):
        pred_seq = ["O"]*len(self.tokens)
        pred_arg = defaultdict(list)
        for (pred_ix, pred_sense), args in group_labels.items():
            all_spans, span, BIO = [], [], []
            init_span, end_span = 0, 0
            for wix, lbl, word in args:
                if self.op_lbl_05.match(lbl) and ")" not in lbl: #Opening Label
                    span.append(word)
                    curr_lbl = lbl.strip("(*")
                    BIO.append("B-"+curr_lbl)
                    init_span = wix
                elif self.cl_lbl_05.match(lbl): #Closing labels
                    span.append(word)
                    end_span = wix
                    all_spans.append(ArgumentSpan(init_span, end_span, curr_lbl, span, pred_ix, pred_sense))
                    pred_arg[pred_sense].append(ArgumentSpan(init_span, end_span, curr_lbl, span, pred_ix, pred_sense))
                    BIO.append("I-"+curr_lbl)
                    span = []
                elif "V" in lbl: # Predicate
                    if fix_cv and "C-V" in lbl:
                        BIO.append("B-C-V")
                    elif fix_amadv and "ADV" in lbl:
                        all_spans.append(ArgumentSpan(wix, wix, lbl, [word], pred_ix, pred_sense))
                        pred_arg[pred_sense].append(ArgumentSpan(wix, wix, lbl, [word], pred_ix, pred_sense))
                        BIO.append("B-"+lbl.strip("(*)"))
                        continue
                    else:
                        BIO.append("B-V")
                    pred_seq[wix] = "V"
                    self.predicate_indices.append(wix)
                elif lbl == "*" and len(span) > 0: # Interior of arg
                    span.append(word)
                    BIO.append("I-"+curr_lbl)
                elif lbl == "*" and len(span) == 0: # Outside Word
                    BIO.append("O")
                    span = []
                else: #Single Word Argument05
                    all_spans.append(ArgumentSpan(wix, wix, lbl, [word], pred_ix, pred_sense))
                    pred_arg[pred_sense].append(ArgumentSpan(wix, wix, lbl, [word], pred_ix, pred_sense))
                    BIO.append("B-"+lbl.strip("(*)"))
            self.BIO_sequences[(pred_ix, pred_sense)] = BIO
            self.predicates_sequences[pred_sense] = pred_seq
        self.predicates_global_seq = pred_seq #Just one seq for all predicates
        self.argument_structure = pred_arg

    def annotate_pred_arg_struct(self, only_senses, fix_cv=False, fix_amadv=False):
        self.predicates_global_seq = ["O"]*len(self.tokens)
        my_preds = self.only_senses if only_senses else self.predicates
        pred_arg = {my_preds[pred_ix]: [] for pred_ix in range(len(my_preds))}
        if len(my_preds) == 0: return None
        if isinstance(self.tokens[0], CoNLL09_Token) or isinstance(self.tokens[0], CoNLL09_FrenchTestToken):
            for tok in self.tokens:
                # pred_ix is from 0 to len(preds)
                for pred_ix, lbl in enumerate(tok.labels):
                    if lbl != "_":
                        pred_arg[my_preds[pred_ix]].append(ArgumentHead(tok.position, lbl, tok.word, pred_ix, self.predicates[pred_ix]))
            self._make_head_spans(pred_arg)
        elif isinstance(self.tokens[0], CoNLL05_Token) or isinstance(self.tokens[0], CoNLL05_Test_Token):
            group_labels = defaultdict(list)
            for tok in self.tokens:
                for pred_ix, lbl in enumerate(tok.labels):
                    group_labels[(pred_ix, my_preds[pred_ix])].append((tok.position, lbl, tok.word))
            self._unify_spans(group_labels, fix_amadv=fix_amadv, fix_cv=fix_cv)

    def get_words(self):
        return [tok.word for tok in self.tokens]

    def get_sentence(self):
        return " ".join([tok.word for tok in self.tokens])

    def show_pred_args(self):
        for predicate, arguments in self.argument_structure.items():
            print("{} --> {}".format(predicate, [arg.show() for arg in arguments]))


def get_annotation(raw_lines, token_class, **kwargs):
    ann = AnnotatedSentence()
    # Annotate the predicates and senses
    for i, line in enumerate(raw_lines):
        tok = token_class(line, i)
        if tok:
            ann.tokens.append(tok)
            if tok.is_pred:
                ann.predicates.append((tok.position, tok.word, tok.pred_sense, tok.pos_tag))
                ann.only_senses.append(tok.pred_sense)
    # print(ann.predicates)
    # print(ann.get_sentence())
    # Annotate the arguments of the corresponding predicates
    ann.annotate_pred_arg_struct(only_senses=False, fix_cv=kwargs.get('fix_cv', False), fix_amadv=kwargs.get('fix_amadv', False))
    return ann


def read_conll(filename, conll_token, args):
    f = open(filename)
    n_sents = 0
    annotated_sentences, buffer_lst = [], []
    for i, line in enumerate(f.readlines()):
        line = line.rstrip()
        if len(line) > 5:
            buffer_lst.append(line)
        else:
            ann = get_annotation(buffer_lst, conll_token, fix_cv=args.cv, fix_amadv=args.amadv)
            n_sents += 1
            buffer_lst = []
            annotated_sentences.append(ann)
    if len(buffer_lst) > 0:
        annotated_sentences.append(get_annotation(buffer_lst, conll_token, fix_cv=args.cv, fix_amadv=args.amadv))
    # print("Read {} Sentences!".format(n_sents))
    return annotated_sentences
