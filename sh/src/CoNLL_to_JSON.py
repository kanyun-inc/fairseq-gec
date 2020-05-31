"""
    RUN EXAMPLE:
        python pre_processing/CoNLL_to_JSON.py --path datasets/raw/ \
            -s mini_europarl-v7.de-en.en -t mini_europarl-v7.de-en.de -o datasets/json/MiniEuroparl.en_to_de.json \
            -ks "<EN>" -kt "<DE>"
"""

from CoNLL_Annotations import CoNLL05_Token, CoNLL05_Test_Token, CoNLL09_Token, CoNLL09_FrenchTestToken, read_conll
from flair.data import Sentence
from flair.models import SequenceTagger
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import json, io
import argparse


def get_token_type(type_str):
    if type_str =="CoNLL05":
        return CoNLL05_Token
    elif type_str == "CoNLL05_Test":
        return CoNLL05_Test_Token
    elif type_str == "CoNLL09_FrenchTest":
        return CoNLL09_FrenchTestToken
    else:
        return CoNLL09_Token


def make_one_pred_per_sent(predicate_json):
    all_preds = []
    with open(predicate_json) as f:
        for line in f:
            obj = json.loads(line)
            all_preds.append(obj["predicates"])
    return all_preds


def get_lang(lang):
    if lang =="<EN>":
        return spacy.load("en")
    elif lang == "<DE>":
        return spacy.load("de")
    elif lang == "<FR>":
        return spacy.load("fr")
    else:
        return None


def tokenize_postag_sentence(sentence_str, spacy_lang=None):
    if not spacy_lang:
        return sentence_str.split(), []
    else:
        toks, pos = zip(*[(tok.text, tok.pos_) for tok in spacy_lang(sentence_str)])
        return toks, pos


def get_source_frames(sentence_tokens, frame_tagger):
    all_frames = []
    if frame_tagger:
        sentence_obj = Sentence(" ".join(sentence_tokens))
        frame_tagger.predict(sentence_obj)
        for frame in sentence_obj.get_spans('frame'):
            if frame.tag != "_":
                indices, tokens = zip(*[(tok.idx - 1, tok.text) for tok in frame.tokens])
                all_frames.append({"predicate_sense": frame.tag, "predicate_word": tokens, "predicate_ix": indices})
    return all_frames


def get_aligned_pred(src_sentence, src_pos, src_preds, pred_tgt):
    def get_sentence_lemmas(src_sentence):
        lemmasent = []
        for w in src_sentence:
            lemmas = lemmatizer(w, u"VERB")
            lemmasent.append(lemmas)
        return lemmasent

    def default_pred():
        sense_lemmas = lemmatizer(pred_tgt[2].split(".")[0], u"VERB")
        sentence_lemmas = get_sentence_lemmas(src_sentence)
        for i, tup in enumerate(sentence_lemmas):
            for wl in tup:
                if wl in sense_lemmas and src_pos[i] == u"VERB":
                    return i, wl, "<UNK>", "V"
        return -1, "-", "<NO-PRED>", "-"

    def choose_pred():
        for sp in src_preds:
            if sp["predicate_sense"] == pred_tgt[2]:
                return sp["predicate_ix"][-1], sp["predicate_word"][-1], sp["predicate_sense"], "V"
        return -1, "-", "<NO-PRED>", "-"

    matched_pred = choose_pred()
    if matched_pred[0] == -1 and len(src_pos) > 0: matched_pred = default_pred()
    src_pred_ix = matched_pred[0]
    IOB_src = ["O" for x in src_sentence]
    if src_pred_ix > 0: IOB_src[src_pred_ix] = "B-V"
    return IOB_src, matched_pred


def get_txt_sentences(filename):
    sents = []
    for line in open(filename).readlines():
        sents.append(line.strip("\n"))
    return sents


def BIO_to_Sequence(word_list, tag_list):
    tagged_tokens = []
    open_tag, close_tag = "", ""
    for word, tag in zip(word_list, tag_list):
        if "B-" in tag:
            if close_tag == "":
                tagged_tokens.append("(#")
                close_tag = tag[2:] + ")"
                tagged_tokens.append(word)
            else:
                tagged_tokens.append(close_tag)
                tagged_tokens.append("(#")
                tagged_tokens.append(word)
                close_tag = tag[2:] + ")"
        elif "I-" in tag:
            tagged_tokens.append(word)
        else:
            if close_tag != "":
                tagged_tokens.append(close_tag)
                close_tag = ""
            tagged_tokens.append(word)
    if close_tag != "": tagged_tokens.append(close_tag)
    return tagged_tokens


def mark_sequence(word_list, tag_list):
    marked_seq = []
    try:
        pred_ix = tag_list.index("B-V")
        for i, word in enumerate(word_list):
            if i == pred_ix: marked_seq.append("<PRED>")
            marked_seq.append(word)
        return marked_seq
    except:
        return word_list


def make_mono_files(file_props, args, append_in_file=False):
    print("---------------------\nProcessing {} file...".format(file_props["in"]))
    no_preds, no_verb, written_in_file = 0, 0, 0
    sentences = read_conll(file_props["in"], conll_token=file_props["token_type"], args=args)
    file_mode = "a" if append_in_file else "w"
    json_file = io.open(file_props["out"], file_mode, encoding='utf8')
    for sent in sentences:
        seq_obj = {}
        my_sent = sent.get_words()
        # print(sent.get_sentence() + "\n")
        # sent.show_pred_args()
        per_predicate = list(sorted(sent.BIO_sequences.items(), key=lambda x: x[0][0]))
        if len(per_predicate) > 0:
            for ix, (pred_sense, seq) in enumerate(per_predicate):
                seq_obj["seq_words"] = my_sent
                seq_obj["BIO"] = seq
                seq_obj["pred_sense"] = sent.predicates[ix]
                if "B-V" in seq:
                    seq_obj["seq_marked"] = mark_sequence(my_sent, seq)
                    seq_obj["seq_tag_tokens"] = BIO_to_Sequence(my_sent, seq)
                else:
                    seq_obj["seq_tag_tokens"] = my_sent
                    no_verb += 1
                seq_obj["src_lang"] = file_props["lang"]
                seq_obj["tgt_lang"] = "<" + file_props["lang"][1:-1] + "-SRL>"
                json_file.write(json.dumps(seq_obj) + "\n")
                written_in_file += 1
        else:
            no_preds += 1
            generic_bio = ["O" for x in my_sent]
            seq_obj["seq_words"] = my_sent
            seq_obj["BIO"] = generic_bio
            seq_obj["pred_sense"] = (-1, "-", "<NO-PRED>", "-")
            seq_obj["seq_marked"] = my_sent
            seq_obj["seq_tag_tokens"] = my_sent
            seq_obj["src_lang"] = file_props["lang"]
            seq_obj["tgt_lang"] = "<" + file_props["lang"][1:-1] + "-SRL>"
            json_file.write(json.dumps(seq_obj) + "\n")
            written_in_file += 1
    print("IN: {} --> OUT: {}\nFound {} in CoNLL --> Wrote {} in JSON".format(file_props["in"], file_props["out"],
                                                                              len(sentences), written_in_file))


def make_parallel_files(file_props, args, append_in_file=False):
    # Prepare Output JSON File
    file_mode = "a" if append_in_file else "w"
    json_file = io.open(file_props["output"], file_mode, encoding='utf8')
    # Load Source Frame Tagger (Flair)
    if file_props["src_lang"] == "<EN>":
        flair_frame_tagger = SequenceTagger.load('frame')
    else:
        flair_frame_tagger = None
    # Get proper tokenizer
    tokenize_text = file_props.get("tokenize", False)
    if tokenize_text:
        src_spacy = get_lang(file_props["src_lang"])
    else:
        src_spacy = None
    # Load Src and Tgt sentences
    src_sentences = get_txt_sentences(file_props["src_txt"])
    tgt_sentences = read_conll(file_props["tgt_conll"], conll_token=CoNLL09_Token, args=args)
    assert len(src_sentences) == len(tgt_sentences), "Src len [{}] and Tgt len [{}] don't match!".format(len(src_sentences), len(tgt_sentences))

    # Construct JSON
    written_in_file = 0
    for tgt_ix, tgt_sent in enumerate(tgt_sentences):
        seq_obj = {}
        per_predicate = list(sorted(tgt_sent.BIO_sequences.items(), key=lambda x: x[0][0]))
        for ix, (pred_sense, tgt_bio) in enumerate(per_predicate):
            src_sentence, src_pos = tokenize_postag_sentence(src_sentences[tgt_ix], src_spacy)

            seq_obj["seq_words"] = src_sentence
            if "B-V" in tgt_bio:
                src_frames = get_source_frames(src_sentence, flair_frame_tagger)
                src_bio, src_pred = get_aligned_pred(src_sentence, src_pos, src_frames, tgt_sent.predicates[ix])
                if not "B-V" in src_bio: continue # Skip sentence for which we didn't find a matching source predicate
                seq_obj["BIO"] = src_bio
                seq_obj["pred_sense_origin"] = src_pred
                seq_obj["pred_sense"] = tgt_sent.predicates[ix]
                seq_obj["seq_marked"] = mark_sequence(src_sentence, src_bio)
                tagged_seq = BIO_to_Sequence(tgt_sent.get_words(), tgt_bio)
                seq_obj["seq_tag_tokens"] = tagged_seq
            else:
                generic_bio = ["O" for x in src_sentence]
                seq_obj["BIO"] = generic_bio
                seq_obj["pred_sense_origin"] = (-1, "-", "<NO-PRED>", "-")
                seq_obj["pred_sense"] = (-1, "-", "<NO-PRED>", "-")
                seq_obj["seq_marked"] = src_sentence
                seq_obj["seq_tag_tokens"] = tgt_sent.get_words()
            seq_obj["src_lang"] = file_props["src_lang"]
            seq_obj["tgt_lang"] = file_props["tgt_lang"]
            json_file.write(json.dumps(seq_obj) + "\n")
            written_in_file += 1
    print(" {} --> {}\n TOTAL: {} --> {}".format(file_props["tgt_conll"], file_props["output"], len(tgt_sentences), written_in_file))


if __name__ == "__main__":
    """
    RUN EXAMPLE:
    
        * Monolingual (source and target share the same language) -
            python pre_processing/CoNLL_to_JSON.py \
                --source_file datasets/raw/CoNLL2005-trial.txt \
                --output_file datasets/json/EN_conll05_trial.json \
                --dataset_type mono \
                --src_lang "<EN>" \
                --token_type CoNLL05
    
        * Cross-lingual (Needs a parallel source text file and target CoNLL tagged file) -
            python pre_processing/CoNLL_to_JSON.py \
                --source_file datasets/raw/CrossLang_ENDE_EN_trial.txt \
                --target_file datasets/raw/CrossLang_ENDE_DE_trial.conll09 \
                --output_file datasets/json/En2DeSRL.json \
                --dataset_type cross \
                --src_lang "<EN>" --tgt_lang "<DE-SRL>"
            
                
    """

    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', help='Path to the source text sentences or to the raw source dataset CoNLL file', required=True)
    parser.add_argument('-t', '--target_file', help='Path to the raw target dataset CoNLL file', default=None)
    parser.add_argument('-o', '--output_file', help='Path and filename where the JSON data will be saved', required=True)
    parser.add_argument('-ls', '--src_lang', help="Language of the unlabeled sentences. Options: '<EN>','<DE>','<FR>'", required=True)
    parser.add_argument('-lt', '--tgt_lang', help="Language of the labeled sentences. Options: '<EN-SRL>','<DE-SRL>','<FR-SRL>'", default="<EN>")
    parser.add_argument('-d', '--dataset_type', help="String indicating if it is a 'mono' or 'cross' mode", default="mono")
    parser.add_argument('-tt', '--token_type', help='String that Indicates the format of the CoNLL file. Options: \
                                                    CoNLL05, CoNLL05_Test, CoNLL09, CoNLL09_FrenchTest', default="CoNLL09")
    parser.add_argument('--fix_amadv', dest='amadv', action='store_true')
    parser.add_argument('--fix_cv', dest='cv', action='store_true')
    args = parser.parse_args()

    if args.dataset_type == "mono":
        token_type = get_token_type(args.token_type)
        props = {"in":  args.source_file,
                "out":  args.output_file,
                "lang": args.src_lang,
                "token_type": token_type
                 }
        make_mono_files(props, args)
    else:
        lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
        props = {"src_txt": args.source_file,
                  "tgt_conll": args.target_file,
                  "output": args.output_file,
                  "src_lang": "<EN>",
                  "tgt_lang": "<DE-SRL>",
                  "tokenize": True}
        make_parallel_files(props, args)

print("DONE!")
