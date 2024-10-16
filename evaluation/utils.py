import re
import spacy_alignments as tokenizations
import numpy as np


def normalized_text(text):
    """Normalize text by converting it to lowercase, removing punctuations, and collapsing whitespaces."""

    if isinstance(text, list):
        if len(text)==1:
            text = text[0]
    
    text = text.lower()
    text = re.sub(r'\W+', ' ',
                  text).strip()  # remove non-alphanumeric characters and collapse multiple spaces into one space

    return text


def check_fuzzy_substring(A, B):
    """Check if fuzzily matched substring B is present in string A"""


    norm_A = normalized_text(A)
    norm_B = normalized_text(B)

    return norm_B in norm_A


def binarize_overlap(A, B):
    if check_fuzzy_substring(A, B):
        try:
            norm_A = normalized_text(A)
            norm_B = normalized_text(B)
            mask = ['M' if a != ' ' else ' ' for a in norm_A]
            idx = norm_A.find(norm_B)
            mask = "".join([m if (i < idx or i > idx + len(norm_B) or m == ' ') else 'R' for i, m in enumerate(mask)])
            mask_binary = [1 if 'R' in m else 0 for m in mask.split(" ")]
            # print(" ".join(norm_A.split(" ")[mask_binary.index(1): mask_binary.index(1) + len(norm_B.split(" "))]), '\n', norm_B)
            assert " ".join(
                norm_A.split(" ")[mask_binary.index(1): mask_binary.index(1) + len(norm_B.split(" "))]) == norm_B
            return mask_binary
        except AssertionError:
            print(" ".join(norm_A.split(" ")[mask_binary.index(1): mask_binary.index(1) + len(norm_B.split(" "))]),
                  '\n', norm_B)
            return mask_binary
    else:
        return None


def normalize_responses(example, idx, shuffle=False):
    # Normalize the responses
    if example[f'response_{idx}'].startswith(('[/INST]', 'Answer:', 'Option')):
        example[f'response_{idx}'] = example[f'response_{idx}'].replace('[/INST] ', '').replace('Answer: ', '').replace(
            'Option ', '')

    if example[f'response_{idx}'].lower().startswith(('(a)', 'a)')):
        example[f'normalized_response_{idx}'] = 0 if shuffle else 1
    elif example[f'response_{idx}'].lower().startswith(('(b)', 'b)', 'no')):
        example[f'normalized_response_{idx}'] = 1 if shuffle else 0
    elif example[f'response_{idx}'].lower().startswith('yes'):
        example[f'normalized_response_{idx}'] = 1
    elif example[f'response_{idx}'].lower().startswith('no'):
        example[f'normalized_response_{idx}'] = 0
    elif '(a)' in example[f'response_{idx}'].lower() and '(b)' not in example[f'response_{idx}'].lower():
        example[f'normalized_response_{idx}'] = 0 if shuffle else 1
    elif '(b)' in example[f'response_{idx}'].lower() and '(a)' not in example[f'response_{idx}'].lower():
        example[f'normalized_response_{idx}'] = 1 if shuffle else 0
    elif 'does contain evidence for' in example[f'response_{idx}'].lower():
        example[f'normalized_response_{idx}'] = 1
    elif 'does not contain evidence for' or 'does not contain any evidence for' in example[f'response_{idx}'].lower():
        example[f'normalized_response_{idx}'] = 0
    else:
        example[f'normalized_response_{idx}'] = 'N/A'

    return example


def create_nested_list(input_list):
    range_list = [a for a in range(len(input_list)) if input_list[a] == 1]
    nested_list = []
    current_list = []
    last_num = range_list[0] - 1
    for num in range_list:
        if num > last_num + 1:
            nested_list.append(current_list)
            current_list = []
        current_list.append(num)
        last_num = num
    nested_list.append(current_list)
    return nested_list


def create_rationale_list(input_list, sentence):
    outer_list = create_nested_list(input_list)
    rationale_list = []
    for inner_list in outer_list:
        rationale = " ".join([sentence[a] for a in inner_list])
        rationale_list.append(rationale)
    return rationale_list


def fix_syntax(example, traceback, rationales_key):
    if 'unexpected EOF while parsing' in traceback.format_exc():
        # print('unexpected EOF while parsing')
        idx_start = example[rationales_key].find('[')
        idx_end = example[rationales_key].rfind(']')
        if idx_end == -1:
            example[rationales_key] = example[rationales_key][idx_start:] + ']'
        else:
            example[rationales_key] = example[rationales_key][idx_start:idx_end + 1]
    elif 'EOL while scanning string literal' in traceback.format_exc():
        # print('EOL while scanning string literal')
        idx_start = example[rationales_key].find('[')
        idx_end = example[rationales_key].rfind('}')
        example[rationales_key] = example[rationales_key][idx_start:idx_end + 1] + ']'
        example[rationales_key] = example[rationales_key].replace('”\n', '"\n')
    elif 'unexpected character after line continuation character' in traceback.format_exc():
        # print('unexpected character after line continuation character')
        example[rationales_key] = example[rationales_key].replace('\\n', '\n').replace('\\"', '"').replace('""', '"')
    elif 'invalid non-printable character' in traceback.format_exc():
        # print('invalid non-printable character')
        if '\xa0' in example[rationales_key]:
            example[rationales_key] = example[rationales_key].replace('\xa0', '') + '}\n]'
    elif "closing parenthesis ']' does not match opening parenthesis '{'" in traceback.format_exc():
        example[rationales_key] = example[rationales_key] + '}\n]'
    else:
        idx_start = example[rationales_key].find('[')
        idx_end = example[rationales_key].rfind(']')
        if idx_end == -1:
            example[rationales_key] = example[rationales_key][idx_start:] + ']'
        else:
            example[rationales_key] = example[rationales_key][idx_start:idx_end + 1]

    return example


# Alignment utils
def align_rationale_to_model(tokens_rationale, tokens_model, scores_rationale, pool_func=np.max):
    a2b, b2a = tokenizations.get_alignments(tokens_rationale, tokens_model)

    tokens_rationale = np.array(tokens_rationale)
    tokens_model = np.array(tokens_model)
    scores_rationale = np.array(scores_rationale)

    scores_vector = np.zeros(len(tokens_model))
    tokens_vector = np.array([None] * len(tokens_model))

    tokens_merged, scores_merged = [], []

    #  print(len(a2b),len(scores_rationale))
    assert len(a2b) == len(scores_rationale)

    for i, ind in enumerate(a2b):
        if scores_rationale[i] == 1:
            scores_vector[ind] = 1
            tokens_vector[ind] = tokens_rationale[i]

    # print(' '.join(tokens_rationale))
    # print(' '.join([''.join(x) for x in tokens_merged]), '\n')

    return tokens_vector, scores_vector


def align_model_to_rationale(tokens_spacy, tokens_model, scores_model, pool_func=np.max):
    a2b, b2a = tokenizations.get_alignments(tokens_spacy, tokens_model)

    tokens_model = np.array(tokens_model)
    scores_model = np.array(scores_model)

    tokens_merged = []
    scores_merged = []

    for ind in a2b:
        #    print(ind, np.array(scores_model)[ind])
        tokens_merged.append(np.array(tokens_model)[ind])
        scores_merged.append(np.array(scores_model)[ind])

    scores_merged = np.array([pool_func(x) if len(x) > 0 else 0 for x in scores_merged])
    #  print(np.sum(scores_merged), np.sum(scores_model0))

    # print(' '.join(tokens_spacy))
    # print(' '.join([''.join(x) for x in tokens_merged]))

    return tokens_merged, scores_merged
