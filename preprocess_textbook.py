"""
Preprocess the textbook "ThinkJava2" (https://github.com/ChrisMayfield/ThinkJava2)
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import Dataset


def get_qg_textbook(chapters=(), no_code=True):
    """
    Load the default textbook content.
    """
    textbook = pd.read_json('data/thinkjava2.json')
    data = Dataset.from_pandas(textbook)

    if chapters:
        data = data.filter(lambda ex: (ex['chapter'] in chapters))
        if no_code:
            data = data.filter(lambda ex: (len(ex['code'].strip()) == 0))

    return data


def extract_keywords(example):
    """
    Transform keywords like {\\bf key} or \\java{key} into a phrase like <hl> key <hl>.
    """
    java_code = []
    key_phrases = []
    regex_kws = r'\w"\';+\-*/%=.\\()&\|!>< '

    # Match Java code
    regex2 = re.compile(r'\\java[ ]?{([' + regex_kws + r']+)}')
    java_code += regex2.findall(example)
    example = regex2.sub(r'\1', example)

    # Highlight key phrases
    for i in [r'{\\bf ([' + regex_kws + r']+)}', r'``([' + regex_kws + r',]+)\'\'']:
        regex1 = re.compile(i)
        key_phrases += regex1.findall(example)
        example = regex1.sub(r'\1', example)

    # Remove control code
    for i in [r'{\\em[ ]?(.*)}', r'\\it[ ]?{(.*)}', r'\\url[ ]?{(.*)}', r'\\ref[ ]?{(.*)}']:
        regex_rm = re.compile(i)
        example = regex_rm.sub(r'\1', example)

    return {'cleaned': example, 'java_code': java_code, 'key_phrases': key_phrases}


def serialize_tex(path_to_tex):
    """
    Serialize a tex file and keep only texts of our interest.
    """

    re_javakw = re.compile(r'\\java{([\w]+)}')

    with open(path_to_tex, 'r') as fin:
        lines = fin.readlines()

    def _get_within_brackets(text: str):
        return text[text.index('{') + 1:text.index('}')]

    def _cleaned(text: str):
        return text.lower().strip()

    blocks = []
    indexes = []
    java_kws = []
    i = -1
    while True:
        i += 1
        if i >= len(lines):
            break

        # Read the next line
        next_line = lines[i]
        next_cleaned = _cleaned(next_line)
        # If it's empty, continue
        if not next_cleaned or next_cleaned.startswith('%'):
            continue

        # \chapter or \section
        if next_cleaned.startswith('\\chapter'):
            chapter = _get_within_brackets(next_cleaned)
            blocks.append(f'chapter: {chapter}')
        elif next_cleaned.startswith('\\section'):
            section = _get_within_brackets(next_cleaned)
            blocks.append(f'section: {section}')
        # \begin, \end
        elif next_cleaned.startswith('\\begin{'):
            if next_cleaned.startswith('\\begin{code}') or next_cleaned.startswith('\\begin{trinket}'):
                code = [lines[i + 1]]  # the first line of the code
                i += 2
                while _cleaned(lines[i]) not in ('\\end{code}', '\\end{trinket}'):
                    code.append(lines[i])
                    i += 1
                blocks.append(f'code: {"".join(code)}')
            elif next_cleaned.startswith('\\begin{stdout}'):
                stdout = [lines[i + 1]]  # the first line of the stdout
                i += 2
                while _cleaned(lines[i]) != '\\end{stdout}':
                    stdout.append(lines[i])
                    i += 1
                blocks.append(f'stdout: {"".join(stdout)}')
            else:
                # Skip the whole unknown begin block until the ned of the block
                # Note, this also skips any other nested begin-end blocks.
                block_name = _get_within_brackets(next_cleaned)
                while not _cleaned(lines[i]).startswith(f'\\end{{{block_name}}}'):
                    i += 1
        # \index
        elif next_cleaned.startswith('\\index'):
            index = _get_within_brackets(next_cleaned)
            indexes.append(f'index: {index}')
        # Any other special text starting with a back slash
        elif next_cleaned.startswith('\\'):
            pass
        # Normal text
        else:
            blocks.append(f'context: {next_cleaned[0].upper() + next_cleaned[1:]}')
            java_kws += re_javakw.findall(next_cleaned)

    # Merge "context:" blocks
    tmp = []
    context = []
    i = -1
    while True:
        i += 1
        if i >= len(blocks):
            break
        next_block = blocks[i]
        if next_block.startswith('context:'):
            context.append(next_block.replace('context: ', ''))
        else:
            if context:
                tmp.append('context: ' + ' '.join(context))
                context = []

            tmp.append(next_block)

    blocks = tmp

    # Build examples (code, pre_context, post_context, chapter, section)

    def _get_aftertag_content(x):
        return x[x.index(':') + 1:].strip()

    examples = []
    chapter = section = None
    for i in range(len(blocks)):
        e = blocks[i]
        if e.startswith('chapter:'):
            chapter = _get_aftertag_content(e)
        elif e.startswith('section:'):
            section = _get_aftertag_content(e)
        elif e.startswith('code:'):
            pre_context = _get_aftertag_content(blocks[i - 1])
            post_context = _get_aftertag_content(blocks[i + 1]) if blocks[i + 1].startswith('context') else ''

            meta_precontext = extract_keywords(pre_context)
            meta_postcontext = extract_keywords(post_context)

            examples.append({
                'code': _get_aftertag_content(e),
                'pre_context': pre_context,
                'post_context': post_context,
                'pre_context_cleaned': meta_precontext['cleaned'],
                'pre_context_javacode': meta_precontext['java_code'],
                'pre_context_key_phrases': meta_precontext['key_phrases'],
                'post_context_cleaned': meta_postcontext['cleaned'],
                'post_context_javacode': meta_postcontext['java_code'],
                'post_context_key_phrases': meta_postcontext['key_phrases'],
                'chapter': chapter,
                'section': section,
            })
        elif e.startswith('context:'):
            meta_context = extract_keywords(_get_aftertag_content(e))
            examples.append({
                'code': '',
                'pre_context': _get_aftertag_content(e),
                'post_context': '',
                'pre_context_cleaned': meta_context['cleaned'],
                'pre_context_javacode': meta_context['java_code'],
                'pre_context_key_phrases': meta_context['key_phrases'],
                'chapter': chapter,
                'section': section,
            })

    return examples, indexes, java_kws


if __name__ == '__main__':
    path_thinkjava2 = Path("data", "ThinkJava2-master")
    path_thinkjava2_out = Path('data', 'thinkjava2.json')
    path_thinkjava2_out1 = Path('data', 'thinkjava2_index.json')
    path_thinkjava2_out2 = Path('data', 'thinkjava2_javakw.json')

    path_chapters = {
        'computer_programming': 'ch01.tex',
        'variable': 'ch02.tex',
        'io': 'ch03.tex',
        'methods_testing': 'ch04.tex',
        'conditionals': 'ch05.tex',
        'loops_strings': 'ch06.tex',
        'arrays_references': 'ch07.tex',
        'recursive_methods': 'ch08.tex',
        'immutable_objects': 'ch09.tex',
        'mutable_objects': 'ch10.tex',
        'designing_classes': 'ch11.tex',
        'arrays_of_objects': 'ch12.tex',
        'objects_of_arrays': 'ch13.tex',
        'extending_classes': 'ch14.tex',
        'arrays_of_arrays': 'ch15.tex',
        'reusing_classes': 'ch16.tex',
        'advanced_topics': 'ch17.tex',
    }

    for k, v in path_chapters.items():
        path_chapters[k] = Path(path_thinkjava2, v)

    docs = []
    index = {}
    javakw = {}
    for topic, path in path_chapters.items():
        doc, index1, javakw1 = serialize_tex(path)
        docs += doc
        index[topic] = index1
        javakw[topic] = Counter(javakw1)

    with open(path_thinkjava2_out, 'w+') as fout:
        json.dump(docs, fout, indent=2)
    with open(path_thinkjava2_out1, 'w+') as fout:
        json.dump(index, fout, indent=2)
    with open(path_thinkjava2_out2, 'w+') as fout:
        json.dump(javakw, fout, indent=2)
