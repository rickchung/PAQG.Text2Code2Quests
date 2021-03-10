"""
Clean and preprocess some datasets for the study.
This script is not about producing data for an nn.Module.
You can find more detail for that case in `nn_data_processor.py`.
"""

from pathlib import Path


def serialize_tex(path_to_tex):
    """
    Serialize a tex file and keep only texts of our interest.
    """

    def _get_within_brackets(text: str):
        return text[text.index('{') + 1:text.index('}')]

    def _cleaned(text: str):
        return text.lower().strip()

    with open(path_to_tex, 'r') as fin:
        lines = fin.readlines()

    blocks = []
    indexes = []
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
                blocks.append(f'code: {code}')
            elif next_cleaned.startswith('\\begin{stdout}'):
                stdout = [lines[i + 1]]  # the first line of the stdout
                i += 2
                while _cleaned(lines[i]) != '\\end{stdout}':
                    stdout.append(lines[i])
                    i += 1
                blocks.append(f'stdout: {stdout}')
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

    return blocks


path_thinkjava2 = Path("Datasets", "ThinkJava2-master")
path_chapters = {
    'variable': Path(path_thinkjava2, 'ch02.tex'),
    'io': Path(path_thinkjava2, 'ch03.tex'),
    'methods_testing': Path(path_thinkjava2, 'ch04.tex'),
    'conditionals': Path(path_thinkjava2, 'ch05.tex'),
    'loops_strings': Path(path_thinkjava2, 'ch06.tex'),
    'arrays_references': Path(path_thinkjava2, 'ch07.tex'),
}

# Retrieve index terms
ch02 = serialize_tex(path_chapters['variable'])
