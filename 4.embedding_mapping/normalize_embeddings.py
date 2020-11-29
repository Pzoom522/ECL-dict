import embeddings

import argparse
import sys


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Normalize word embeddings')
    parser.add_argument('actions', choices=['unit', 'center', 'unitdim', 'centeremb'], nargs='*', default=[], help='the actions to perform in order')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input word embedding file (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output word embedding file (defaults to stdout)')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    args = parser.parse_args()

    # Read input embeddings
    f = open(args.input, encoding=args.encoding, errors='surrogateescape')
    words, matrix = embeddings.read(f)

    # Perform normalization actions
    embeddings.normalize(matrix, args.actions)

    # Write normalized embeddings
    f = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')
    embeddings.write(words, matrix, f)


if __name__ == '__main__':
    main()
