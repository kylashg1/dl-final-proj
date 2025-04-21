from argparse import ArgumentParser
import tensorflow as tf

def parse_args():
    """
    Command-line argument parsing
    """
    parser = ArgumentParser(description='“Transforming” Protein sequences into images humans can understand!')
    parser.add_argument('--data', required=True, type=str, help='File path to data csv')
    parser.add_argument('--epochs', required=True, type=int, help='Number of epochs used in training')
    return parser.parse_args()

def main(args):
    print(f'Data CSV file path: {args.data}')
    print(f'Epochs: {args.epochs}')
    print(f'Num GPUs Available: {len(tf.config.list_physical_devices("GPU"))}')

if __name__ ==  "__main__":
    main(parse_args())
