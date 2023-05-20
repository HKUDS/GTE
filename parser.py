import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='algorithm Params')
    parser.add_argument('--k', default=3, type=int, help='layers')
    parser.add_argument('--data', default='sparse_tmall', type=str, help='name of dataset')
    parser.add_argument('--device', default='cpu', type=str, help='device to use')
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')
    return parser.parse_args()
args = parse_args()
