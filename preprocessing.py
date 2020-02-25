"""Preprocess hdf5 dataset.
"""

import argparse
import h5py
import progressbar


def save_dataset(dataset, output):
    annos = h5py.File(dataset, 'r')
    features = annos['features']
    total = features.shape[0]
    
    h5file = h5py.File(output, "w")
    d_no_mlp = h5file.create_dataset(
        "no_mlp", (total, 10, 125), dtype='f')
    d_mlp = h5file.create_dataset(
        "mlp", (total, 1250), dtype='f')
    d_conditions = h5file.create_dataset(
        "conditions", (total, 11), dtype='f')

    # Iterate and save all the questions and images.
    bar = progressbar.ProgressBar(maxval=total)
    bar.start()
    for i in range(0, total, 10000):
        if i+10000 < total:
            d_mlp[i:i+10000] = features[i:i+10000, 11:]
            d_no_mlp[i:i+10000] = features[i:i+10000, 11:].reshape(10000, 10, -1) 
            d_conditions[i:i+10000] = features[i:i+10000, :11]
        else:
            d_mlp[i:] = features[i:, 11:]
            d_no_mlp[i:] = features[i:, 11:].reshape(total - i, 10, -1) 
            d_conditions[i:] = features[i:, :11]
        bar.update(i)
    h5file.close()
    annos.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/X_train_KBHhQ0d.h5',
                        help='directory for original trainset')
    parser.add_argument('--output', type=str,
                        default='data/processed_train_dataset.hdf5',
                        help='directory for resized images.')
    args = parser.parse_args()

    save_dataset(args.dataset, args.output)
    print('Wrote dataset to %s' % args.output)