"""
Linear evaluation using scikit-learn's LinearSVC.
Replacement for liblinear-based SVM in VCLR/SeCo evaluation pipeline.
"""

import numpy as np
import argparse
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser('svm_perf_sklearn')
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--trainsplit', type=str, required=True)
    parser.add_argument('--valsplit', type=str, required=True)
    parser.add_argument('--num_replica', type=int, default=8)
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()

    # Load training features
    feat_train = []
    feat_train_cls = []

    feat_train.append(np.load(os.path.join(args.output_dir, f'feature_{args.trainsplit}_None.npy')))
    feat_train_cls.append(np.load(os.path.join(args.output_dir, f'feature_{args.trainsplit}_cls_None.npy')))
    vid_num_train = np.load(os.path.join(args.output_dir, f'vid_num_{args.trainsplit}.npy'))
    train_padding_num = vid_num_train[0] % args.num_replica
    if train_padding_num > 0:
        for i in range(train_padding_num, args.num_replica):
            feat_train[i] = feat_train[i][:-1, :]
            feat_train_cls[i] = feat_train_cls[i][:-1]
    feat_train = np.concatenate(feat_train, axis=0).squeeze()
    feat_train_cls = np.concatenate(feat_train_cls, axis=0).squeeze()
    print('feat_train:', feat_train.shape)
    print('feat_train_cls:', feat_train_cls.shape)

    # Load validation features
    feat_val = []
    feat_val_cls = []
    feat_val.append(np.load(os.path.join(args.output_dir, f'feature_{args.valsplit}_None.npy')))
    feat_val_cls.append(np.load(os.path.join(args.output_dir, f'feature_{args.valsplit}_cls_None.npy')))
    vid_num_val = np.load(os.path.join(args.output_dir, f'vid_num_{args.valsplit}.npy'))
    val_padding_num = vid_num_val[0] % args.num_replica
    if val_padding_num > 0:
        for i in range(val_padding_num, args.num_replica):
            feat_val[i] = feat_val[i][:-1, :]
            feat_val_cls[i] = feat_val_cls[i][:-1]
    feat_val = np.concatenate(feat_val, axis=0)
    feat_val_cls = np.concatenate(feat_val_cls, axis=0)
    print('feat_val:', feat_val.shape)
    print('feat_val_cls:', feat_val_cls)

    # Train Linear SVM using scikit-learn
    print(f"Training LinearSVC with C={args.cost}")
    clf = LinearSVC(C=args.cost, max_iter=10000, verbose=1)
    clf.fit(feat_train, feat_train_cls)

    # Save model weights (as npz)
    model_path = os.path.join(args.output_dir, f'linear_svc_c{args.cost}.npz')
    np.savez(model_path, coef=clf.coef_, intercept=clf.intercept_)
    print(f"SVM model saved to {model_path}")

    # Evaluate
    pred = clf.predict(feat_val)
    acc = accuracy_score(feat_val_cls, pred)
    print(f"Validation Accuracy: {acc:.4f}")

    # Save results
    results_path = model_path + ".txt"
    with open(results_path, 'w') as f:
        f.write(f'Validation Accuracy: {acc:.4f}\n')
    print(f"Results saved to {results_path}")
    print('Done')


if __name__ == '__main__':
    main()
