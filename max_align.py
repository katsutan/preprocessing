# %%
import numpy as np
import cupy as cp
import fasttext as ft
from tqdm import tqdm
import time
import sys
import argparse


# %%
def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=str, help="path of tokenized Corpus")
    parser.add_argument(
        "y", type=str, help="path of another tokenized Corpus you want to align")
    parser.add_argument("ft", type=str,
                        help="path of pretrained fasttext model")
    parser.add_argument("--gpu", type=int, default=0,
                        help="-1:cpu 0:gpu  TODO[Implemente to can select gpu device]")
    parser.add_argument("-t", "--threshold", default=0.28, type=float,
                        help="Threshold of word alignment, Word similarity below threshold is not used to calculate sentence similarity")
    return parser.parse_args()


def ft_load(w2v_file):
    print("load fasttext... ", w2v_file, file=sys.stderr)
    return ft.load_model(w2v_file)


def text_load(text_file, max_tl, rev=False):
    print("load text:" + text_file, file=sys.stderr)
    with open(text_file) as f:
        sents = [raw.strip().split(' ') for raw in f]
    sents.sort(key=len, reverse=rev)
    sort_file = "/tmp/" + text_file.split('/')[-1]
    print("save sorted", sort_file, file=sys.stderr)
    with open(sort_file, "w") as f:
        for raw in sents:
            if len(raw) <= max_tl:
                print(' '.join(raw), file=f)
    for i, raw in enumerate(sents):
        if rev:
            if len(raw) <= max_tl:
                return (sents[i:], sents[:i]), (len(sents[i:]), i), sort_file
        else:
            if len(raw) > max_tl:
                return (sents[:i], sents[i:]), (i, len(sents[i:])), sort_file
    return (sents, []), (len(sents), 0), sort_file


def norm_vec(v):
    l2 = np.linalg.norm(v)
    if l2 == 0:
        return v
    return v / l2


args = args_parse()
model = ft_load(args.ft)
print("word dim:" + str(model.get_dimension()), file=sys.stderr)

max_tl = 20
X, len_X, X_sfile = text_load(args.x, max_tl)
Y, len_Y, Y_sfile = text_load(args.y, max_tl, rev=True)
print("X,Y sentences:", len_X, len_Y, file=sys.stderr)
print(X_sfile, Y_sfile, flush=True)  # for next

pad = np.zeros(model.get_dimension())
print("word embedding 1", file=sys.stderr, end=' ', flush=True)
Xv = [[norm_vec(model[w]) for w in ws] for ws in X[0]]
print("2", file=sys.stderr, flush=True)
Yv = [[norm_vec(model[w]) for w in ws] for ws in Y[0]]

# %%

print("GPU:", args.gpu, file=sys.stderr)
xp = cp if args.gpu >= 0 else np
Xbatch = min(200, len_X[0])
Ybatch = min(20000, len_Y[0])
print("X,Y batch:", Xbatch, Ybatch, " threshold:",
      args.threshold, file=sys.stderr)

print("Padding Y", file=sys.stderr)
Yv_set = []
for i in range(0, len_Y[0], Ybatch):
    data_batch = Yv[i: i + Ybatch]
    lmax = max(map(len, data_batch))
    data_padding = np.array(
        [vs + [pad] * (lmax - len(vs)) for vs in data_batch])
    Yv_set.append(data_padding.transpose(0, 2, 1))


print("start Maximum_Alignment", file=sys.stderr)
fin_count = 0
# for i in tqdm(range(X_num_ite)):
pbar = tqdm(total=len_X[0])
while fin_count < len_X[0]:
    assert Xbatch > 0, "too big Ybatch"
    Xv_batch = Xv[fin_count: Xbatch + fin_count]
    Xv_lmax = max(map(len, Xv_batch))
    Xv_pad = xp.array([vs + [pad] * (Xv_lmax - len(vs)) for vs in Xv_batch])
    Z_sim = np.empty((Xv_pad.shape[0], 0), "f")
    pbar.set_description("[Batch %d]" % Xv_pad.shape[0])
    try:
        for Yv_pad in Yv_set:
            XdYa = xp.dot(Xv_pad, xp.asarray(Yv_pad))
            XdY_ro = XdYa.transpose(0, 2, 1, 3)
            XY_sims = xp.max(XdY_ro, 3)
            YX_sims = xp.max(XdY_ro, 2)
            if args.gpu >= 0:
                XY_sims = cp.asnumpy(XY_sims)
                YX_sims = cp.asnumpy(YX_sims)
            XY_sims = np.ma.masked_array(
                XY_sims, mask=XY_sims < args.threshold)
            XY_means = np.mean(XY_sims, axis=2)
            YX_sims = np.ma.masked_array(
                YX_sims, mask=YX_sims < args.threshold)
            YX_means = np.mean(YX_sims, axis=2)
            Z_tmp = (XY_means + YX_means) / 2
            Z_sim = np.append(Z_sim, Z_tmp, axis=1)
    except cp.cuda.memory.OutOfMemoryError:
        Xbatch -= 1
        continue
    # for sims in Z_sim:
    #     for sim in sims:
    #         print(sim, end=' ')
    #     print()
    np.savetxt(sys.stdout.buffer, Z_sim)
    time.sleep(0.1)
    pbar.update(Xv_pad.shape[0])
    fin_count += Xbatch

pbar.close()
