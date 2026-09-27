"""m1 - The flow encoder: English request -> flowchart.

    vocab.py     the label spaces: step kinds, settings, edge labels, span roles
    encoder.py   the swappable text encoder; the only file that knows about BERT
    data.py      corpus record -> tensors, aligning character spans to tokens
    model.py     encoder + slot, setting, edge, span and mode heads
    decode.py    model output -> IR2 flow
    evaluate.py  structure accuracy, and end-to-end replay of the stored scenarios
    train.py     the training loop

The encoder is behind one small interface (encoder.py). The POC runs on a
pretrained BERT, as a temporary stand-in. The plan is to replace it with our
own encoder trained from scratch, and nothing outside encoder.py should
need to change when that happens.
"""
