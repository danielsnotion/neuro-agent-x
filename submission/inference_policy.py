import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from models.value_model import ValueModel
import onnxruntime as ort
import numpy as np

ONNX_PATH = "submission/policy.onnx"
STOP_TOKEN = 6


# 🔥 stable softmax
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# 🔧 operations
def apply_op(grid, op):

    if op == 0:
        return np.rot90(grid)
    elif op == 1:
        return np.rot90(grid, 2)
    elif op == 2:
        return np.flip(grid, axis=1)
    elif op == 3:
        return grid  # disable unsafe op
    elif op == 5:
        return grid.T

    return grid


def apply_sequence(grid, seq, max_steps=3):

    for i, op in enumerate(seq):

        if op == STOP_TOKEN:
            break

        if op == -1:
            break

        if i >= max_steps:
            break

        grid = apply_op(grid, op)

    return grid


def beam_decode(probs, input_grid, value_model, beam_width=3, top_k=3):

    seq_len = probs.shape[1]

    beams = [([], 0.0)]  # (sequence, score)

    for t in range(seq_len):   # 🔥 THIS WAS MISSING

        new_beams = []

        for seq, score in beams:

            top_ops = np.argsort(probs[0, t])[::-1][:top_k]

            for op in top_ops:

                new_seq = seq + [int(op)]

                # 🔥 simulate partial execution
                partial_out = apply_sequence(input_grid.copy(), new_seq, max_steps=3)

                # 🔥 value score
                value_score = evaluate_with_model(value_model, input_grid, partial_out)

                # 🔥 policy confidence
                prob = probs[0, t, op]

                new_score = score + 0.7 * value_score + 0.3 * prob

                new_beams.append((new_seq, new_score))

        # 🔥 keep top beams
        new_beams = sorted(
            new_beams,
            key=lambda x: x[1],
            reverse=True
        )[:beam_width]

        beams = new_beams

    return [seq for seq, _ in beams]

def evaluate_output(out, input_grid):

    # prefer meaningful transformations
    diff = np.sum(out != input_grid)

    # avoid trivial identity
    if diff == 0:
        return -100

    # reward structured output
    variance = np.var(out)

    return diff - 0.1 * variance

def evaluate_with_model(value_model, input_grid, output_grid):

    combined = np.zeros((30, 30))

    combined[:input_grid.shape[0], :input_grid.shape[1]] = input_grid
    combined[15:15+output_grid.shape[0], :output_grid.shape[1]] = output_grid

    tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        score = value_model(tensor).item()

    return score

def run(grid):

    sess = ort.InferenceSession(ONNX_PATH)

    padded = np.zeros((30, 30))
    padded[:grid.shape[0], :grid.shape[1]] = grid

    input_tensor = padded.astype(np.float32)[None, :, :]

    logits = sess.run(None, {"grid": input_tensor})[0]
    probs = softmax(logits, axis=2)

    value_model = ValueModel()
    value_model.load_state_dict(torch.load("models/checkpoints/value.pt", map_location="cpu"))
    value_model.eval()

    # 🔥 beam decode
    candidate_seqs = beam_decode(probs, grid, value_model, beam_width=3)

    best_output = grid.copy()
    best_seq = [-1] * 5  # safe default
    best_score = -1e9

    if not candidate_seqs:
        print("⚠️ No candidate sequences, using fallback")
        return grid, best_seq

    

    for seq in candidate_seqs:

        try:
            out = apply_sequence(grid.copy(), seq)
        except Exception:
            continue
        
        value_score = evaluate_with_model(value_model, grid, out)
        # 🔥 neural confidence
        seq_prob = 1.0
        for i, op in enumerate(seq):
            if op == -1:
                break
            seq_prob *= probs[0, i, op]

        score = 0.8 * value_score + 0.2 * seq_prob

        if score > best_score:
            best_score = score
            best_output = out
            best_seq = seq

    return best_output, best_seq

