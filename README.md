# VSIBench BBox Evaluation

This directory evaluates VSIBench questions with bbox-only scene inputs.

The normal workflow is:

1. deploy the model with `run_deploy.sh`
2. convert SpatialEncoder bbox outputs with `convert.py`
3. run inference with `eval.py`
4. compute metrics with `acc.py`

## 1. Deploy Model

Run from the evaluation directory:

```bash
cd evaluation_commit/evaluation

bash run_deploy.sh /path/to/model
```

`run_deploy.sh` starts two SGLang instances by default:

```text
ports: 23000,23001
TP_SIZE: 1
NUM_INSTANCES: 2
```

If `CUDA_VISIBLE_DEVICES` is already set, the script shards those GPUs across the two instances. Keep this terminal running while evaluation is active.

## 2. Convert BBox Results

`eval.py` expects one bbox JSON dict:

```json
{
  "scene0025_01": [
    {
      "bbox_3d": [x, y, z, x_size, y_size, z_size, roll, pitch, yaw],
      "label": "chair"
    }
  ]
}
```

For SpatialEncoder result directories shaped like:

```text
results_root/
  scene0025_01/
    merged_bbox_3dconf_top5meanwhl1r_timestamp.json
  scene0153_00/
    merged_bbox_3dconf_top5meanwhl1r_timestamp.json
```

convert them with:

```bash
python convert.py \
  --input_dir /scratch/ayuille1/qichen/results/scannet_spatialencoder_ckpt64_cfgmatch_fa_false \
  --output_json /scratch/ayuille1/jchen293/wcloong/spatialcode_workspace/training_commit/data/scannet_spatialencoder_ckpt64_cfgmatch_fa_false_bbox.jsonl
```

The merged SpatialEncoder files store 10D boxes:

```text
[x, y, z, w, h, length, qw, qx, qy, qz]
```

`convert.py` converts them to the 9D yaw format used by `eval.py`:

```text
[x, y, z, w, h, length, 0.0, 0.0, yaw]
```

By default, instance suffixes are removed from labels, so `chair_1` becomes `chair`. To keep labels unchanged:

```bash
python convert.py \
  --input_dir /path/to/results_root \
  --output_json /path/to/output_bbox.jsonl \
  --keep_instance_suffix
```

## 3. Run Evaluation

Example for all datasets:

```bash
python eval.py \
  --model_name qwen3 \
  --node <node>,<node> \
  --port 23000,23001 \
  --output_path results/output.jsonl \
  --qa_json ../data/vsibench_qa.json \
  --bbox_json ../data/code.jsonl \
  --include all \
  --dataset scannet \
  --concurrency 128
```

`eval.py` writes JSONL during inference and converts it to JSON at the end:

```text
results/scannet_spatialencoder_eval.json
```

### Dataset Filtering

Use `--dataset` to evaluate only selected dataset sources:

```bash
--dataset scannet
--dataset arkitscenes
--dataset scannetpp
--dataset scannet,scannetpp
--dataset all
```

### Task Filtering

Use `--include all` for every question type.

## 4. Compute Metrics

Run:

```bash
python acc.py \
  -f results/scannet_spatialencoder_eval.json
```

`acc.py` prints:

- the selected/all dataset aggregate score
- separate scores for `arkitscenes`, `scannet`, and `scannetpp`
- empty prediction counts

It also writes:

```text
vsibench_combined_results.csv
```

You can restrict metric computation to one dataset:

```bash
python acc.py \
  -f results/scannet_spatialencoder_eval.json \
  -d scannet
```

## Notes

- `eval.py` reads bbox only from `--bbox_json`; it does not read bbox fields from the QA file.
- `eval.py` filters bboxes by the QA record's `key_object` field when available. If `key_object` is empty or missing, all bboxes for that scene are passed to the model.
- The QA file should usually be `training_commit/data/vsibench_qa.json`.
- The bbox file can use a `.jsonl` suffix, but its content must be one JSON dict, not line-delimited records.
