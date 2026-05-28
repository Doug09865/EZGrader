"""
export_onnx.py
--------------
Converts a trained Keras card grader model to ONNX format
for deployment on GitHub Pages (runs in-browser via ONNX Runtime Web).

Usage:
    python export_onnx.py --model saved/best_mobilenet.keras --output ../docs/card_grader.onnx

The exported .onnx file is loaded by docs/app.js using ONNX Runtime Web,
allowing free browser-side inference with no server required.
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import tensorflow as tf
    import tf2onnx
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install tensorflow tf2onnx onnx")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------
def export_model(
    keras_model_path: str,
    output_onnx_path: str,
    img_size: tuple = (224, 224),
    opset: int = 13
) -> str:
    """
    Load a trained Keras model and export it to ONNX.

    Args:
        keras_model_path:  path to .keras or .h5 model file
        output_onnx_path:  output path for .onnx file
        img_size:          (height, width) the model was trained with
        opset:             ONNX opset version (13 is widely supported)

    Returns:
        path to exported ONNX file
    """
    print(f"Loading model: {keras_model_path}")
    
    # Load custom objects (grade_mae metric)
    def grade_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred)) * 10.0

    model = tf.keras.models.load_model(
        keras_model_path,
        custom_objects={'grade_mae': grade_mae}
    )

    print(f"Model loaded: {model.name}")
    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # Define input signature for ONNX conversion
    input_signature = [
        tf.TensorSpec(
            shape=[None, img_size[0], img_size[1], 3],
            dtype=tf.float32,
            name='card_image'
        )
    ]

    # Convert to ONNX
    print(f"\nConverting to ONNX (opset {opset})...")
    os.makedirs(os.path.dirname(output_onnx_path) or '.', exist_ok=True)

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=output_onnx_path,
    )

    # Verify the model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)

    # Report file size
    size_bytes = os.path.getsize(output_onnx_path)
    size_mb    = size_bytes / (1024 * 1024)
    print(f"\n✅ ONNX export complete!")
    print(f"   Output path : {output_onnx_path}")
    print(f"   File size   : {size_mb:.1f} MB")

    if size_mb > 50:
        print(f"\n⚠️  File is {size_mb:.0f}MB — consider quantizing for faster browser loads.")
        print(f"   Run: python export_onnx.py --quantize to reduce size by ~4x")

    # Validate with a dummy inference
    print("\nValidating with dummy inference...")
    _validate_onnx(output_onnx_path, img_size)

    return output_onnx_path


# ---------------------------------------------------------------------------
# ONNX Runtime validation
# ---------------------------------------------------------------------------
def _validate_onnx(onnx_path: str, img_size: tuple):
    """Run a quick inference test on the exported ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (skipping validation — install onnxruntime to validate)")
        print("  pip install onnxruntime")
        return

    sess    = ort.InferenceSession(onnx_path)
    dummy   = np.random.rand(1, img_size[0], img_size[1], 3).astype(np.float32)
    inp_name = sess.get_inputs()[0].name

    result = sess.run(None, {inp_name: dummy})
    output = result[0][0]  # Shape: (4,) — [centering, corners, edges, surface]

    scores = {
        'centering': float(output[0] * 10),
        'corners':   float(output[1] * 10),
        'edges':     float(output[2] * 10),
        'surface':   float(output[3] * 10),
    }

    print(f"  Test inference passed!")
    print(f"  Sample output: {scores}")
    print(f"  (These are random — train the model for real predictions)")


# ---------------------------------------------------------------------------
# Optional: Quantize for smaller file size
# ---------------------------------------------------------------------------
def quantize_model(onnx_path: str, output_path: str = None) -> str:
    """
    Apply dynamic quantization to reduce model size by ~4x.
    This slightly reduces accuracy but dramatically improves load time.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("Install onnxruntime for quantization:")
        print("  pip install onnxruntime")
        return onnx_path

    if output_path is None:
        base = os.path.splitext(onnx_path)[0]
        output_path = f"{base}_quantized.onnx"

    print(f"Quantizing model to {output_path}...")
    quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)

    orig_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    quant_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Original  : {orig_mb:.1f} MB")
    print(f"  Quantized : {quant_mb:.1f} MB")
    print(f"  Reduction : {(1 - quant_mb/orig_mb)*100:.0f}%")

    return output_path


# ---------------------------------------------------------------------------
# Export metadata sidecar
# ---------------------------------------------------------------------------
def export_metadata(onnx_path: str, img_size: tuple, model_name: str):
    """
    Write a JSON metadata file alongside the ONNX model.
    This is read by the browser app to know how to interpret outputs.
    """
    metadata = {
        'model_name':     model_name,
        'img_size':       list(img_size),
        'input_name':     'card_image',
        'output_dims':    ['centering', 'corners', 'edges', 'surface'],
        'output_range':   '0-1 (multiply by 10 for 0-10 grade scale)',
        'grade_weights':  {'centering': 0.20, 'corners': 0.25, 'edges': 0.25, 'surface': 0.30},
        'scales_supported': ['PSA', 'Beckett (BGS)', 'SGC', 'CGC'],
        'version':        '1.0.0',
    }

    meta_path = os.path.splitext(onnx_path)[0] + '_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata written: {meta_path}")
    return meta_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export trained card grader to ONNX')
    parser.add_argument('--model',    default='saved/best_mobilenet.keras',
                        help='Path to trained .keras model file')
    parser.add_argument('--output',   default='../docs/card_grader.onnx',
                        help='Output path for .onnx file')
    parser.add_argument('--img-size', default=224, type=int,
                        help='Image size the model was trained with (default: 224)')
    parser.add_argument('--opset',    default=13, type=int,
                        help='ONNX opset version (default: 13)')
    parser.add_argument('--quantize', action='store_true',
                        help='Also export a quantized (smaller) version')
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)

    onnx_path = export_model(
        keras_model_path=args.model,
        output_onnx_path=args.output,
        img_size=img_size,
        opset=args.opset,
    )

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    export_metadata(onnx_path, img_size, model_name)

    if args.quantize:
        quantize_model(onnx_path)
