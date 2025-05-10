print("PYTHON_TEST_TF: Script starting...")
try:
    print("PYTHON_TEST_TF: Attempting to import TensorFlow...")
    import tensorflow as tf
    print(f"PYTHON_TEST_TF: Successfully imported TensorFlow version: {tf.__version__}")

    # Optional: Try a very simple TensorFlow operation to be more thorough
    print("PYTHON_TEST_TF: Attempting a simple TensorFlow operation...")
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    c = tf.add(a, b)
    print(f"PYTHON_TEST_TF: Result of 1.0 + 2.0 = {c.numpy()}")
    print("PYTHON_TEST_TF: TensorFlow basic operation seems OK.")

except Exception as e:
    print(f"PYTHON_TEST_TF_ERROR: Error importing or using TensorFlow: {e}")
    # Print detailed traceback
    import traceback
    print("PYTHON_TEST_TF_ERROR: Traceback follows:")
    traceback.print_exc()

print("PYTHON_TEST_TF: Script finished.") 